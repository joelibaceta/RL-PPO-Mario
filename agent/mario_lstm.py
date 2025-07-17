import torch
import torch.nn as nn

class MarioConvLSTM(nn.Module):
    def __init__(self, obs_shape, lstm_hidden_size, n_actions):
        """
        CNN adaptada con soporte para LSTM según la arquitectura del paper.
        Args:
            obs_shape: (channels, height, width)
            lstm_hidden_size: Tamaño del hidden state del LSTM
            n_actions: Número de acciones posibles
        """
        super(MarioConvLSTM, self).__init__()
        c, h, w = obs_shape

        # Convolution Blocks
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ReLU()
        )

        # Calcula tamaño de la salida de las CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_output = self.feature_extractor(dummy_input)
            self.flat_dim = conv_output.view(1, -1).shape[1]

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.flat_dim, hidden_size=lstm_hidden_size, batch_first=False)

        # Política y Valor
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward_conv(self, x):
        """
        Pasa solo por las capas convolucionales para visualizar filtros.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        elif x.ndim == 4:
            pass  # (batch, C, H, W)
        else:
            raise ValueError(f"[forward_conv] Unexpected x shape: {x.shape}")

        return self.feature_extractor(x)

    def forward(self, x, hidden_state=None): 

        if x.ndim == 4:
            # (batch, C, H, W) → (1, batch, C, H, W)
            x = x.unsqueeze(0)
        elif x.ndim == 5:
            # (steps, batch, C, H, W) OK
            pass
        else:
            raise ValueError(f"[forward] Unexpected x shape: {x.shape}")

        steps, batch_size, C, H, W = x.size()
        x = x.view(steps * batch_size, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(steps, batch_size, -1)

        # Inicializa hidden_state si no se pasa
        if hidden_state is None:
            hidden_state = (
                torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device),
                torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device),
            )

        lstm_out, next_hidden = self.lstm(features, hidden_state)

        logits = self.policy_head(lstm_out)
        value = self.value_head(lstm_out).squeeze(-1)
        return logits, value, next_hidden
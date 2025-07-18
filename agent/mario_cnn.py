import torch
import torch.nn as nn
import torch.nn.functional as F

class MarioCNN(nn.Module):
    def __init__(self, obs_shape, lstm_hidden_size, n_actions):
        """
        CNN ligera basada en la arquitectura clásica de DeepMind.
        Compatible con la firma de MarioConvLSTM.
        """
        super(MarioCNN, self).__init__()
        c, h, w = obs_shape  # obs_shape = (stack, H, W)
        print(f"📺 CNN Input Shape recibido: {obs_shape}")

        # CNN feature extractor (más ligera)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        # Calcula tamaño de salida para las capas lineales
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_out = self.feature_extractor(dummy_input)
            flat_dim = conv_out.shape[1]
            print(f"📐 Flat dim after convs: {flat_dim}")

        # Heads de política y valor (256 unidades)
        self.policy_head = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, hidden_state=None):
        """
        x: (batch, C, H, W)
        hidden_state: None (para compatibilidad con LSTM)
        """
        if x.ndim == 5:
            # (steps, batch, C, H, W) -> (steps*batch, C, H, W)
            steps, batch, C, H, W = x.shape
            x = x.view(steps * batch, C, H, W)

        z = self.feature_extractor(x / 255.0)  # Normalización
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)

        return logits, value, hidden_state  # hidden_state=None por compatibilidad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class MarioCNN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        """
<<<<<<< HEAD
        CNN ligera basada en la arquitectura clásica de DeepMind.
        Compatible con la firma de MarioConvLSTM.
=======
        CNN para PPO (sin canal C).
>>>>>>> c8df178 (last attempt)
        """
        super(MarioCNN, self).__init__()
        c, h, w = obs_shape  # obs_shape = (stack, H, W)

        # CNN feature extractor (más ligera)
        self.feature_extractor = nn.Sequential(
<<<<<<< HEAD
            nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(),
=======
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
>>>>>>> c8df178 (last attempt)
            nn.Flatten()
        )

        # Calcular tamaño para capa densa
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            flat_dim = self.feature_extractor(dummy_input).shape[1]

<<<<<<< HEAD
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
=======
        # Heads de política y valor
        self.policy_head = nn.Linear(flat_dim, n_actions)
        self.value_head = nn.Linear(flat_dim, 1)

    def forward(self, x):
>>>>>>> c8df178 (last attempt)
        """
        x: (batch, stack, H, W)
        """
        x = x / 255.0  # Normaliza
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
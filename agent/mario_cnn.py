import torch
import torch.nn as nn

class MarioCNN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape  # obs_shape = (stack, H, W)
        print(f"📺 CNN Input Shape recibido: {obs_shape}")

        self.feature = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), nn.ReLU(),  # + más filtros
            nn.Flatten()
        )   

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.feature(dummy).shape[1]

        self.policy = nn.Sequential(
            nn.Linear(flat_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(flat_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        z = self.feature(x / 255.0)  # Normaliza solo una vez
        logits = self.policy(z)
        value = self.value(z).squeeze(-1)
        return logits, value
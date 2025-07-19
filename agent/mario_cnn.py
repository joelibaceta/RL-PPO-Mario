import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class MarioCNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(MarioCNN, self).__init__()  # 👈 IMPORTANTE

        c, h, w = input_shape

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcula tamaño de salida de feature_extractor dinámicamente
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.feature_extractor(dummy_input)
            linear_input_size = dummy_output.shape[1]

        self.policy_head = nn.Linear(linear_input_size, n_actions)
        self.value_head = nn.Linear(linear_input_size, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        policy_logits = self.policy_head(features)
        state_value = self.value_head(features)
        return policy_logits, state_value
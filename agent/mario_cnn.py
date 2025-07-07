import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MarioCNN(BaseFeaturesExtractor):
    """
    Red convolucional robusta para Super Mario.
    - Soporta observaciones (H, W), (C, H, W), o FrameStack.
    - Detecta y añade canales si faltan.
    - Normaliza imágenes uint8 -> float32 [0,1].
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # Forma de la observación
        obs_shape = observation_space.shape
        if len(obs_shape) == 3:
            # (C, H, W)
            n_input_channels, height, width = obs_shape
        elif len(obs_shape) == 2:
            # (H, W) → asumimos canal único
            height, width = obs_shape
            n_input_channels = 1
        else:
            raise ValueError(f"[ERROR] Forma inesperada: {obs_shape}")

        print(f"[DEBUG] Detected input shape: {n_input_channels}x{height}x{width}")

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Calcula tamaño de salida automáticamente
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            if sample.ndim == 4:
                # Ya tiene batch y canales
                pass
            elif sample.ndim == 3:
                # Añadir canal explícito
                sample = sample.unsqueeze(1)
            elif sample.ndim == 2:
                # Añadir batch y canal
                sample = sample.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"[ERROR] Tensor inesperado: {sample.shape}")

            if sample.max() > 1.0:
                sample /= 255.0

            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.dtype == th.uint8:
            observations = observations.float() / 255.0
        if observations.ndim == 3:
            observations = observations.unsqueeze(1)  # añade canal
        return self.linear(self.cnn(observations))

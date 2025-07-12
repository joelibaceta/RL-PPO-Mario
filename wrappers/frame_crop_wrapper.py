import gym
import numpy as np

class FrameCropWrapper(gym.ObservationWrapper):
    """
    Recorta la parte superior del frame y la rellena con el color dominante
    de la primera fila restante. Soporta RGB (H,W,C) y escala de grises (H,W).
    """
    def __init__(self, env, crop_top=30):
        super().__init__(env)
        self.crop_top = crop_top

        shape = self.observation_space.shape
        self.has_channel = len(shape) == 3  # True si (H,W,C), False si (H,W)

        # No cambia el tamaño del espacio de observación
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8
        )

    def observation(self, obs):
        obs_cropped = obs.copy()

        if self.has_channel:
            # Frame RGB
            first_rows = obs_cropped[self.crop_top:self.crop_top+1, :, :]  # (1, W, C)
            # Calcula color dominante
            pixels = first_rows.reshape(-1, first_rows.shape[-1])  # (W, C)
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_color = unique[counts.argmax()]
            # Rellena parte superior
            obs_cropped[:self.crop_top, :, :] = dominant_color
        else:
            # Frame escala de grises
            first_rows = obs_cropped[self.crop_top:self.crop_top+1, :]  # (1, W)
            # Calcula valor dominante
            unique, counts = np.unique(first_rows, return_counts=True)
            dominant_value = unique[counts.argmax()]
            obs_cropped[:self.crop_top, :] = dominant_value

        return obs_cropped
import cv2
import gym
import numpy as np

class ResizeObservation(gym.ObservationWrapper):
    """
    Wrapper que redimensiona y simplifica las observaciones.
    - Redimensiona a un tamaño definido.
    - Reduce la cantidad de colores o mantiene RGB.
    """
    def __init__(self, env, target_shape=(128, 128), color_mode="rgb", num_colors=None):
        """
        Args:
            target_shape: tamaño final (H, W).
            color_mode: 'rgb' mantiene color, 'quantize' reduce paleta.
            num_colors: número de colores si se usa color_mode='quantize'.
        """
        super().__init__(env)
        self.target_shape = target_shape
        self.color_mode = color_mode
        self.num_colors = num_colors

        channels = 3  # Solo RGB
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.target_shape[0], self.target_shape[1], channels),
            dtype=np.uint8
        )

    def observation(self, obs):
        # Redimensiona
        obs = cv2.resize(obs, self.target_shape[::-1], interpolation=cv2.INTER_AREA)

        # Simplifica colores
        if self.color_mode == "quantize" and self.num_colors:
            obs = self.reduce_colors(obs, self.num_colors)

        return obs

    def reduce_colors(self, img, k):
        """Reduce la cantidad de colores usando k-means clustering."""
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_img = res.reshape((img.shape))
        return result_img
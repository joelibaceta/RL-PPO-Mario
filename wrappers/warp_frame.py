import gym
import numpy as np
import cv2
from gym.spaces import Box

class WarpFrame(gym.ObservationWrapper):
    """
    Convierte cada frame RGB a escala de grises y redimensiona a 84×84×1.
    """
    def __init__(self, env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width  = width
        self.height = height
        self.observation_space = Box(
            low=0, high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8
        )

    def observation(self, frame):
        # RGB → grayscale
        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize a (width, height)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Expandir canal
        return resized[:, :, None]
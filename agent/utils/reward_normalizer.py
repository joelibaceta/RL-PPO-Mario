import numpy as np

class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def normalize(self, rewards):
        batch_mean = rewards.mean()
        batch_var = rewards.var()

        # Actualiza estadísticas globales
        self.mean = 0.99 * self.mean + 0.01 * batch_mean
        self.var = 0.99 * self.var + 0.01 * batch_var
        self.count += len(rewards)

        return (rewards - self.mean) / (np.sqrt(self.var) + 1e-8)
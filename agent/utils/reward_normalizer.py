import numpy as np

class RewardNormalizer:
    def __init__(self, epsilon=1e-8, alpha=0.999):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.alpha = alpha

    def normalize(self, rewards):
        rewards = np.clip(rewards, -5.0, 5.0)  # 🚨 Clipping antes de normalizar
        batch_mean = rewards.mean()
        batch_var = rewards.var()

        # Actualiza estadísticas globales más lento
        self.mean = self.alpha * self.mean + (1 - self.alpha) * batch_mean
        self.var  = self.alpha * self.var + (1 - self.alpha) * batch_var
        self.count += len(rewards)

        return (rewards - self.mean) / (np.sqrt(self.var) + 1e-8)
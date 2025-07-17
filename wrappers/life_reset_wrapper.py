# wrappers/life_reset_wrapper.py
import gym

class LifeResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", 2)  # Normalmente 2 o 3 vidas al empezar
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = info.get("lives", self.lives)
        # Si perdió una vida, forzar terminado
        if current_lives < self.lives and current_lives > 0:
            terminated = True
        self.lives = current_lives
        return obs, reward, terminated, truncated, info
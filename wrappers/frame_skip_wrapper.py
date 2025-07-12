import gym

class FrameSkipWrapper(gym.Wrapper):
    """
    Ejecuta la misma acción por `skip` steps consecutivos y devuelve
    la suma de las recompensas. Útil para acelerar entornos como NES-Py.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info
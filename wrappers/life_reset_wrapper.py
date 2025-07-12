# wrappers/life_reset_wrapper.py
import gym

class LifeResetWrapper(gym.Wrapper):
    """
    Wrapper que detecta la pérdida de vidas en Mario y reinicia
    los contadores internos del monitor sin terminar el episodio.
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Obtiene las vidas actuales
        if "lives" in info:
            self.lives = info["lives"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Detecta pérdida de vidas
        if "lives" in info:
            current_lives = info["lives"]
            if self.lives is not None and current_lives < self.lives:
                # Se perdió una vida: reinicia métricas del monitor
                if hasattr(self.env, "writer"):  # Si tiene monitor
                    print("🔄 Reiniciando métricas del monitor tras pérdida de vida")
                    self.env.monitor_episode_reward = 0
                    self.env.monitor_episode_length = 0

            self.lives = current_lives

        return obs, reward, terminated, truncated, info
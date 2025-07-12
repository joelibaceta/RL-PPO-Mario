import time
import gym
from torch.utils.tensorboard import SummaryWriter

class MonitorWrapper(gym.Wrapper):
    def __init__(self, env, log_dir="logs_cleanrl"):
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")
        self.writer.add_text("run_name", run_name)
        print(f"📓 TensorBoard logs en: {log_dir}/{run_name}")

        super().__init__(env)

        # 🔥 Métricas POR EPISODIO (actuales)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_max_x_pos = 0
        
        # 🔥 Métricas ACUMULADAS durante el update
        self.update_episode_rewards = []
        self.update_episode_lengths = []
        self.update_max_x_positions = []
        self.update_deaths = 0
        
        self.global_step = 0
        self.update_count = 0
        self.update_start_time = time.time()

    def reset(self, **kwargs):
        # Solo reinicia métricas del episodio actual
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_max_x_pos = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        # Actualiza métricas del episodio actual
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.global_step += 1

        # 🔥 Trackea la posición MÁXIMA alcanzada (no donde murió)
        current_x_pos = info.get("x_pos", 0)
        self.current_max_x_pos = max(self.current_max_x_pos, current_x_pos)

        if done:
            # 🔥 Guarda métricas del episodio completo
            self.update_episode_rewards.append(self.current_episode_reward)
            self.update_episode_lengths.append(self.current_episode_length)
            self.update_max_x_positions.append(self.current_max_x_pos)
            self.update_deaths += 1

        return obs, reward, terminated, truncated, info

    def end_update(self, num_steps, losses=None):
        elapsed = time.time() - self.update_start_time
        fps = num_steps / elapsed

        # 🔥 Métricas PROMEDIO de todos los episodios en el update
        if self.update_episode_rewards:
            avg_reward = sum(self.update_episode_rewards) / len(self.update_episode_rewards)
            avg_length = sum(self.update_episode_lengths) / len(self.update_episode_lengths)
            max_x_pos = max(self.update_max_x_positions)  # El MEJOR x_pos del update
            
            self.writer.add_scalar("charts/episode_reward", avg_reward, self.update_count)
            self.writer.add_scalar("charts/episode_length", avg_length, self.update_count)
            self.writer.add_scalar("charts/x_pos", max_x_pos, self.update_count)

        # 📊 Deaths normalizadas (deaths por episodio)
        episodes_in_update = len(self.update_episode_rewards) if self.update_episode_rewards else 1
        death_rate = self.update_deaths / episodes_in_update
        self.writer.add_scalar("charts/deaths", death_rate, self.update_count)

        # 📊 Métricas de rendimiento
        self.writer.add_scalar("charts/fps", fps, self.update_count)
        self.writer.add_scalar("charts/update_time_sec", elapsed, self.update_count)

        # 📉 Pérdidas
        if losses:
            self.writer.add_scalar("loss/entropy", losses["entropy"], self.update_count)
            self.writer.add_scalar("loss/policy", losses["policy"], self.update_count)
            self.writer.add_scalar("loss/value", losses["value"], self.update_count)
            self.writer.add_scalar("loss/total", losses["total"], self.update_count)

        # 🔄 Reset contadores del update
        self.update_episode_rewards = []
        self.update_episode_lengths = []
        self.update_max_x_positions = []
        self.update_deaths = 0
        self.update_count += 1
        self.update_start_time = time.time()

    def close(self):
        self.writer.close()
        super().close()
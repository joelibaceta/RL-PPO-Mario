import numpy as np
from collections import deque
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

class RealTimeTrainingMonitor:
    """
    Monitor independiente para seguir el progreso sin callback.
    """
    def __init__(self, log_freq=5000):
        self.log_freq = log_freq
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_x_pos = deque(maxlen=100)
        self.last_log_step = 0
        
    def update(self, info_dict):
        """
        Actualizar con información de un paso de entrenamiento.
        
        Args:
            info_dict: Dict con keys 'timestep', 'episode_reward', 'episode_length', 'x_pos'
        """
        if 'episode_reward' in info_dict:
            self.episode_rewards.append(info_dict['episode_reward'])
        
        if 'episode_length' in info_dict:
            self.episode_lengths.append(info_dict['episode_length'])
            
        if 'x_pos' in info_dict:
            self.episode_x_pos.append(info_dict['x_pos'])
            
        # Log periódico
        timestep = info_dict.get('timestep', 0)
        if timestep - self.last_log_step >= self.log_freq:
            self._print_stats(timestep)
            self.last_log_step = timestep
    
    def _print_stats(self, timestep):
        """Imprimir estadísticas actuales."""
        if len(self.episode_rewards) == 0:
            return
            
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        print(f"\n🎮 MARIO PROGRESS - Step {timestep:,}")
        print(f"📊 Avg Reward: {avg_reward:.2f} (last {len(self.episode_rewards)} episodes)")
        print(f"⏱️ Avg Length: {avg_length:.1f} steps")
        
        if len(self.episode_rewards) >= 2:
            recent_5 = list(self.episode_rewards)[-5:]
            older_5 = list(self.episode_rewards)[-10:-5] if len(self.episode_rewards) >= 10 else []
            
            if older_5:
                trend = np.mean(recent_5) - np.mean(older_5)
                trend_symbol = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                print(f"{trend_symbol} Trend: {trend:+.2f}")
        
        if len(self.episode_x_pos) > 0:
            avg_x = np.mean(self.episode_x_pos)
            max_x = np.max(self.episode_x_pos)
            print(f"🏃 Avg Distance: {avg_x:.1f}, Max: {max_x:.1f}")
        
        print("="*50)
import numpy as np
from collections import Counter
import torch

def log_metrics(writer, update, rewards_buf, max_x_pos, info, dist_entropy, action_counter, action_space_n, loss, annealed_temperature):
    """
    Loguea métricas y pérdidas en TensorBoard.

    :param writer: SummaryWriter
    :param update: número de update actual
    :param rewards_buf: buffer de rewards
    :param max_x_pos: máximo X alcanzado
    :param info: diccionario con flag_get
    :param dist_entropy: entropía media de la política
    :param action_counter: Counter con acciones elegidas
    :param action_space_n: número total de acciones
    :param loss: valor actual de la pérdida total
    """
    # Proporción de acciones
    total_actions = sum(action_counter.values())
    for action_id in range(action_space_n):
        count = action_counter[action_id]
        proportion = count / total_actions if total_actions > 0 else 0.0
        writer.add_scalar(f"actions/action_{action_id}_proportion", proportion, update)

    # Reward medio
    avg_reward = np.mean(rewards_buf)
    max_x_pos_update = max_x_pos.max()
    avg_x_pos_update = max_x_pos.mean()
    flag_get = np.array(info["flag_get"]).astype(np.float32).mean()

    # Logueo
    writer.add_scalar("metrics/flag_get", flag_get, update)
    writer.add_scalar("metrics/policy_entropy", dist_entropy, update)
    writer.add_scalar("charts/max_x_pos", max_x_pos_update, update)
    writer.add_scalar("charts/avg_x_pos", avg_x_pos_update, update)
    writer.add_scalar("charts/episode_reward", avg_reward, update)
    writer.add_scalar("loss/total", loss, update)
    writer.add_scalar("loss/annealed_temperature", annealed_temperature, update)
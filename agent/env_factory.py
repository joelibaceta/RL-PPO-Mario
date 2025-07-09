import mo_gymnasium as mo_gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation
from gymnasium.spaces import Box, Discrete
import gymnasium as gym
import numpy as np
from agent.pre_process_pipeline import EnvPreprocessingPipeline


class MoMarioActionWrapper(gym.ActionWrapper):
    """
    Wrapper to map SIMPLE_MOVEMENT actions to mo-gymnasium action space.
    This provides compatibility with the original action space expectations.
    """
    
    # Define simplified action mapping (similar to SIMPLE_MOVEMENT)
    SIMPLE_ACTIONS = [
        0,    # NOOP
        1,    # Right
        2,    # Right + A (run/jump)
        3,    # Right + B (fireball)
        4,    # Right + A + B
        5,    # A (jump)
        6,    # Left
    ]
    
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(len(self.SIMPLE_ACTIONS))
    
    def action(self, action):
        # Map simplified action to mo-gymnasium action
        return self.SIMPLE_ACTIONS[action]


class MoMarioRewardWrapper(gym.RewardWrapper):
    """
    Wrapper to handle multi-objective rewards from mo-gymnasium.
    Converts the 5D reward vector to a single scalar reward.
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # mo-gymnasium returns 5D reward: [x_pos, time, death, coin, enemy]
        # We primarily care about x-position progress (first component)
        if isinstance(reward, (list, tuple, np.ndarray)) and len(reward) > 1:
            # Weighted combination: prioritize x-position, penalize time, reward coins
            x_pos, time, death, coin, enemy = reward
            return x_pos - 0.1 * time + coin * 10 + enemy * 5 - death * 100
        return reward


class MarioEnvFactory:
    """
    Factory for creating Super Mario Bros environments.

    - Uses a default preprocessing pipeline (Resize, Grayscale, FrameStack).
    - Allows passing a custom pipeline to override defaults.
    - Always normalizes observation space as a final step unless disabled.
    - Supports additional wrappers after preprocessing.
    """

    def __init__(
        self,
        world: str = "mo-supermario-v0",
        actions=None,
        render: bool = False,
        preprocess_pipeline: EnvPreprocessingPipeline = None,
        custom_wrappers: list = None,
        auto_normalize: bool = True,
    ):
        """
        :param world: Mo-gymnasium world version (e.g., "mo-supermario-v0").
        :param actions: Deprecated parameter for compatibility (mo-gymnasium handles actions internally).
        :param render: If True, enables rendering in the environment.
        :param preprocess_pipeline: EnvPreprocessingPipeline object. If None, uses default pipeline.
        :param custom_wrappers: List of callables(env) -> env applied after preprocessing.
        :param auto_normalize: Automatically normalize observation space to [0.0, 1.0].
        """
        self.world = world
        self.actions = actions  # Kept for compatibility but not used
        self.render = render
        self.preprocess_pipeline = preprocess_pipeline or self.default_pipeline()
        self.custom_wrappers = custom_wrappers or []
        self.auto_normalize = auto_normalize

    def default_pipeline(self):
        """
        Returns a pipeline with: Grayscale, FrameStack, Normalize.
        Using simpler processing to avoid opencv dependency.
        """
        builder = EnvPreprocessingPipeline()
        # Skip ResizeObservation for now to avoid opencv dependency
        builder.add(GrayscaleObservation, keep_dim=False)
        builder.add(FrameStackObservation, stack_size=4)
        builder.add(self._normalize_observation_space)
        return builder

    def make_base_factory(self, render_mode=None):
        """
        Returns a factory that creates a raw mo-gymnasium Mario environment.
        """
        def _init():
            env = mo_gym.make(self.world, render_mode=render_mode)
            # Apply action and reward wrappers for compatibility
            env = MoMarioActionWrapper(env)
            env = MoMarioRewardWrapper(env)
            return env
        return _init

    def _normalize_observation_space(self, env):
        """
        Normalizes the observation space to [0.0, 1.0].
        """
        orig_space = env.observation_space
        env.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=orig_space.shape,
            dtype=np.float32
        )
        return env

    def make(self):
        """
        Returns a callable that creates a fresh Mario environment.
        """
        # Wrap the base factory with preprocessing pipeline
        factory = self.preprocess_pipeline.build(self.make_base_factory())

        def _init():
            env = factory()  # create preprocessed env

            # Apply additional custom wrappers
            for wrapper in self.custom_wrappers:
                env = wrapper(env)

            # Patch reset to maintain (obs, info) tuple
            original_reset = env.reset

            def reset(*args, **kwargs):
                kwargs.pop("seed", None)  # Remove unsupported kwargs
                kwargs.pop("options", None)
                obs_info = original_reset(*args, **kwargs)
                if isinstance(obs_info, tuple) and len(obs_info) == 2:
                    return obs_info
                return obs_info, {}

            env.reset = reset

            return env

        return _init
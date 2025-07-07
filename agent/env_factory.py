import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack
from gym.spaces import Box
import numpy as np
from agent.pre_process_pipeline import EnvPreprocessingPipeline


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
        world: str = "SuperMarioBros-v0",
        actions=None,
        render: bool = False,
        preprocess_pipeline: EnvPreprocessingPipeline = None,
        custom_wrappers: list = None,
        auto_normalize: bool = True,
    ):
        """
        :param world: Mario world version (e.g., "SuperMarioBros-v0").
        :param actions: List of allowed actions (defaults to SIMPLE_MOVEMENT).
        :param render: If True, enables rendering in the environment.
        :param preprocess_pipeline: EnvPreprocessingPipeline object. If None, uses default pipeline.
        :param custom_wrappers: List of callables(env) -> env applied after preprocessing.
        :param auto_normalize: Automatically normalize observation space to [0.0, 1.0].
        """
        self.world = world
        self.actions = actions or SIMPLE_MOVEMENT
        self.render = render
        self.preprocess_pipeline = preprocess_pipeline or self.default_pipeline()
        self.custom_wrappers = custom_wrappers or []
        self.auto_normalize = auto_normalize

    def default_pipeline(self):
        """
        Returns a pipeline with: Resize, Grayscale, FrameStack, Normalize.
        """
        builder = EnvPreprocessingPipeline()
        builder.add(ResizeObservation, shape=(80, 75))
        builder.add(GrayScaleObservation, keep_dim=False)
        builder.add(FrameStack, num_stack=4)
        builder.add(self._normalize_observation_space)
        return builder

    def make_base_factory(self, render_mode=None):
        """
        Returns a factory that creates a raw Mario environment.
        """
        def _init():
            env = gym_super_mario_bros.make(self.world, apply_api_compatibility=True, render_mode=render_mode)
            env = JoypadSpace(env, self.actions)
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
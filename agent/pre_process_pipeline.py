class EnvPreprocessingPipeline:
    """
    Builder-style class to construct a preprocessing pipeline for Gym environments.
    """

    def __init__(self):
        self.steps = []

    def add(self, wrapper_func, **kwargs):
        """
        Add a preprocessing step (wrapper).
        :param wrapper_func: The Gym wrapper or callable to apply.
        :param kwargs: Keyword arguments for the wrapper.
        :return: self (for chaining).
        """
        self.steps.append((wrapper_func, kwargs))
        return self

    def build(self, base_factory):
        """
        Build a factory that applies all wrappers to a newly created environment.
        :param base_factory: Callable that creates the base environment.
        :return: A new factory that creates wrapped environments.
        """
        def wrapped_factory():
            env = base_factory()
            for wrapper_func, kwargs in self.steps:
                env = wrapper_func(env, **kwargs)
            return env
        return wrapped_factory
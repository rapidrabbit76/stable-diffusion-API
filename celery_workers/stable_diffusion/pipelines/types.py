import abc


class StableDiffusionCallback(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, **kwds):
        pass

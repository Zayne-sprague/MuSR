from abc import abstractmethod, ABCMeta
from typing import List, Dict, Any, Generator


class Model(metaclass=ABCMeta):

    @abstractmethod
    def inference(self, prompt: str, *args, **kwargs) -> Any:
        """
        Most simple inference to a model that takes a prompt and gets an output object back.

        :param prompt: The prompt to give to the language model
        :param args: If the specific model needs more arguments
        :param kwargs: If the specific model needs more keyword arguments
        :return: Generated response from the language model.
        """
        raise NotImplementedError("All models need an inference call implemented.")

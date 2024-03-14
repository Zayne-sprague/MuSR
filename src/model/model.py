from abc import abstractmethod, ABC
from typing import List, Dict, Any, Generator


class Model(ABC):

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

    @classmethod
    def load_model(cls, name: str, **kwargs):
        from src.model import HFModel, OpenAIModel

        if '/' not in name:
            try:
                return HFModel(name, **kwargs)
            except:
                pass
            raise Exception(f"Error loading the model. Please check the name of the model. {name}")

        tag = name.split('/')[0].strip()
        model_name = name.split('/', 1)[-1]

        if tag == 'openai':

            if 'prompt_cost' not in kwargs and 'completion_cost' not in kwargs:
                prompt_price = 0.0
                completion_price = 0.0
                if model_name == 'gpt-3.5-turbo':
                    prompt_price = 0.001 / 1000
                    completion_price = 0.02 / 1000
                elif model_name == 'gpt-3.5-turbo-0125':
                    prompt_price = 0.0005 / 1000
                    completion_price = 0.0015 / 1000
                elif model_name == 'gpt-3.5-turbo-16k':
                    prompt_price = 0.003 / 1000
                    completion_price = 0.004 / 1000
                elif model_name == 'gpt-4':
                    prompt_price = 0.03 / 1000
                    completion_price = 0.06 / 1000
                elif model_name == 'gpt-4-1106-preview':
                    prompt_price = 0.01 / 1000
                    completion_price = 0.03 / 1000
                elif model_name == 'gpt-3.5-turbo-instruct':
                    prompt_price = 0.0015 / 1000
                    completion_price = 0.0020 / 1000
                    kwargs['api_endpoint'] = 'completion'
                    kwargs['echo'] = False
                elif model_name == 'text-davinci-003':
                    prompt_price = 0.02 / 1000
                    completion_price = 0.02 / 1000
                    kwargs['api_endpoint'] = 'completion'
                    kwargs['echo'] = False
                kwargs['prompt_cost'] = prompt_price
                kwargs['completion_cost'] = completion_price


            return OpenAIModel(model_name, **kwargs)
        elif tag == 'hf':
            return HFModel(model_name, **kwargs)
        else:
            return HFModel(name, **kwargs)


    def parse_out(self, out):
        from src.model import OpenAIModel, HFModel
        if isinstance(self, OpenAIModel):
            if self.api_endpoint == 'completion':
                return [x['text'] for x in out.choices]
            else:
                return [x['message']['content'] for x in out.choices]
        else:
            return out
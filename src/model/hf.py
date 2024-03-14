import os
import itertools
import time
import openai
from datetime import timedelta
import random

from typing import List, Dict, Union, Any, Generator

import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast, AutoModel, AutoTokenizer, Pipeline, AutoModelForCausalLM, PretrainedConfig,\
    AutoConfig, T5ForConditionalGeneration

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = {}

from transformers.utils import ModelOutput

from src.model import Model
from src import cache


class HFModel(Model):
    """
    Wrapper for a huggingface model that mostly benefits from a caching mechanism.

    NOTE: the caching mechanism here is fairly aggressive and doesn't distinguish hyperparameters from the model
    (i.e. A model at temperature 1 vs 0.5 will have the same request cached under the same key!)
    """

    model_name: str

    def __init__(
            self,
            model_name: str,
            *args,
            load_in_4bit: bool = False,
            num_samples: int = 1,
            temperature: float = 1.0,
            max_tokens: int = 512,
            config: PretrainedConfig = None,
            **kwargs,
    ):
        """
        :param model_name: Huggingface model name
        :param args: Model arguments that will be passed into AutoModelForCausalLM.from_pretrained().generate(,**args)
        :param load_in_4bit: Bits and Bytes quantization to 4bit.
        """
        if config is None:
            config = AutoConfig.from_pretrained(model_name)
            #config = AutoConfig.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
            # config = AutoConfig.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')

        super(HFModel, self).__init__()
        self.config = config

        self.model_name = model_name
        self.load_in_4bit = load_in_4bit

        self.model_args = args

        # Note initialized here so you can have instantiations of the model floating around.
        self.model = None
        self.tokenizer = None

        self.temperature = temperature
        self.num_samples = num_samples
        self.max_tokens = max_tokens

        self.total_cost = 0.0

    def load_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=self.load_in_4bit, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="mps") #, quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        device_map = 'mps'

        #self.model = T5ForConditionalGeneration.from_pretrained(
        #    self.model_name, device_map=device_map, load_in_4bit=self.load_in_4bit)  # Change to the appropriate size (e.g., t5-base, t5-large, etc.)
        #self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # super().load_model()

    @cache.cached(data_ex=timedelta(days=30), no_data_ex=timedelta(hours=1), prepended_key_attr='model_name')
    def inference(self, prompt: str, *args, only_new_tokens=True, tokenizer_args=None, model_args=None, decode_args=None, num_samples=None, **kwargs) -> Any:
        if model_args is None:
            model_args = {'max_new_tokens': self.max_tokens, 'temperature': self.temperature, 'num_return_sequences': self.num_samples or num_samples, 'do_sample': True}
        if tokenizer_args is None:
            tokenizer_args = {'max_length': 2000, 'truncation': True}
        if decode_args is None:
            decode_args = {}

        if not self.model or not self.tokenizer:
            self.load_model()

        model_inputs = self.tokenizer(prompt, return_tensors="pt", **tokenizer_args).to('cuda' if torch.cuda.is_available() else 'mps')
        output = self.model.generate(**model_inputs, **model_args)
        if only_new_tokens:
            output = self.tokenizer.batch_decode(output[:, model_inputs['input_ids'].shape[1]:], skip_special_tokens=True, **decode_args)
        else:
            output = self.tokenizer.batch_decode(output, skip_special_tokens=True, **decode_args)
        return output

if __name__ == "__main__":
    model = HFModel("facebook/opt-125m")
    print(model)

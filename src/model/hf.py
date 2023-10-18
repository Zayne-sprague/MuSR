import os
import itertools
import time
import openai
from datetime import timedelta
import random

from typing import List, Dict, Union, Any, Generator
from tqdm import tqdm
from transformers import GPT2TokenizerFast, AutoModel, AutoTokenizer, Pipeline, AutoModelForCausalLM

from src.model.model import Model
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
    ):
        """
        :param model_name: Huggingface model name
        :param args: Model arguments that will be passed into AutoModelForCausalLM.from_pretrained().generate(,**args)
        :param load_in_4bit: Bits and Bytes quantization to 4bit.
        """

        self.model_name = model_name
        self.load_in_4bit = load_in_4bit

        self.model_args = args

        # Note initialized here so you can have instantiations of the model floating around.
        self.model = None
        self.tokenize = None

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", load_in_4bit=self.load_in_4bit)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @cache.cached(data_ex=timedelta(days=30), no_data_ex=timedelta(hours=1), prepended_key_attr='model_name')
    def inference(self, prompt: str, *args, tokenizer_args=None, model_args=None, decode_args=None, **kwargs) -> Any:
        if model_args is None:
            model_args = {'max_length': 2000}
        if tokenizer_args is None:
            tokenizer_args = {}
        if decode_args is None:
            decode_args = {}

        if not self.model or not self.tokenizer:
            self.load_model()

        model_inputs = self.tokenizer(prompt, return_tensors="pt", **tokenizer_args).to('cuda')
        output = self.model.generate(**model_inputs, **model_args)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True, **decode_args)
        return output

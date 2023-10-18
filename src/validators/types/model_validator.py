import sys
from typing import List, Union, Tuple, Optional

from src.logic_tree.tree import LogicNode, LogicNodeFactType
from src.validators.validator import Validator
from src.model import Model


class ModelValidator(Validator):
    """
    Prompt a Language Model to answer some question given the deduction, then check to see the models answer.

    You can specify an early escape model if one model has a low false positive rate (we use gpt3.5 to early escape)
    """
    def __init__(
            self,
            model: Model,
            prompt: str,
            reason_why: str,
            answer_for_validity: str = 'no',
            conditional: Optional[str] = None,
            early_escape_model: Optional[Model] = None
    ):
        """
        :param model: The main model you will prompt.
        :param prompt: What to give the models.
        :param reason_why: Used in the retry prompt for explaining why we need to retry.
        :param answer_for_validity: What to check in the model response that means the deduction is valid.
        :param conditional: Conditional content/word that a parent node must have before calling this.
        :param early_escape_model: Model that is called first and if it matches answer_for_validity we escape before
            calling the main model.
        """

        self.model = model
        self.prompt = prompt
        self.reason_why = reason_why
        self.answer_for_validity = answer_for_validity
        self.condtional = conditional
        self.early_escape_model = early_escape_model

    def validate(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> bool:

        if self.condtional:
            check_validity = False

            p = template.parent
            while p is not None:
                if self.condtional.lower() in p.value.lower():
                    check_validity = True
                    break
                p = p.parent

            if not check_validity:
                return True

        if self.early_escape_model:
            early_prompt = f'{self.prompt}\n\nThe Deduction:\n{raw_output}\n\nWrite your answer in the following format:\nANSWER: (yes/no)'
            early_output = self.early_escape_model.inference(early_prompt)
            early_output = early_output.choices[0]['message']['content']
            early_answer = early_output.split('ANSWER:')[-1]

            if self.answer_for_validity.lower() in early_answer.lower():
                return True

        prompt = f'{self.prompt}\n\nThe Deduction:\n{raw_output}\n\nWrite a short description of your reasoning then answer in the following format:\nANSWER: (yes/no)'
        output = self.model.inference(prompt)
        output = output.choices[0]['message']['content']
        answer = output.split('ANSWER:')[-1]

        if self.answer_for_validity.lower() in answer.lower():
            return True
        return False

    def retry_prompt(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> str:
        return f'''
Your previous output:

{raw_output}

This is incorrect.  

{self.reason_why}
        '''

from typing import List, Tuple, Optional

from src.logic_tree.tree import LogicNode


class Validator:

    def retry_prompt(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> str:
        """
        Prompt that will be used when the validator fails.  This will be used to requery the LLM for a new deduction.

        This will be appended to the original prompt, so only explain why we need to retry (not what a deduction is etc)

        :param template: Node we were trying to fill in
        :param explicit_facts: The explicit facts that were generated in the invalid completion.
        :param commonsense_facts: The commonsense facts that were generated in the invalid completion
        :param raw_output: The raw output from the LLM (unparsed)
        """
        raise NotImplemented("Implement a retry prompt for every validator.")

    def validate(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> bool:
        """
        Function that validates the output of an LLM call for creating a deduction.

        :param template: Node we were trying to fill in
        :param explicit_facts: The explicit facts that were generated in the invalid completion.
        :param commonsense_facts: The commonsense facts that were generated in the invalid completion
        :param raw_output: The raw output from the LLM (unparsed)
        :return: True for valid or False for invalid (should retry)
        """
        raise NotImplemented("Implement a validate function for every validator")

    def __call__(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Wrapper for the validate call which just adds the retry prompt to the output when the validate call fails."""
        valid = self.validate(template, explicit_facts, commonsense_facts, raw_output, *args, **kwargs)
        if not valid:
            return valid, self.retry_prompt(template, explicit_facts, commonsense_facts, raw_output, *args, **kwargs)
        return valid, None

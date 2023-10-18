from typing import List

from src.logic_tree.tree import LogicNode, LogicNodeFactType
from src.validators.validator import Validator


class StructureValidator(Validator):
    """
    Simple validator that makes sure the number of explicit and commonsense facts in the template node match the same
    lengths of the ones generated.
    """

    def validate(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> bool:
        return \
            len(explicit_facts) == len([
                x for x in template.children if x.fact_type == LogicNodeFactType.EXPLICIT
            ]) and \
            len(commonsense_facts) == len([
                x for x in template.children if x.fact_type == LogicNodeFactType.COMMONSENSE
            ])

    def retry_prompt(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> str:
        return f'You erroneously produced this last time.\n{raw_output}\n\nThis does not match the structure or include two facts that are the same.  Please return the correct number of "Facts From Story" and "Commonsense Knowledge" this time and make sure they are unique.\n\nOutput:'

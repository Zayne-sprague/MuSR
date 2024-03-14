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
            retry_idx: int = 0,
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
            retry_idx: int = 0,
            *args,
            **kwargs
    ) -> str:
        if len(explicit_facts) < len([
                x for x in template.children if x.fact_type == LogicNodeFactType.EXPLICIT
            ]):
            needed = len([
                x for x in template.children if x.fact_type == LogicNodeFactType.EXPLICIT
            ])
            needed_cs = len([
                x for x in template.children if x.fact_type == LogicNodeFactType.COMMONSENSE
            ])
            if retry_idx == 0 or retry_idx > 4:
                return f'You erroneously produced this last time.\n{raw_output}\n\nYou need {"more" if needed > len(explicit_facts) else "fewer"} explicit facts in the story.  Try {"expanding or breaking apart" if needed > len(explicit_facts) else "reducing or combining"} the explicit facts such that there are {needed} explicit facts.'

            if retry_idx == 1 or retry_idx == 3:
                exp_facts_str = ' '.join([f'"{x}"' for x in explicit_facts])
                if needed > len(explicit_facts):
                    error_msg = f'You produced too few facts from story nodes, you need {needed - len(explicit_facts)} more.  Try expanding or breaking apart the facts you did create, \n\n{exp_facts_str}\n\n such that there are a total of {needed} facts from the story plus the {needed_cs} commonsense fact{"" if needed_cs == 1 else "s"}.'
                else:
                    error_msg = f'You produced too many facts, you need just {needed}.  Try reducing and focusing on the main facts for this deduction.'
            else:
                exp_facts_str = ' '.join([f'"{x}"' for x in explicit_facts])

                error_msg = f'You produced\n"{exp_facts_str}"\nwhich is good but you need {needed - len(explicit_facts)} more facts to go with it!  Try making {needed - len(explicit_facts)} new fact{"" if needed - len(explicit_facts) == 1 else "s"} that compliment this one next time so that you have the correct number of facts and are following the rules!'
            return error_msg
        else:
            needed_cs = len([
                x for x in template.children if x.fact_type == LogicNodeFactType.COMMONSENSE
            ])
            if len(explicit_facts) > len([
                x for x in template.children if x.fact_type == LogicNodeFactType.EXPLICIT
            ])*2:
                return f'You erroneously produced way too many story facts last time.\n\nThis does not match the structure or include two facts that are the same.  Please return the correct number of "Facts From Story" and "Commonsense Knowledge" this time and make sure they are unique.'
            if len(commonsense_facts) > needed_cs * 3:
                return f'You erroneously produced way too many commonsense facts last time.\n\nThis does not match the structure or include two facts that are the same.  Please return the correct number of "Facts From Story" and "Commonsense Knowledge" this time and make sure they are unique.'

            return f'You erroneously produced this last time.\n{raw_output}\n\nThis does not match the structure or include two facts that are the same.  Please return the correct number of "Facts From Story" and "Commonsense Knowledge" this time and make sure they are unique.'

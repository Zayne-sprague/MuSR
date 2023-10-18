from typing import List, Union, Tuple


from src.logic_tree.tree import LogicNode, LogicNodeFactType
from src.validators.validator import Validator


class ForbiddenTextValidator(Validator):
    """
    A validator that checks the text of the explicit and commonsense facts for keywords.  If they appear, the deduction
    is considered invalid.

    You can also condition the keywords based on parent node content (prune deductions mentioning "motive" in the
    "means" branch for example.)
    """

    def __init__(
            self,
            forbidden_words: List[Union[Tuple[str, str], str]],
            reason_why: str = None
    ):
        """
        :param forbidden_words: A list where strings are forbidden words anywhere, a sublist is formatted as
            [conditional word, forbidden word] where the "conditional word" must appear in a parent branch before we
            check for the "forbidden word" in the new deduction.
        :param reason_why: Used in the retry prompt to explain why we don't want a specific word.
        """

        self.forbidden_words = [x for x in forbidden_words if x != '']
        self.reason_why = reason_why

    def validate(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> bool:

        for word in self.forbidden_words:
            if isinstance(word, str):
                # If it's just a word, check the deduction as is.
                forbidden_word = word
            else:
                # If it's a list, check if the conditional word appears in any parent nodes, then check the deduction.
                conditional_text = word[0]
                forbidden_word = word[1]

                check_validity = False

                p = template
                while p is not None:
                    if conditional_text.lower() in p.value.lower():
                        check_validity = True
                        break
                    p = p.parent

                if not check_validity:
                    continue
            if any([forbidden_word.lower() in x.lower() for x in explicit_facts]) or  any([forbidden_word.lower() in x.lower() for x in commonsense_facts]):
                return False
        return True

    def retry_prompt(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> str:
        used_forbidden_words = []
        for word in self.forbidden_words:
            if isinstance(word, str):
                forbidden_word = word
            else:
                conditional_text = word[0]
                forbidden_word = word[1]

                check_validity = False

                p = template.parent
                while p is not None:
                    if conditional_text.lower() in p.value.lower():
                        check_validity = True
                        break
                    p = p.parent

                if not check_validity:
                    continue
            used_forbidden_words.append(forbidden_word)
        used_forbidden_words_str = '\n'.join([f'- {x}' for x in used_forbidden_words])
        reason_str = f'\nThe reason why we want to avoid using these is because {self.reason_why}' if self.reason_why else ''
        return f'''
        
Your old output:

{raw_output}

Some of the facts in your output contain content we don't want to include.  Regenerate this deduction and avoid using these words:

{used_forbidden_words_str}
{reason_str}
        '''.strip()

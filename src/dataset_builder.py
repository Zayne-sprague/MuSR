import sys
import time
from typing import Dict, Any, List, Callable, Union
from copy import deepcopy
import random
from tqdm import tqdm
from functools import partial

random.seed(0)

from src.madlib.madlib import Madlib
from src.logic_tree.tree import LogicNode, LogicTree, LogicNodeFactType
from src.model import Model
from src.validators import Validator, StructureValidator


# This prompt can be overwritten when needed, but this is the base prompt we use to create a deduction.
_base_completion_prompt_intro_ = '''
We are making a text adventure story guide.  To do this, we are using an entailment tree.  An entailment tree is a tree structure where intermediate nodes are entailed by their children.  They create a natural language reasoning proof for some collection of facts.

To fill out this tree we need to complete an entailment. Completing an entailment is akin to filling out one subtree of the entailment tree. To fill in this step, you must follow the structure of the step.

Facts From Story are facts that will be explicitly stated when we write the story.
Commonsense Knowledge are facts that most people would agree are true and don't need to be explicitly said.

All facts for the step must combine to entail the root parent fact.

No facts may contradict the current tree.  

Always match the exact structure of the entailment step I give you.  Give the same number of Facts From Story and Commonsense Knowledge facts.  Give them in the same order as well.
'''.strip()


def __create_completion_prompt__(
        example_trees: List[LogicTree],
        example_nodes: List[LogicNode],
        example_descriptions: List[str],
        intro=_base_completion_prompt_intro_,
        pad_char: str = '> ',
        because_clause_after: int = -1,
        because_clause: str = 'Because, ',
        use_complex_facts: bool = False,
) -> Callable[[LogicTree, LogicNode, str], str]:
    """
    Every time we prompt for a deduction we pass in the current state of the tree and current entailment we want to
    create a deduction for.  In order to keep that call simple for the recursive function, we bake a large prompt
    (usually detailing what a deduction is, what type of story we are making, etc.) into a partial that requires
    only the tree and entailment step as input and returns the final prompt.

    This function accomplishes that.

    All ICL arguments are zipped together so they are required to have matching lengths.

    Prompt format is

    "
    {intro_prompt}

    Here's an example

    Scenario:
    {ICL example desc 1}

    Current Tree:
    {ICL Tree 1}

    Entailment Step to Complete:
    {ICL Node 1}

    Output:
    {ICL Node 1 with its children shown (trick so you only have to pass in the node).}

    ... Repeat for all ICLs ...

    Your turn.

    Scenario:
    {description passed into the partial, not here}

    Current Tree:
    {current_tree, passed into the partial, not here}

    Entailment Step to Complete:
    {current_step, passed into the partial, not here}

    Output:
    "

    :param example_trees: ICL trees
    :param example_nodes: ICL nodes/entailment steps to be completed
    :param example_descriptions: ICL descriptions that are used to guide the deduction creation.
    :param intro: Introduction prompt, see _base_completion_prompt_intro_ (can be overwritten by passing into partial)
    :param pad_char: Pad char for LogicTree (see LogicTree for more details)
    :param because_clause_after: Include the because_clause string to every node after this depth.
    :param because_clause: The because clause, "node.value + 'Because, '", for example.  This string is appended to the end of the node value.  Anecdotally we found this helpful for creating deductions that flow from the parent.
    :param use_complex_facts: Say "Complex Fact" for nodes that have children.  We found this helpful anecdotally so that the LLM will create a fact complex enough that it can be broken down later.
    :return: A partial fn call that takes in some args later on.
    """

    def node_str(node, pad_char: str = '> ', completed: bool = False, because_clause_after: int = -1, because_clause: str = 'Because, ', use_complex_facts: bool = False):
        def node_line(_node, level, pad_char, tailing_clause: bool = True, because_clause_after: int = -1, because_clause: str = 'Because, '):
            return f'{pad_char*level}{_node.value}' + (('' if _node.value.lower().endswith('unless...') or level <= because_clause_after else f' {because_clause}') if tailing_clause else '')

        parents = []
        p = node.parent
        while p is not None:
            parents.append(p)
            p = p.parent
        parents = parents[::-1]

        estep = []
        for level, p in enumerate(parents):
            estep.append(node_line(p, level, pad_char, because_clause_after=because_clause_after, because_clause=because_clause))

        estep.append(node_line(node, len(parents), pad_char, because_clause_after=because_clause_after, because_clause=because_clause))
        for c in node.children:
            child_str = f'{pad_char * (len(parents) + 1) }'
            if c.fact_type == LogicNodeFactType.EXPLICIT:
                if completed:
                    child_str += node_line(c, 0, pad_char, tailing_clause=False, because_clause_after=because_clause_after, because_clause=because_clause) + ' | '
                if use_complex_facts and len(c.children) > 0:
                    child_str += 'Complex Fact'
                else:
                    child_str += 'Fact From Story'
            else:
                if completed:
                    child_str += node_line(c, 0, pad_char, tailing_clause=False, because_clause_after=because_clause_after, because_clause=because_clause) + ' | '
                child_str += 'Commonsense Knowledge'
            estep.append(child_str)
        return "\n".join(estep)

    ex_strs = []
    for (example_tree, example_node, example_description) in zip(example_trees, example_nodes, example_descriptions):
        example_tree_str = example_tree.print_for_gpt(pad_space=1, pad_char=pad_char)
        example_node_str = node_str(example_node, because_clause_after=because_clause_after, use_complex_facts=use_complex_facts, because_clause=because_clause)
        example_completion_str = node_str(example_node, completed=True, because_clause_after=because_clause_after, use_complex_facts=use_complex_facts, because_clause=because_clause)

        ex_strs.append(f'''
Scenario: 
{example_description}

Current Tree:
{example_tree_str}

Entailment Step to complete:
{example_node_str}

Output:
{example_completion_str}

            '''.strip())

    ex_str = "\nHere is another example.\n\n".join(ex_strs)

    def prompt(
        tree: LogicTree,
        node: LogicNode,
        description: str,
        ex_str: str,
        _intro: str,
        pad_char: str = '> '
    ):
        p= f'''
{_intro}

Here's an example.

{ex_str}

Your Turn.

Scenario: {description}

Current Tree:
{tree.print_for_gpt(pad_char=pad_char, pad_space=1, print_only_nodes_with_value=True)}

Entailment Step to Complete:
{node_str(node, pad_char=pad_char, because_clause_after=because_clause_after, use_complex_facts=use_complex_facts, because_clause=because_clause)}

{"Output:" if False else ""}
            '''.strip()
        # print(p + '\n'*3)
        return p

    return partial(prompt, ex_str=ex_str, _intro=intro, pad_char=pad_char)



def __create_completion_prompt_v2__(
        example_trees: List[LogicTree],
        example_nodes: List[LogicNode],
        example_descriptions: List[str],
        intro=_base_completion_prompt_intro_,
        pad_char: str = '> ',
        because_clause_after: int = -1,
        because_clause: str = 'Because, ',
        use_complex_facts: bool = False,
) -> Callable[[LogicTree, LogicNode, str], str]:
    """
    Every time we prompt for a deduction we pass in the current state of the tree and current entailment we want to
    create a deduction for.  In order to keep that call simple for the recursive function, we bake a large prompt
    (usually detailing what a deduction is, what type of story we are making, etc.) into a partial that requires
    only the tree and entailment step as input and returns the final prompt.

    This function accomplishes that.

    All ICL arguments are zipped together so they are required to have matching lengths.

    Prompt format is

    "
    {intro_prompt}

    Here's an example

    Scenario:
    {ICL example desc 1}

    Current Tree:
    {ICL Tree 1}

    Entailment Step to Complete:
    {ICL Node 1}

    Output:
    {ICL Node 1 with its children shown (trick so you only have to pass in the node).}

    ... Repeat for all ICLs ...

    Your turn.

    Scenario:
    {description passed into the partial, not here}

    Current Tree:
    {current_tree, passed into the partial, not here}

    Entailment Step to Complete:
    {current_step, passed into the partial, not here}

    Output:
    "

    :param example_trees: ICL trees
    :param example_nodes: ICL nodes/entailment steps to be completed
    :param example_descriptions: ICL descriptions that are used to guide the deduction creation.
    :param intro: Introduction prompt, see _base_completion_prompt_intro_ (can be overwritten by passing into partial)
    :param pad_char: Pad char for LogicTree (see LogicTree for more details)
    :param because_clause_after: Include the because_clause string to every node after this depth.
    :param because_clause: The because clause, "node.value + 'Because, '", for example.  This string is appended to the end of the node value.  Anecdotally we found this helpful for creating deductions that flow from the parent.
    :param use_complex_facts: Say "Complex Fact" for nodes that have children.  We found this helpful anecdotally so that the LLM will create a fact complex enough that it can be broken down later.
    :return: A partial fn call that takes in some args later on.
    """

    def node_str(node, pad_char: str = '> ', completed: bool = False, because_clause_after: int = -1, because_clause: str = 'Because, ', use_complex_facts: bool = False, print_only_nodes_with_value: bool = False):
        def node_line(_node, level, pad_char, tailing_clause: bool = True, because_clause_after: int = -1, because_clause: str = 'Because, '):
            return f'{pad_char*level}{_node.value}' + (('' if _node.value.lower().endswith('unless...') or level <= because_clause_after else f' {because_clause}') if tailing_clause else '')

        parents = []
        p = node.parent
        while p is not None:
            parents.append(p)
            p = p.parent
        parents = parents[::-1]

        estep = []
        for level, p in enumerate(parents):
            estep.append(node_line(p, level, pad_char, because_clause_after=because_clause_after, because_clause=because_clause))

        estep.append(node_line(node, len(parents), pad_char, because_clause_after=because_clause_after, because_clause=because_clause))
        for c in node.children:
            if print_only_nodes_with_value and c.value == '':
                continue
            child_str = f'{pad_char * (len(parents) + 1) }'
            if c.fact_type == LogicNodeFactType.EXPLICIT:
                if completed:
                    child_str += node_line(c, 0, pad_char, tailing_clause=False, because_clause_after=because_clause_after, because_clause=because_clause) + ' | '
                if use_complex_facts and len(c.children) > 0:
                    child_str += 'Complex Fact'
                else:
                    child_str += 'Fact From Story'
            else:
                if completed:
                    child_str += node_line(c, 0, pad_char, tailing_clause=False, because_clause_after=because_clause_after, because_clause=because_clause) + ' | '
                child_str += 'Commonsense Knowledge'
            estep.append(child_str)
        return "\n".join(estep)

    ex_strs = []
    for (example_tree, example_node, example_description) in zip(example_trees, example_nodes, example_descriptions):
        example_tree_str = example_tree.print_for_gpt(pad_space=1, pad_char=pad_char)
        example_node_str = node_str(example_node, because_clause_after=because_clause_after, use_complex_facts=use_complex_facts, because_clause=because_clause)
        example_completion_str = node_str(example_node, completed=True, because_clause_after=because_clause_after, use_complex_facts=use_complex_facts, because_clause=because_clause)

        ex_strs.append(f'''
Scenario: 
{example_description}

Current Tree:
{example_tree_str}

Entailment Step to complete:
{example_node_str}

Output:
{example_completion_str}

            '''.strip())

    ex_str = "\nHere is another example.\n\n".join(ex_strs)

    def prompt(
        tree: LogicTree,
        node: LogicNode,
        description: str,
        ex_str: str,
        _intro: str,
        pad_char: str = '> '
    ):
        p= f'''
{_intro}

Here's an example.

{ex_str}

Your Turn.

Scenario: {description}

Current Tree:
{tree.print_for_gpt(pad_char=pad_char, pad_space=1, print_only_nodes_with_value=True)}

Entailment Step to Complete:
{node_str(node, pad_char=pad_char, because_clause_after=because_clause_after, use_complex_facts=use_complex_facts, because_clause=because_clause)}

Output:
{node_str(node, pad_char, because_clause_after=because_clause_after, because_clause=because_clause, print_only_nodes_with_value=True)}
            '''.strip()
        # print(p + '\n'*3)
        return p

    return partial(prompt, ex_str=ex_str, _intro=intro, pad_char=pad_char)


class DatasetBuilder:
    """
    A class meant to help createa MuSR domain dataset.  Almost all datasets will inherit and modify the methods in this
    class (because each domain is doing something unique).  However, high level stuff like the recursive tree expansion
    method is used across them all.

    You can think of this class as more of a nice set of functions.
    """


    def build_madlib(
            self,
            model: Model,
            things_to_create_names: List[str],
            things_to_create_description: List[str],
            examples_of_things_to_create: List[List[str]],
            max_n_creations: int = 200
    ) -> Madlib:
        """
        If you don't have a collection of json files filled with lists of items you want to sample, you can call an LLM
        to get them.  This function will sample the LLM for the items specified and return them in the Madlib class.

        :param model: The LLM to sample the items from.
        :param things_to_create_names: What should we call these items in the madlib class.
        :param things_to_create_description: A description of what the items are (used in the prompt)
        :param examples_of_things_to_create: ICL examples of the items we want to sample (used in the prompt and returned in the resulting list)
        :param max_n_creations: How many items should we sample (used in the prompt, not set as a hard constraint).
        """

        assert len(things_to_create_description) == len(things_to_create_names) == len(examples_of_things_to_create), \
            'Names, descriptions, and examples of things to create must all be equal.'
        madlib_data = {}
        for name, x, examples in zip(things_to_create_names, things_to_create_description, examples_of_things_to_create):
            example_str = '\n'.join(examples)
            prompt = f'Create {max_n_creations} {x}\n\nHere are some examples.\n\nOutput:\n{example_str}\n\nYour Turn.\n\nOutput:'
            output, _ = self.inference(prompt, model)
            items = output.split('\n')
            madlib_data[name] = items
            madlib_data[name].extend(examples)

        return Madlib(madlib_data)

    def sample_madlib(
            self,
            madlib: Madlib,
            sampled_items: List[Union[str, List[str]]],
            description_string_format: str = None,
            previously_sampled: List[str] = None,
            sampled_item_names: List[str] = None,
            n_samples: int = 1
    ):
        """
        Complicated sampling method for getting items from a madlib.

        You may specify what you want to sample in the "sampled_items" list. However, you can also specify a random
        sample across multiple items in the madlib by passing in a sublist
        sampled_items=['motive', ['female_names', 'male_names']] for example will randomly choose between the
        female_name or male_name key.

        If you need to sample items based on the outcome of the random sample, you can comma separate the names instead
        of passing them in as a list.  I.E.
        sampled_items=['motive', ['female_names,female_relationships', 'male_names,male_relationships']]

        :param madlib: The Madlib to sample from
        :param sampled_items: What to sample (names, see above for details)
        :param description_string_format: For the strings returned, how do we format the sampled items, i.e. "{name}'s motive is {motive}."
        :param previously_sampled: What was previously sampled from the madlib (to ensure uniqueness)
        :param sampled_item_names: Rename from the sampled_items list i.e. when sampled_items = ["motive", "motive"] we can set sampled_item_names=["motive_1", "motive_2"] so we can distinguish in "description_string_format"
        :param n_samples: How many samples to pull.
        :return: List of strings formatted, List of dictionaries of sampled items, List of previously sampled items including the ones we sampled here.
        """
        if description_string_format is None:
            description_string_format = ''
        if previously_sampled is None:
            previously_sampled = []

        out_strings = []
        out_dicts = []
        n = 0
        while n < n_samples:
            out_string = deepcopy(description_string_format)
            sample = []
            out_dict = {}
            for idx, i in enumerate([y for x in sampled_items for y in (x.split(',') if isinstance(x, str) else random.sample(x, 1)[0].split(','))]):
                ignore_list = [x[idx] for x in previously_sampled]
                val = madlib.sample(i, [])[0]
                sample.append(val)

                if sampled_item_names is None:
                    out_string = out_string.replace('{'+i+'}', val)
                    out_dict[i] = val
                else:
                    out_string = out_string.replace('{'+sampled_item_names[idx]+'}', val)
                    out_dict[sampled_item_names[idx]] = val

            if sample in previously_sampled:
                continue
            n+=1
            previously_sampled.append(sample)
            out_strings.append(out_string)
            out_dicts.append(out_dict)
        return out_strings, out_dicts, previously_sampled

    def build_structure(
            self,
            depth: int = 4,
            bf_factor: Dict[int, float] = None,
            chance_to_prune_all: float = 0.45,
            chance_to_prune: float = 0.5,
            root_nodes: List[LogicNode] = None
    ) -> LogicTree:
        """
        Build a LogicTree with reduced parameters to make it easier.  See LogicTree for more info.
        """

        return LogicTree(
                chance_of_or=0.0,
                chance_of_cs_fact=0.0,
                depth=depth,
                chance_to_prune=chance_to_prune,
                chance_to_prune_all=chance_to_prune_all,
                bf_factor=bf_factor,
                enforce_cs_fact_per_level=True,
                deduction_type_sample_rate=None,
                root_structure=root_nodes
            )


    def create_completion_prompt(
            self,
            example_trees: List[LogicTree],
            example_node_completions: List[LogicNode],
            example_descriptions: List[str],
            intro: str = _base_completion_prompt_intro_,
            pad_char: str = '> ',
            because_clause_after: int = -1,
            because_clause: str = 'Because, ',
            use_complex_facts: bool = False
    ):
        """Wrapper for __create_completion_prompt__, see __create_completion_prompt__ for more detail."""
        # return __create_completion_prompt_v2__(example_trees, example_node_completions, example_descriptions, intro=intro, pad_char=pad_char, because_clause_after=because_clause_after, because_clause=because_clause, use_complex_facts=use_complex_facts)
        return __create_completion_prompt__(example_trees, example_node_completions, example_descriptions,
                                               intro=intro, pad_char=pad_char,
                                               because_clause_after=because_clause_after, because_clause=because_clause,
                                               use_complex_facts=use_complex_facts)

    def complete_structure(
            self,
            _tree: LogicTree,
            model: Model,
            description: str,
            completion_prompt_fn: Callable[[LogicTree, LogicNode, str], str],
            max_retries_on_error: int = 1,
            inplace: bool = False,
            retry_model: Model = None,
            progress_bar: bool = False,
            test_prompt: bool = False,
            use_iterative_complete_v2: bool = False,
            validators: List[Validator] = (StructureValidator())
    ) -> LogicTree:
        """
        This is the beginning of the Recursive Reasoning Tree Expansion algorithm.

        Here, we will be given some tree that has it's full structure (meaning all the children have been made and the
        populate/prune functions have been called).  However, not all nodes may have content.  This function will
        recurse through the tree and fill in all the levels in the tree that do not have content.

        :param _tree: The template tree we are going to fill in.
        :param model: The model used for the completion (deductions)
        :param description: Description for the deductions that we will use (story/example specific)
        :param completion_prompt_fn: A function that takes as input the current entailment we are creating (node with
            content-less children), the current state of the tree, and the description and returns a prompt for creating
            a deduction.
        :param max_retries_on_error: Number of times we will retry the deduction prompt if the validators fail.
        :param inplace: Update the tree inplace.
        :param retry_model: Model used on retries (i.e. GPT 3.5 as the first model and then 4 as the retry is sometimes
            useful, however, for the dataset we use GPT4 for both).
        :param progress_bar: Show a TQDM progress bar while we fill in the tree.
        :param test_prompt: Prints the first prompt for the first deduction then kills the entire program (used for debugging)
        :param use_iterative_complete_v2: For full use of validators beyond structural set this to True (our datasets use this)
        :param validators: List of validators to be used.
        """

        def get_num_steps(node):
            return (1 if any([x.value == '' for x in node.children]) else 0) + sum(
                [get_num_steps(x) for x in node.children])
        pbar = tqdm(total=sum([get_num_steps(x) for x in _tree.nodes]), desc='Filling in structure.', disable=not progress_bar)

        if retry_model is None:
            retry_model = model

        if not inplace:
            tree = deepcopy(_tree)
        else:
            tree = _tree

        def iteratively_complete(
                description: str,
                tree: LogicTree,
                node: LogicNode,
                model: Model,
                retry_model: Model,
                completion_prompt_fn,
                pbar,
                pad_char='> ',
                max_retries_on_error: int = 1,
                test_prompt: bool = False
        ):
            """Recursive Reasoning Tree Expansion Algorithm v1 (no validators) DEPRECATED"""
            children = node.children

            if any([x.value == '' for x in children]):

                prompt = completion_prompt_fn(tree, node, description)

                if test_prompt:
                    print(prompt)
                    sys.exit(0)

                raw = model.inference(prompt)
                output = raw.choices[0]['message']['content']

                def parse_out(output):
                    facts_from_story = []
                    cs_knowledge = []

                    for l in output.split('\n'):

                        val = '|'.join(l.replace(f'{pad_char}', '').split('|')[:-1])
                        if val == node.value:
                            continue

                        if '| Fact From Story' in l or '| Complex Fact' in l:
                            if val not in facts_from_story and val not in cs_knowledge:
                                facts_from_story.append(val)
                        elif '| Commonsense Knowledge' in l:
                            if val not in facts_from_story and val not in cs_knowledge:
                                cs_knowledge.append(val)
                    return facts_from_story, cs_knowledge

                facts_from_story, cs_knowledge = parse_out(output)

                retry_idx = 0
                while retry_idx <= max_retries_on_error and len(facts_from_story) + len(cs_knowledge) != len(node.children):
                    retry_idx += 1

                    prompt += f'You erroneously produced this last time.\n{output}\n\nThis does not match the structure or included two facts that are the same.  Please return the correct number of "Facts From Story" and "Commonsense Knowledge" this time and make sure they are unique.\n\nOutput:'

                    if retry_idx == 1:
                        raw = model.inference(prompt)
                    else:
                        raw = retry_model.inference(prompt)

                    output = raw.choices[0]['message']['content']

                    facts_from_story, cs_knowledge = parse_out(output)

                try:
                    for c in node.children:
                        if c.fact_type == LogicNodeFactType.COMMONSENSE:
                            c.value = cs_knowledge.pop()
                        elif c.fact_type == LogicNodeFactType.EXPLICIT:
                            c.value = facts_from_story.pop()

                except Exception as e:
                    print('ERROR: ' + str(e))
                    node.children = []

                pbar.update(1)
            for c in children:
                iteratively_complete(description, tree, c, model, retry_model, completion_prompt_fn, pbar, max_retries_on_error=max_retries_on_error, test_prompt=test_prompt)

        if use_iterative_complete_v2:
            [self.iteratively_complete_v2(description, tree, x, model, retry_model, completion_prompt_fn, pbar, max_retries_on_error=max_retries_on_error, test_prompt=test_prompt, validators=validators) for x in tree.nodes]
        else:
            [iteratively_complete(description, tree, x, model, retry_model, completion_prompt_fn, pbar, max_retries_on_error=max_retries_on_error, test_prompt=test_prompt) for x in tree.nodes]

        return tree

    def inference(
            self,
            prompt: str,
            model: Model,
            temperature: float = None
    ) -> [str, Any]:
        """Helper function for running inference on the models."""
        if temperature:
            raw = model.inference(prompt, temperature=temperature)
        else:
            raw = model.inference(prompt)
        # output = raw.choices[0]['message']['content']
        output = model.parse_output(raw)

        return output, raw

    def create_dataset_question_object(
            self,
            context: str,
            questions: List[str],
            answers: List[int],
            choices: List[List[str]],
            intermediate_trees: List[List[LogicTree]],
            intermediate_data: List[List[Any]] = None,
    ):
        """
        This keeps our datasets in the same format.

        :param context: The context for the question we will ask (i.e. the actual murder mystery)
        :param questions: List of questions we might want to ask over the context
        :param answers: List of answers for each question (should be the idx of the choices)
        :param choices: List of the possible answer choices for each question (list of lists)
        :param intermediate_trees: Intermediate reasoning trees (usually one per answer choice per question).
        :param intermediate_data: Any intermediate data (list for data per question).
        """

        if intermediate_data is None:
            intermediate_data = [[None] * len(intermediate_trees)]
        intermediate_trees = [[x.to_json() if isinstance(x, LogicTree) else x for x in y] for y in intermediate_trees]
        intermediate_data = [[x.to_json() if isinstance(x, LogicTree) else x for x in y] for y in intermediate_data]
        questions = [{'question': x, 'answer': y, 'choices': z, 'intermediate_trees': i, 'intermediate_data': j} for x, y, z, i, j in zip(questions, answers, choices, intermediate_trees, intermediate_data)]

        return {
            'context': context,
            'questions': questions,
        }

    def iteratively_complete_v2(
            self,
            description: str,
            tree: LogicTree,
            node: LogicNode,
            model: Model,
            retry_model: Model,
            completion_prompt_fn,
            pbar,
            pad_char='> ',
            max_retries_on_error: int = 1,
            test_prompt: bool = False,
            validators: List[Validator] = (StructureValidator()),
    ):
        """Recursive Reasoning Tree Expansion Algorithm v2"""

        children = node.children

        if any([x.value == '' for x in children]):
            # If any child has an empty value in the current node, we will prompt to create a deduction.

            def parse_out(output):
                """Parses the output of the LLM for explicit and commonsense facts."""
                facts_from_story = []
                cs_knowledge = []

                for l in output.split('\n'):

                    val = '|'.join(l.replace(f'{pad_char}', '').split('|')[:-1])
                    if val == node.value:
                        continue

                    if '| Fact From Story' in l or '| Complex Fact' in l or '| Deduced Fact' in l:
                        if val not in facts_from_story and val not in cs_knowledge:
                            facts_from_story.append(val)
                    elif '| Commonsense Knowledge' in l:
                        if val not in facts_from_story and val not in cs_knowledge:
                            cs_knowledge.append(val)
                return facts_from_story, cs_knowledge

            prompt = completion_prompt_fn(tree, node, description)
            prompt = f'<<SYS>>\nFollow the instructions\n<</SYS>>\n{prompt}'
            prompt_parts = prompt.split('Your Turn.')
            prompt = f'{prompt_parts[0].strip()}Your turn\n\n[INST]User:\n{prompt_parts[1].strip()}\n[/INST]\nAssistant:\nOutput:'

            if test_prompt:
                print(prompt)
                sys.exit(0)

            facts_from_story = []
            cs_knowledge = []

            retry_idx = 0

            # Do any of our validators fail?
            all_valid = True

            invalidated_by = []

            orig_prompt = prompt
            retry_messages = []

            while retry_idx <= max_retries_on_error:
                all_valid = True

                if retry_idx > 1:
                    temperature = 1.5
                    if retry_idx > 3:
                        temperature = 1.75
                    elif retry_idx > 4:
                        temperature = 1.9
                    elif retry_idx > 5:
                        temperature = 2.0

                    raw = model.inference(prompt, temperature=temperature)
                else:
                    raw = model.inference(prompt)

                output = model.parse_out(raw)[0] #raw.choices[0]['message']['content']

                facts_from_story, cs_knowledge = parse_out(output)

                for v in validators:
                    valid, retry_prompt = v(node, facts_from_story, cs_knowledge, output, retry_idx=retry_idx)

                    if not valid:
                        # If we fail, we will append the retry prompt from the validator to our deduction prompt before
                        # we ask for the new deduction.
                        retry_messages.append(retry_prompt)

                        if retry_idx >= 2:
                            # splitter = "Entailment Step to Complete:"
                            splitter = "Here's an example"
                            # prompt_parts = orig_prompt.split('Entailment Step to Complete:')
                            prompt_parts = orig_prompt.split(splitter)
                            example_errors = '\n\nAnother example of an error, do not do this.  Pay attention to the error message.\n\n'.join(retry_messages)

                            prompt = prompt_parts[0] + f'\n\nHere are examples of errors\n---\n{example_errors}\n\n---\nTry to avoid producing errors and follow the prompt correctly, specifically by following the template below exactly.\n\n{splitter}\n{prompt_parts[1]}'
                        else:
                            prompt_parts = prompt.split('Entailment Step to Complete:')
                            prompt = prompt_parts[0] + f'\n\n{retry_prompt}\n\nEntailment Step to Complete:\n{prompt_parts[1]}'

                        print("===  OUTPUT  ===")
                        print(output)
                        print("=== ===  === ===")

                        all_valid = False
                        invalidated_by.append(v.__class__.__name__)
                        print('Validator Failed: ' + v.__class__.__name__)
                        break
                if all_valid:
                    print("SUCCESS")
                    break

                retry_idx += 1

            if not all_valid:
                print('ERROR Validators Failed (Killing Branch)')
                print("Invalidated by")
                for x in invalidated_by:
                    print(x)
                node.children = []
            else:

                try:
                    for c in node.children:
                        if c.fact_type == LogicNodeFactType.COMMONSENSE:
                            c.value = cs_knowledge.pop()
                        elif c.fact_type == LogicNodeFactType.EXPLICIT:
                            c.value = facts_from_story.pop()
                except Exception as e:
                    print('ERROR (Killing Branch): ' + str(e))
                    node.children = []

            pbar.update(1)
        for c in children:
            self.iteratively_complete_v2(
                description,
                tree,
                c,
                model,
                retry_model,
                completion_prompt_fn,
                pbar,
                max_retries_on_error=max_retries_on_error,
                test_prompt=test_prompt,
                validators=validators
            )
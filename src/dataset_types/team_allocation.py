import sys
from typing import List, Dict, Union, Callable, Tuple
from functools import partial
import random
from copy import deepcopy

random.seed(0)

from src.madlib.madlib import Madlib
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.dataset_builder import DatasetBuilder
from src.model.openai import OpenAIModel
from src.validators import StructureValidator, ForbiddenTextValidator

# We use this to overwrite the original _base_completion_prompt_intro_ in dataset_builder.py
# Specifically, this is used when we are creating the object move trees.
__team_allocation_completion_intro__ = '''
We are creating a story about people being assigned to do certain jobs by a manager (you).  You have to look into the personal details of your worker, their past experiences, their likes and dislikes, and their social interactions to determine what they are good and bad at doing.

To make this story, we've created a tree structure that outlines the narrative.  Your job is to fill out the entailment trees that prove a set of facts on how well someone does at a particular job or how well two people work together.

For the facts involving skills and jobs, make the facts story driven focusing on past experience and personal details about the character.  For example, if the fact is that "George is bad at tennis" you might say that "George was never athletic growing up", "George actively avoids sporting events now" and "If someone isn't athletic and avoids sporting events, they probably aren't good at sports like tennis".

For the facts involving teamwork, focus on the social element and how the two individuals have interacted in the past.  For example, if the fact is that "George and Bill work well together" then you might say "George and Bill grab lunch from time to time.", "George and Bill were able to get a job done in average time working as a team last week", "If two people spend time together and are able to do adequate work together, then they probably work well together."

Most importantly, the facts should be interesting and personal; do not make them bland.

An entailment tree is a tree structure where intermediate nodes are entailed by their children.  They create a natural language reasoning proof for some collection of facts.

To fill out this tree we need to complete an entailment. Completing an entailment is akin to filling out one subtree of the entailment tree. To fill in this step, you must follow the structure of the step.

Facts From Story are facts that will be explicitly stated when we write the story.
Commonsense Knowledge are facts that most people would agree are true and don't need to be explicitly said.
Complex Facts are facts that will be entailed by simpler facts from the story (they will be filled in later through a recursive call back to you!)

All facts for the step must combine to entail the root parent fact.

No facts may contradict the current structure tree.  

Always match the exact structure of the entailment step I give you.  Give the same number of Facts From Story and Commonsense Knowledge facts.  Give them in the same order as well.

Each fact should be crucial to the deduction.  Intentionally leave out details so that the other facts can account for them.  If one fact is missing, the conclusion should not be entailed.  Try not to reuse the same facts.

Always use the persons name instead of a pronoun like "He" or "She", if you know someones name, use the name.
'''.strip()


class TeamAllocationDataset(DatasetBuilder):
    """
    Wrapper of the datasetbuilder class for specific Team Allocation functionality.

    Details specific to Team Allocation will be covered here.  Details about generic dataset building can be found in
    DatasetBuilder's file.
    """

    def build_assignment(
            self,
            people: List[str],
    ):
        """
        This function builds a matrix of sorts where we assign skill levels to each person and relatonship scores.

        The gold assignment is guaranteed to have a higher sum of assignments than any other combination.

        The assignment score is calculated as (in pseudo vars):

        score = Person_1_skill_1 + Person_2_skill_2 + Person_3_skill_2 + Relation_of_Person_2_and_3

        Note: we don't need to know the skills here yet (they're just skill 1 and 2 for now)

        :param people: List of names for people
        """

        N = 0
        GOOD = 3
        OKAY = 2
        BAD = 1

        def asgn(x=0, y=None):
            if x > 2:
                return x
            return random.sample([BAD, OKAY, GOOD][x:y], 1)[0]

        best_pair = [[people[0]], list(sorted([people[1], people[2]]))]

        paired_score = asgn()
        people_levels = {
            people[0]: {'skills': [asgn(), BAD], 'cooperation': [N, BAD, BAD]},
            people[1]: {'skills': [BAD, asgn()], 'cooperation': [BAD, N, paired_score]},
            people[2]: {'skills': [BAD, asgn()], 'cooperation': [BAD, paired_score, N]},
        }

        def score(pair):
            pair_sum = 0
            pair_sum += people_levels[pair[0][0]]['skills'][0]
            pair_sum += sum([people_levels[pair[1][0]]['skills'][1], people_levels[pair[1][1]]['skills'][1]])
            pair_sum += people_levels[pair[1][0]]['cooperation'][people.index(pair[1][1])]

            return pair_sum

        def score_pairs(pairs):
            gold_pair = None
            scored_pairs = []
            for p in pairs:
                s = score(p)
                if p == best_pair:
                    gold_pair = [s, p]
                    continue
                scored_pairs.append([s, p])


            return [gold_pair, *scored_pairs]

        def gen_pairs():
            pairs = []
            for i in people:
                for j in people:
                    for x in people:
                        if len(list(set([i, j, x]))) != 3:
                            continue
                        p = [[i], list(sorted([j, x]))]
                        if p in pairs:
                            continue
                        pairs.append(p)
            return pairs

        scored_pairs = score_pairs(gen_pairs())
        delta = scored_pairs[0][0] - max([x[0] for x in scored_pairs[1:]])

        max_updates = 10
        update_idx = 0

        # We update the matrix of skills / relationships until an assignment is fairly close to the gold (but still
        # subpar.)  We only update 1 persons skill or 1 relationship score at a time.
        while delta > 2 and update_idx < max_updates:
            update_idx += 1

            # max_allowed_assignment = min(3, delta-1)
            second_best = list(sorted(scored_pairs[1:], key=lambda x: x[0]))[0][1]

            group = 1 if random.random() > 0.75 else 0
            if group == 0:
                people_levels[second_best[0][0]]['skills'][0] = asgn(x=people_levels[second_best[0][0]]['skills'][0])  # Chance at a better skill level.
            else:
                inc = 1 if random.random() > 0.66 else 0
                if inc == 0:
                    person = random.sample(second_best[1], 1)[0]
                    people_levels[person]['skills'][1] = asgn(x=people_levels[person]['skills'][1])
                else:
                    people_levels[second_best[1][0]]['cooperation'][people.index(second_best[0][0])] = asgn(x=people_levels[second_best[1][0]]['cooperation'][people.index(second_best[0][0])])
                    people_levels[second_best[0][0]]['cooperation'][people.index(second_best[1][0])] = people_levels[second_best[0][0]]['cooperation'][people.index(second_best[1][0])]

            scored_pairs = score_pairs(gen_pairs())
            delta = scored_pairs[0][0] - max([x[0] for x in scored_pairs[1:]])

            # This should never happen (meaning that there is an assignment better than the gold).  But it's here just
            # in case.
            assert delta > 0, 'WRONG!'
        return people_levels, best_pair, [x[1] for x in scored_pairs]

    def create_facts(self, people_levels, people, skills):
        """
        Given the skill scores and relationship scores of the list of people and the skill names, let's create the
        set "F" (gold facts) that the reasoning tree will expand upon.

        :param people_levels: Main output of build_assignment
        :param people: List of people
        :param skills: List of skills (2)
        """

        prof_levels = {1: 'bad', 2: 'okay', 3: 'good'}
        coop_levels = {1: 'badly', 2: 'okay', 3: 'well'}

        facts = [
            f'{name} is {prof_levels[prof]} at {skill.lower()}.'
            for name, vals in people_levels.items()
            for skill, prof in zip(skills, vals['skills'])
        ]

        facts.append(
            f'{people[0]} and {people[1]} work {coop_levels[people_levels[people[0]]["cooperation"][1]]} together.'
        )
        facts.append(
            f'{people[0]} and {people[2]} work {coop_levels[people_levels[people[0]]["cooperation"][2]]} together.'
        )
        facts.append(
            f'{people[1]} and {people[2]} work {coop_levels[people_levels[people[1]]["cooperation"][2]]} together.'
        )
        return facts

    def create_fact_trees(
            self,
            model: OpenAIModel,
            facts: List[str],
            tasks,
            description: str,
            example_completion_trees: List[LogicTree],
            example_completion_nodes: List[LogicNode],
            example_completion_descriptions: List[str],
            depth: int = 4,
            bf_factor: Dict[int, float] = None,
            chance_to_prune_all: float = 0.45,
            chance_to_prune: float = 0.5,
            max_retries_on_error: int = 1,
            progress_bar: bool = False,
            test_complete_structure_prompt: bool = False,
            retry_model: OpenAIModel = None,
            use_complex_facts: bool = True,
            use_validators: bool = False
    ):
        """

        :param model: See datasetbuilder
        :param facts: Output from create_facts
        :param tasks: The skills/tasks
        :param description: See datasetbuilder
        :param example_completion_trees: See datasetbuilder
        :param example_completion_nodes: See datasetbuilder
        :param example_completion_descriptions: See datasetbuilder
        :param depth: See LogicTree
        :param bf_factor: See LogicTree
        :param chance_to_prune_all: See LogicTree
        :param chance_to_prune: See LogicTree
        :param max_retries_on_error: See datasetbuilder
        :param progress_bar: See datasetbuilder
        :param test_complete_structure_prompt: See datasetbuilder
        :param retry_model: See datasetbuilder
        :param use_complex_facts: See datasetbuilder
        :param use_validators: See datasetbuilder
        """

        nodes = [LogicNode(f'{x}  Because we find out in the story that, ') for x in facts]

        validators = [
            StructureValidator(),
            ForbiddenTextValidator(
                forbidden_words=[
                    *tasks
                ]
            )
        ]


        tree = self.complete_structure(
            self.build_structure(
                depth=depth,
                bf_factor=bf_factor,
                chance_to_prune_all=chance_to_prune_all,
                chance_to_prune=chance_to_prune,
                root_nodes=[LogicNode(description, nodes, frozen=True, prunable=False)]
            ),
            model,
            description=description,
            completion_prompt_fn=self.create_completion_prompt(example_completion_trees, example_completion_nodes,
                                                               example_completion_descriptions, intro=__team_allocation_completion_intro__, because_clause_after=-1, use_complex_facts=use_complex_facts),
            max_retries_on_error=max_retries_on_error,
            inplace=True,
            progress_bar=progress_bar,
            retry_model=retry_model,
            test_prompt=test_complete_structure_prompt,
            use_iterative_complete_v2=use_validators,
            validators=validators
        )

        return tree

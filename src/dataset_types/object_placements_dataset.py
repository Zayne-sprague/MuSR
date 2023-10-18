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
__object_placements_completion_intro__ = '''
We are creating a story where people are going to move a lot of items many times.  The goal of the story is to be interesting and unique but also have a clear tracking of where objects are so we can quiz readers and language models later on about the world state and about who knows what.

To make this story, we've created a tree structure that outlines the narrative and updates to the world.  Your job is to fill out the entailment trees that prove a person saw or did not see an event happen.

An entailment tree is a tree structure where intermediate nodes are entailed by their children.  They create a natural language reasoning proof for some collection of facts.


To fill out this tree we need to complete an entailment. Completing an entailment is akin to filling out one subtree of the entailment tree. To fill in this step, you must follow the structure of the step.

Facts From Story are facts that will be explicitly stated when we write the story.
Commonsense Knowledge are facts that most people would agree are true and don't need to be explicitly said.
Complex Facts are facts that will be entailed by simpler facts from the story (they will be filled in later through a recursive call back to you!)

All facts for the step must combine to entail the root parent fact.

No facts may contradict the current structure tree.  

Do not include facts about other people, focus on the facts for the person who is seeing or not seeing something move.

Always match the exact structure of the entailment step I give you.  Give the same number of Facts From Story and Commonsense Knowledge facts.  Give them in the same order as well.

Never explicitly say someone didn't see something or did see something.  Your job is to provide facts that suggest this.  For example, if May saw Greg move something, we might say "May was watching Greg while doing her chores", and that "by watching someone, you are seeing what they are doing".  Notice we describe the physics of seeing something, but we don't outright say that someone saw something.

Never mention an item being moved or reuse an item.  For example, if theres a fact like "Greg didn't see his iphone" and you are proving why "Joel didn't see the apple move", never reuse Greg's iphone in your facts.  Our program strictly controls where items are placed, we don't want you introducing item placements we haven't accounted for.

Each fact should be crucial to the deduction.  Intentionally leave out details so that the other facts can account for them.  If one fact is missing, the conclusion should not be entailed.  Try not to reuse the same facts.

Always use the persons name instead of a pronoun like "He" or "She", if you know someones name, use the name.

Only perform one deduction at a time.  Your deduction should match the "Entailment Step to Complete" template exactly so we can parse it later on.
'''.strip()


class ObjectPlacementsDataset(DatasetBuilder):
    """
    Wrapper of the datasetbuilder class for specific Object Placements functionality.

    Details specific to Object Placements will be covered here.  Details about generic dataset building can be found in
    DatasetBuilder's file.
    """

    def create_sequence_v2(
            self,
            items: List[str],
            locs: List[str],
            people: List[str],
            max_sequence_length: int = 10,
            chance_subject_sees: float = 0.25,
            max_location_use_per_item: int = 2,
            max_movers_per_event: int = 1,
            allowed_movers: List[str] = None,
            initial_starting_positions: List[Tuple[str, str]] = None
    ):
        """
        This code will create a sequence of events where someone moves an item to a new location and other people may
        or may not see that move happen.  We use this to check the validity of a GPT4 sequence.  It's not optimal, but
        it works.


        The complexity of this algo is in making sure someone moves an item only if they've seen the item move to it's
        current location (you can't move something if you don't know where that thing is). As well as keeping track of
        everyones belief states (where people think an item is)

        :param items: List of items people can move
        :param locs: List of locations someone can move items to.
        :param people: List of people who can move stuff
        :param max_sequence_length: Max # of moves
        :param chance_subject_sees: Chance for someone (who is not the mover) to see the item move.
        :param max_location_use_per_item: # of times an item can move.
        :param max_movers_per_event: # of movers per event (usually 1)
        :param allowed_movers: Not used (you can make someone be an observer only)
        :param initial_starting_positions: Where the items are currently.
        """

        def respect_article(item, people):
            if any([item.startswith(name) for name in people]):
                return item
            return f'the {item}'

        alive_locs_per_item = {item: deepcopy(locs) for item in items}

        if initial_starting_positions is not None:
            items_to_locations = {
                item: loc for (item, loc) in initial_starting_positions
            }
            for item in items:
                if item in list(items_to_locations.keys()):
                    continue
                else:
                    items_to_locations[item] = random.sample(locs, 1)[0]

            items = list(items_to_locations.keys())

        else:
            items_to_locations = {
                item: loc for item, loc in zip(items, [random.sample(locs, 1)[0] for _ in range(len(items))])
            }

        items_to_people_info = {
            item: {name: {'known_location': True} for name in people} for item in items
        }
        location_histories = {item: {loc: 0 for loc in locs} for item in items}

        events = [
            [f'{n} sees {respect_article(i, people)} at {respect_article(l, people)}.' for n in people for (i, l) in items_to_locations.items()]
        ]
        beliefs = [
            {n: {i: l for (i, l) in items_to_locations.items()} for n in people}
        ]
        actual_locs = [deepcopy(items_to_locations)]
        event_structure = [
            {
                'event': 'opening scene',
                'immutable_sequence': events[0],
                'sequence': []
            }
        ]

        for event_idx in range(max_sequence_length):

            # You can only move items if you know where they are.
            possible_items = {k: [x for x in items if items_to_people_info[x][k]['known_location']] for k in people}

            # A mover can only be a mover if they have something to move.
            possible_movers = [x for x in (people if allowed_movers is None else allowed_movers) if len(possible_items[x]) > 0]

            updates = []

            for mover_idx in range(max_movers_per_event):
                if len(possible_movers) == 0:
                    break
                structure = {
                    'event': '',
                    'immutable_sequence': [],
                    'sequence': []
                }

                mover = random.sample(possible_movers, 1)[0]
                possible_movers.remove(mover)

                item = random.sample(possible_items[mover], 1)[0]

                possible_locations = [x for x in alive_locs_per_item[item] if items_to_locations[item] != x]
                location = random.sample(possible_locations, 1)[0]

                # Mover updates
                _update = f'{mover} moves {respect_article(item, people)} to {respect_article(location, people)}.'
                updates.append(_update)
                structure['event'] = _update

                # The mover now has seen any items at the current spot.
                for i in [x for x in items if items_to_locations[x] == location and not items_to_people_info[x][mover]['known_location']]:
                    items_to_people_info[i][mover]['known_location'] = True
                    _update = f'{mover} saw {respect_article(i, people)} at {respect_article(location, people)} when moving {respect_article(item, people)}.'
                    structure['immutable_sequence'].append(_update)
                    updates.append(_update)

                items_to_locations[item] = location
                location_histories[item][location] += 1

                if location_histories[item][location] == max_location_use_per_item:
                    alive_locs_per_item[item].remove(location)

                for subject in [x for x in people if x != mover]:
                    sees_update = random.random() < chance_subject_sees

                    if sees_update:
                        items_to_people_info[item][subject]['known_location'] = True
                        _update = f'{subject} saw {respect_article(item, people)} move to {respect_article(location, people)}.'
                        updates.append(_update)
                        structure['sequence'].append(_update)
                    else:
                        items_to_people_info[item][subject]['known_location'] = False
                        _update = f'{subject} did not see {respect_article(item, people)} move to {respect_article(location, people)}.'
                        updates.append(_update)
                        structure['sequence'].append(_update)

                event_structure.append(structure)

            if len(updates) == 0:
                continue

            beliefs.append({
                name: {item: items_to_locations[item] if items_to_people_info[item][name]['known_location'] else beliefs[-1][name][item] for item in items} for name in people
            })
            actual_locs.append(deepcopy(items_to_locations))
            events.append(updates)

        return events, beliefs, actual_locs, event_structure

    def generate_end_questions(
            self,
            ending_beliefs: Dict[str, Dict[str, str]],
            people: List[str],
            items: List[str],
            locations: List[str],
            event_structure: List[Dict[str, str]],
    ):
        """
        Generate the questions and the answers to those questions.

        :param ending_beliefs: Where do people think items are.
        :param people: The people who are moving things.
        :param items: The items being moved.
        :param locations: The locations things can move to.
        :param event_structure: The main output of create_sequence_v2
        :return:
        """
        def respect_article(item, people):
            if any([item.startswith(name) for name in people]):
                return item
            return f'the {item}'

        last_moves = []

        for i in items:
            for e in reversed(event_structure):
                if i.lower() in e['event'].lower():
                    for p in people:
                        if p.lower() in e['event'].lower().split(' ')[0]:
                            last_moves.append([p, i])
                            break
                    break

        questions = []
        answers = []
        for x in people:
            for y in items:
                if [x, y] in last_moves:
                    continue

                q = f'Which location is the most likely place {x} would look to find {respect_article(y.lower(), people)} given the story?'
                questions.append(q)
                answers.append(locations.index(ending_beliefs[x][y]))
        return questions, answers

    def create_event_trees(
            self,
            model: OpenAIModel,
            event_structure: List[Dict[str, Union[str, List[str]]]],
            items: List[str],
            locations: List[str],
            completion_description: str,
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
        :param event_structure: Main output of create_sequence_v2
        :param items: items that can move
        :param locations: Locations items can move to
        :param completion_description: Description we use that is super close to the entailment step (we found this
            helpful for making the trees include more story specific details rather than bland move details.)
        :param description: General description of the story.
        :param example_completion_trees: See datasetbuilder
        :param example_completion_nodes: See datasetbuilder
        :param example_completion_descriptions:  See datasetbuilder
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

        nodes = []
        for e in event_structure:

            # usually the beginning of the story (where everything initially is, is immutable and we don't want to add deductions to it)
            children = [
                LogicNode(x, prunable=False, can_be_leaf=True, frozen=True) for x in e['immutable_sequence']
            ]
            # For each move, however, we want to add deductions to why or why not people saw the move.
            children = [*children, *[LogicNode(x) for x in e['sequence']]]
            nodes.append(LogicNode(e['event'], children, frozen=True, prunable=False))

        validators = [StructureValidator(),]
        if use_validators:
            validators.append(ForbiddenTextValidator(
                forbidden_words=[
                    *items, *locations,
                ],
                reason_why="We are building a story where the reader must track where objects are and if other people saw an object move or not.  Because of this, when we create facts about someones observations, we do not want to infer about what other people are doing, where other items are, or about other locations.  All of these things are controlled for explicitly to create a setting where there is one correct answer without ambiguity."#  We also want to avoid using pronouns because it can make it confusing who we are talking about in the story when we use the facts to make the story.  Avoid using pronouns and use real names at all times even if the sentences are harder to read."
            ))

        tree = self.complete_structure(
            self.build_structure(
                depth=depth,
                bf_factor=bf_factor,
                chance_to_prune_all=chance_to_prune_all,
                chance_to_prune=chance_to_prune,
                root_nodes=[LogicNode(description, nodes)]
            ),
            model,
            description=completion_description,
            completion_prompt_fn=self.create_completion_prompt(example_completion_trees, example_completion_nodes,
                                                               example_completion_descriptions,
                                                               intro=__object_placements_completion_intro__, because_clause_after=0,
                                                               because_clause='Because in the story we find out,',
                                                               use_complex_facts=use_complex_facts),
            max_retries_on_error=max_retries_on_error,
            inplace=True,
            progress_bar=progress_bar,
            retry_model=retry_model,
            test_prompt=test_complete_structure_prompt,
            validators=validators,
            use_iterative_complete_v2=use_validators
        )

        return tree

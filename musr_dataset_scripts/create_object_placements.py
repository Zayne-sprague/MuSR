"""
RUN THIS FILE TO CREATE AWESOME STORIES USING AN LLM :)

Go to the main() function for arguments/control over the dataset creation.

NOTE: Expects your openai api key to be in the environment.  "OPENAI_API_KEY=api_key python script.py" (if you are using
openai LLMs)

NOTE: By default, datasets go into "{ROOT_FOLDER}/datasets/{dataset_name}.json"
"""

import json
import copy
import sys
import time
from copy import deepcopy
from pathlib import Path
import random
import pprint
import re
import traceback

random.seed(0)

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.utils.paths import OUTPUT_FOLDER

from src.dataset_types.object_placements_dataset import ObjectPlacementsDataset
from functools import partial


def respect_article(item, people):
    if any([item.startswith(name) for name in people]):
        return item
    return f'the {item}'

def respect_plural(item):
    if item.endswith('s'):
        return f'{item} are'
    return f'{item} is'

cLogicNode = partial(LogicNode, fact_type=LogicNodeFactType.COMMONSENSE)

"""ICL EXAMPLES for creating deductions. See datasetbuilder for more info."""

example1_description = 'Paul and Alice are at a karaoke bar.'
example1_tree = LogicTree(
    nodes=[
        LogicNode('Paul and Alice are at a karaoke bar.', [
            LogicNode('Opening Scene', [
                LogicNode('Paul sees the microphone at the stage.'),
                LogicNode('Alice sees the microphone at the stage.'),
                LogicNode('Paul sees the beer at the bar.'),
                LogicNode('Alice sees the beer at the bar.')
            ]),
            LogicNode('Paul moves the beer to the table.', [
                LogicNode('Alice did not see the beer move to the table.', [
                    LogicNode('Alice was facing away from the table.', [
                        LogicNode('Alice was talking to another patron.'),
                        LogicNode('The other patron was facing the table.'),
                        cLogicNode('Usually people talk to each other while facing each other, so if one person is looking in one direction the other person is looking in the opposite direction.')
                    ]),
                    cLogicNode('If someone is facing away from something else, they cannot see things transpire near that something else.')
                ])
            ]),
            LogicNode('Alice moves the microphone to the table', [
                LogicNode('Alice saw the beer at the table when moving the microphone.'),
                LogicNode('Paul saw the microphone move to the table.', [
                    cLogicNode("Paul was drinking the beer at the table."),
                    LogicNode("When something happens where a person is at, they usually see things that are happening there.")
                ])
            ]),
            LogicNode('Alice moves the beer to the trash can.', [
                LogicNode('Paul did not see the beer move to the trash can.')
            ])
        ])

    ], prune=False, populate=False
)
example1_node_completion_tree =  LogicNode('Paul and Alice are at a karaoke bar.', [LogicNode('Alice moves the beer to the trash can.', [
            LogicNode('Paul did not see the beer move to the trash can.', [
                LogicNode("Alice tricked Paul into looking \"over there\"."),
                LogicNode("Alice pointed in the opposite direction of the trash can to Paul."),
                cLogicNode("If you trick someone into looking elsewhere, they cannot see what happens in the other direction.")
            ])
        ])])


example_descriptions = [example1_description]
example_trees = [example1_tree]
example_node_completions = [example1_node_completion_tree.children[0].children[0]]

def remove_prepended_numbers(strings):
    return [re.sub(r'^\d+\.\s*', '', s) for s in strings]

def main():
    # CACHE
    cache.enable()

    # PARAMS (if not with a comment, look at the Object Placements dataset class for more info.)

    out_file = OUTPUT_FOLDER / 'custom_object_placements.json'
    if out_file:
        out_file.parent.mkdir(exist_ok=True, parents=True)

    max_sequence_len = 3
    chance_to_see = 0.33

    max_examples = 1

    tree_depth = 3
    max_structure_completion_retries = 3

    verbose = True
    use_validators = True

    # CREATION LOGIC
    total_cost = 0

    gpt35 = OpenAIModel(engine='gpt-3.5-turbo', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=1500, num_samples=1, prompt_cost=0.0015/1000, completion_cost=0.002/1000)
    gpt16k35 = OpenAIModel(engine='gpt-3.5-turbo-16k', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=1500, num_samples=1, prompt_cost=0.003/1000, completion_cost=0.004/1000)
    gpt4 = OpenAIModel(engine='gpt-4', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.03/1000, completion_cost=0.06/1000)

    model_to_use = gpt4

    dataset = []

    previous_samples = []

    __idx = 0
    raw_idx = 0
    max_idx = int(max_examples * 10)

    while __idx < max_examples and raw_idx < max_idx:
        print(f"STORY: {__idx+1}")

        raw_idx += 1
        creator = ObjectPlacementsDataset()

        # Sample a scenario to build the story around.
        madlib = creator.build_madlib(
            model_to_use,
            things_to_create_names=['scenario_descriptions'],
            things_to_create_description=["Create a scenario where a group of people are together and there is an item of great importance to at least one of those people. If this item is moved, and they don't know it has been moved, they could be negatively impacted. Do not number them, give the scenario separated by newlines only."],
            examples_of_things_to_create=[["Sarah is making coffee at her work, she wants to use almond milk for her customer.", "Aunt Mays medicine is always behind her mirror, she needs to take some of her pills before the day starts.", "Evidence is imperative for a detective, Winston needs his key to get into the evidence locker."]]
        )

        # This scenario acts like a high level description of the story.
        descriptions, _, previous_samples = creator.sample_madlib(madlib, ['scenario_descriptions'], '{scenario_descriptions}', previously_sampled=previous_samples)
        description = descriptions[0]

        # Here we prompt a model to give us details about the scenario. For example, the items that will be moved, the
        # people who will move them, why they are moving things around etc.  We found this extremely helpful in creating
        # cohesive stories about people moving things that don't sound too "dream-like".
        # We also ask the model to produce the "moves" (who moved what to where) in one go.
        # This forces the model to create a "narrative" about why things are happening before we make the actual story.
        prompt = f'''
You will build out an outline of a dramatic story given a short description of the scenario.  To do this, we are going to create three characters each with their own roles in the story and motivations.  

We will then set the scene.  We will determine what the three characters doing together (the goal of the story).

Then we will create a list of three "moves". A "move" involves one character moving an item from one location to a new location.  The items that are being moved should be smaller and tangible where as the locations should be places all the items could fit into.   For example, a shelf may be a location and an item may be a bag of coffee, this works out because a shelf can reasonably hold a bag of coffee.

For each of the three moves, we will say why someone is doing this and how it relates to the story.  Most importantly, the justification for doing something should not depend on other people in the story -- we want people to be doing their own activities and moving things around so we can later ask questions about the story and the characters observations in it for the reader.

Rules:
1) When describing characters make sure you are using real names and their roles fit given the description.
2) The motivation and location of the story should make sense given the characters and their roles.
3) The story outline should involve all three people working on something similar with one major plot point.  For example, making coffee for a customer.
4) The moves should make sense given the story thus far, the characters and their roles, and the location.  For example, a customer should not be moving milks around in a cafe because they don't work there.
5) For the moves, you must use tangible small items, do not use ideas "a performance" for example, do not use large items like "a tv".  Instead use small, easily moved items like "Iphone", "Notebook", "Laptop" etc.
6) The locations you pick must be able to house the items you made.  So if you said "Golf club" was an item, all locations must be able to fit a golf club in them, you would not say "coat pocket" for example.
7) Your justifications for why someone moved an item should involve the character moving the object only! You can include details about the story and location, but it should never involve another person (as this defeats our Question and Answer test for the reader later on).
8) Only two items may be introduced into the story, three people may be introduced, and four locations.  One person may move an item twice, but no more.
9) Locations should be general, for example say "from his desk to her shelf" do not say things like "from Carl's desk to Sarah's shelf" as this makes it difficult for us to parse out the locations from your output.
10) Follow the moves template exactly so our python program can parse it. The format is: "[persons name] moves the [item name] from the [from_location] to the [destination_location]."  Do not diverge from this format.

You cannot use any other items name in the justification for the move except for the item moving.  For example, if the items are "cards and apple", when justifying the move for the "cards" you cannot say "apple" in the justification.
There must always be two unique items.

Here's an example

Description: Sarah is making coffee at her work, she wants to use almond milk for her customer.

Output:

Character 1
Name: Sarah
Role: The Barista
Motivation: Sarah wants to make Luis an almond milk coffee.

Character 2
Name: Luis
Role: A customer
Motivation: Luis is having a rough week with deadlines looming over him, so he wanted his favorite coffee with almond milk.

Character 3
Name: John
Role: A cafe worker
Motivation: John is having his first day working at the coffee cafe, he is working hard to make sure all the tables are clean for the customers.

Story outline:
Luis was having a super hard week with his paper deadline approaching and with all his experiments now failing he was in desperate need of his favorite cup of jo, with a dash of almond milk.  He ordered it from Sarah who is about to start making it.  Sarah is a skilled barista and has worked there for awhile, she loves making her customers feel welcomed.  However, John is new and constantly fumbling around with things (which is expected since he's new).  To keep him busy, Sarah put him on cleaning duty.

Moves:

Move 1 - Luis moves the almond milk from the fridge to the back shelves.
Mover: Luis
Item: almond milk
From: fridge
To: back shelves
Reason - Luis is cleaning the fridge so everyone can work more efficiently.

Move 2 - Sarah moves the coffee bag from the back shelves to the front counter.
Mover: Sarah
Item: coffee bag
From: back shelves
To: front counter
Reason - Sarah ran out of beans for making coffee and had to go back to get the spare.

Move 3 - Sarah moves the almond milk from the back shelves to the fridge.
Mover: Sarah
Item: almond milk
From: bach shelves
To: fridge
Reason - She noticed the milk was left out for too long and put it back before it spoiled.

Your turn!

Description: {description}

Output:

'''.strip()
        if verbose:
            print('--- MADLIB PROMPT ---')
            print(prompt)

        output, _ = creator.inference(prompt, model_to_use)

        if verbose:
            print("=== MADLIB OUTPUT ===")
            print(output)

        # Here we will parse out all the info we want from the generation
        items = []
        people = []
        people_data = []
        moves = []
        move_strs = []
        locations = []
        world_state = []

        try:
            lines = output.split('\n\n')

            for c in lines[0:3]:
                info = c.split('\n')
                people_data.append({
                    'name': info[1].replace('Name: ', ''),
                    'role': info[2].replace('Role: ', ''),
                    'motivation': info[3].replace('Motivation: ', '')
                })
                people.append(info[1].replace('Name: ', ''),)

            story_desc = lines[3].replace('Story outline:\n','').replace('Story Outline:\n','')

            for move_info in lines[5:]:
                m = move_info.split('\n')
                """Move 3 - Sarah moves the almond milk from the back shelves to the fridge.
Mover: Sarah
Item: almond milk
From: bach shelves
To: fridge
Reason - She noticed the milk was left out for too long and put it back before it spoiled.
"""
                move_data = {
                    'mover': m[1].replace('Mover: ', ''),
                    'item': m[2].replace('Item: ', ''),
                    'from': m[3].replace('From: ', '').replace('his', f'{m[1].replace("Mover: ", "")}\'s').replace('her', f'{m[1].replace("Mover: ", "")}\'s'),
                    'to': m[4].replace('To: ', '').replace('his', f'{m[1].replace("Mover: ", "")}\'s').replace('her', f'{m[1].replace("Mover: ", "")}\'s'),
                    'justification': m[5].replace('Reason - ', '')
                }

                moves.append(move_data)

                locations.extend([move_data['from'], move_data['to']])
                items.append(move_data['item'])

                if move_data['item'] not in [x[0] for x in world_state]:
                    world_state.append([move_data['item'], move_data['from']])

                move_str = f'{move_data["mover"]} moves {respect_article(move_data["item"], people)} to {respect_article(move_data["to"], people)}.'
                move_strs.append(move_str)

            people = list(sorted(set(people)))
            items = list(sorted(set(items)))
            locations = list(sorted(set(locations)))

            items_str = '\n'.join([f'- {x}' for x in items])

            # Gotta have 2 items always.
            if len(items) != 2:
                print("ERROR: WRONG ITEM COUNT")
                continue

            # You cannot move an item because of another item (it makes our stuff downstream harder).
            if any([z.lower() in y['justification'].lower() and x.lower() != y['item'].lower() for x in items for y in moves for z in x.split(' ')]):
                print("ERROR: YOU CAN'T MENTION AN ITEM IN THE JUSTIFICATION")
                continue

            # To check if the moves produced by the model are valid, we recreate them using our code.  This also gives
            # us the belief states and observations of other people in the story.  This could be cleaner, but deadlines.
            max_retries = 100_000
            mri = 0
            sampled = False
            while mri < max_retries:
                mri += 1
                events, beliefs, actual_locs, event_structure = creator.create_sequence_v2(
                    items, deepcopy(locations), people, max_sequence_length=max_sequence_len, chance_subject_sees=chance_to_see, initial_starting_positions=world_state
                )

                retry = False
                for gidx, (gm, e) in enumerate(zip(move_strs, events[1:])):
                    move = e[0]
                    if move.lower() != gm.lower():
                        retry = True
                        break
                if retry:
                    continue
                sampled = True
                break
            if not sampled:
                raise Exception("Couldn't get a sample from the code that matched the gold move from GPT.")

            question, answers = creator.generate_end_questions(
                ending_beliefs=beliefs[-1], people=people, items=items, locations=locations, event_structure=deepcopy(event_structure)
            )
        except Exception as e:
            # If for any reason something fails, we will just retry the whole loop.
            print("ERROR")
            print(e)
            traceback.print_exc()
            print(flush=True)
            time.sleep(1)
            continue

        print(f"STORY: {__idx}")

        completion_description = f'''
{story_desc}

Character 1:
Name: {people_data[0]['name']}
Role in story: {people_data[0]["role"]}
Motivation in story: {people_data[0]["motivation"]}

Character 2:
Name: {people_data[1]['name']}
Role in story: {people_data[1]["role"]}
Motivation in story: {people_data[1]["motivation"]}


Character 3:
Name: {people_data[2]['name']}
Role in story: {people_data[2]["role"]}
Motivation in story: {people_data[2]["motivation"]}

Respect the peoples roles when creating deductions and try to create motivations for them in the story that would influence if they see or do not see an event based on the story.  Try to keep these motivations or stories continuing across "moves".
        '''

        for eidx, e in enumerate(event_structure[1:]):
            event_structure[eidx+1]['event'] += f' Because, {moves[eidx]["justification"]}'

        tree = creator.create_event_trees(
            model_to_use,
            event_structure,
            items=items,
            locations=locations,
            completion_description=completion_description,
            description=description,
            example_completion_trees=example_trees,
            example_completion_nodes=example_node_completions,
            example_completion_descriptions=example_descriptions,
            depth=tree_depth,
            bf_factor={2: 1.0},
            chance_to_prune=0.0,
            chance_to_prune_all=0.0,
            max_retries_on_error=max_structure_completion_retries,
            progress_bar=True,
            test_complete_structure_prompt=False,
            retry_model=model_to_use,
            use_validators=use_validators
        )

        if verbose:
            print('=== TREE OUTPUT ===')
            print(tree.print_for_gpt(pad_char='> ', pad_space=1))


        # We will now create the story in many chapters.
        # Although these are created in chapters, we take advantage of the autoregressive nature of LLMs.  Specifically,
        # every chapter should start with the story so far, and then ask the model to continue it.
        failed_parse = False
        chapters = []
        story_so_far = ''
        for loop_idx, __n in enumerate(tree.nodes[0].children):
            n = deepcopy(__n)
            if n.value == 'opening scene':
                # The opening scene "chapter" is meant to introduce the setting but also say that everyone knows where
                # everything is initially (a starting point.)
                facts_str = '\n'.join([f'- {respect_article(respect_plural(x), people)} at {respect_article(y, people)}' for x, y in actual_locs[0].items()])

                opening_prompt = f"""
Create an opening scene description for a story.  It will be short.  Only write about the objects we list out and their location.  Your story MUST include each item and their location from the list.  Your story also MUST indicate that all the people we give you saw the location of all these items!

You may use the description to infer the correct scenery to describe, but are only allowed to talk about the facts presented in the list.

You must state that everyone knows where everything is, "They were all aware of each items location" or something like that is a safe way to ensure this condition is met.  Try to make this coherent with the story though.  For example, if someone doesn't know the exact location you could say "Everyone was aware that the item was somewhere in the location, and they definitely all knew that the other item was in the other location", or something like this.

Here is an example.

Description: Alex is making Ramen and needs the noodles to cook.

Items and Locations:
- The pans are at the stove.
- The noodles are at the fridge.
- The kitchen knife is at the table.

Character 1:
Name: Alex
Role in story: A want to be chef
Motivation in story: To make a bowl of ramen.

Character 2:
Name: Carol
Role in story: the roommate
Motivation in story: Hanging out with Alex, she is also hungry.

Character 3:
Name: Joshua
Role in story: A random visitor
Motivation in story: Joshua was a friend of Alex but showed up unannounced and hungry.

Output: Alex and Carol were having a peaceful evening hanging out.  Both getting a bit peckish, they decided to eat some ramen, which Alex had been practicing making for awhile now. Everyone knew Alex was the "Chef Friend", meaning that he was always cooking something delicious up. In fact, that's why a hungry friend, who showed up unannounced, Joshua decided to show up.  All three of them noticed that the pans were already on the stove, perfect for ramen making! The kitchen knife was on the table, and the noodles were in the fridge.

Your turn.

Description: {story_desc}

Items and Locations:
{facts_str}

Character 1:
Name: {people_data[0]['name']}
Role in story: {people_data[0]["role"]}
Motivation in story: {people_data[0]["motivation"]}

Character 2:
Name: {people_data[1]['name']}
Role in story: {people_data[1]["role"]}
Motivation in story: {people_data[1]["motivation"]}


Character 3:
Name: {people_data[2]['name']}
Role in story: {people_data[2]["role"]}
Motivation in story: {people_data[2]["motivation"]}

Output:
                                """.strip()
                if verbose:
                    print('--- OPENING PROMPT ---')
                    print(opening_prompt)

                opening_output, _ = creator.inference(opening_prompt, model_to_use)

                if verbose:
                    print('=== OPENING OUTPUT ===')
                    print(opening_output)

                chapters.append(opening_output)
                story_so_far += opening_output
            else:
                # For every move, we will generate two "chapters".  A chapter for the actual move happening and a
                # chapter for the observations of everyone else.  We keep these separate to prevent the LLM from saying
                # things like "Claire wasn't able to see X because she was cooking" or things like that.  We would
                # rather the LLM just say "Claire was cooking".

                paragraphs = []

                children = [x for x in n.children if 'when moving' in x.value]
                paras = [x for x in n.children if 'when moving' not in x.value]
                _n = copy.deepcopy(n)
                _n.children = children
                mtree = copy.deepcopy(tree)
                mtree.nodes[0].children = [_n]

                facts = [n.value, *children]  # mtree.get_facts()
                facts_str = "\n".join([f'- {x}' for x in facts])

                try:
                    moving_character = [x for x in people_data if x['name'] == facts[0].split(' ')[0]][0]
                except Exception:
                    failed_parse = True
                    break

                move_prompt = f"""
You are going to continue our story that we have written by writing a short description of this event that will happen next.  Only write about the move, do not add any additional information.

Never say "someone didn't see something" or infer someones ability to infer where something is.  Never say "Unbeknownst" or anything like this!
Here is an example.

Only write one or two sentences.  It should be a very short continuation.

Description: Timmy was angry at Bob for cheating his way into the job Timmy deserved! So he started throwing away Bobs possessions.

Character:
Name: Timmy
Role in story: A recent graduate who is sharing an apartment.
Motivation in story: Timmy is angry because he interviewed for a job that his roommate got, but only because he cheated.

Event:
- Timmy moves the car keys to the trash bin. Because, Timmy was angry with Bob and wanted to throw away his keys.
- Timmy saw the iphone at the trash bin when moving the car keys.

Output: With an angry thrust, the keys clanked against the tin trash bin.  An unexpected *smack* followed though... curiosity overtaking his anger, Timmy looked in the trash and saw the iphone in there as well.

Here is another example.

Description: Carol had just moved into her new apartment, but, the previous tenant made a huge mess! The landlord wouldn't do anything, so it looks like she has to clean it all up herself.

Character:
Name: Carol
Role in story: Just moved into a new messy apartment.
Motivation in story: Carol wants to clean her new apartment that was left a mess by the previous tenant and has exactly no help from management.

Event:
- Carol moves the noodles to the pantry. Because, Carol was excited to have a clean apartment finally, and the noodles were the last step!

Output: Carol excitingly places the noodles back into the pantry.  What was once thought of as a never ending onslaught of trash and random items finally concluded and the apartment was finally clean again!

Your turn.

Description: {story_desc}

Character:
Name: {moving_character['name']}
Role in story: {moving_character["role"]}
Motivation in story: {moving_character["motivation"]}

Event:
{facts_str}

Output:
{story_so_far}
                """.strip()

                if verbose:
                    print('--- MOVE PROMPT ---')
                    print(move_prompt)

                output, _ = creator.inference(move_prompt, model_to_use)

                if verbose:
                    print('=== MOVE OUTPUT ===')
                    print(output)

                paragraphs.append(output)
                story_so_far += f'\n\n{output}'

                stree = copy.deepcopy(tree)
                stree.nodes = [n]
                stree.nodes[0].children = []

                for p in paras:
                    if len(p.children) > 0:
                        stree.nodes[0].children.extend(p.children)
                    else:
                        stree.nodes[0].children.append(p)

                obs_facts = stree.get_facts()
                obs_facts_str = "\n".join(sorted([f'- {x.value}' for x in obs_facts]))

                obs_retry = 0
                obs_prompt_beginning = f"""
Continue the story we have written so far by writing about the observational facts below. Only write about the facts and do not add new information.  Never say "Someone saw" or "Did not notice" and never indicate if someone sees something, unless the only fact you have is "someone sees X".

Stick to the facts, there will be more information about the story that you can use to set the tone, but you should always use the facts as the main guide for the story.

Never mention the key items in the story:
{items_str}

{"There will be another event after this paragraph, so end this paragraph abruptly sticking only with the facts.  Make no general statements. The last sentence should be something about the facts we listed out only. It should be a complete sentence."
if loop_idx < len(tree.nodes[0].children) - 1 else
"This is the end of the story, write a concluding sentence after your paragraph."}

Your story should take place during the most recent move.  So while this is happening:

"{output}"

the facts you will be writing about are happening at the same time.

Here is an example.

Description: Jerry, Marry and Timmy are getting ready for the day.  Jerry has a huge meeting that he needs to prep for.  Marry is excited to help Jerry for his meeting and to hear about it later that day.  Timmy was getting ready for his test, but is being a bit inconsiderate to his dad, Jerry, with respect to his big meeting.

Character 1:
Name: Jerry
Role in story: the husband
Motivation in story: Jerry had a huge meeting coming up, one that could decide the fate of his career.

Character 2:
Name: Marry
Role in story: the wife
Motivation in story: Marry is always super supportive and wants the best for her family.

Character 3:
Name: Timmy
Role in story: the son
Motivation in story: Timmy has a huge test coming up in his school which he is nervous for and accidentally makes him a bit inconsiderate to everyone else.

Observational Facts:
- Jerry is cooking breakfast
- The trash bin is not in the kitchen.
- Marry is outside watering her garden.
- Marry has a window into the room with the trash bin.

Output: Jerry was hungry before he starts his day, so he was cooking his breakfast.  The kitchen turned out to not have the trash bin though.  Marry, always with her green thumb, was watering her garden and could see the trash bin through a nearby window.  

Your turn.

Description: {story_desc}

Character 1:
Name: {people_data[0]['name']}
Role in story: {people_data[0]["role"]}
Motivation in story: {people_data[0]["motivation"]}

Character 2:
Name: {people_data[1]['name']}
Role in story: {people_data[1]["role"]}
Motivation in story: {people_data[1]["motivation"]}

Character 3:
Name: {people_data[2]['name']}
Role in story: {people_data[2]["role"]}
Motivation in story: {people_data[2]["motivation"]}

Observational Facts:
{obs_facts_str}
                                """.strip()

                output_obs_prompt = f'''
Output:
{story_so_far}
'''
                good_parse = False
                while obs_retry < 3:
                    obs_retry += 1
                    if verbose:
                        print('--- OBS STORY PROMPT ---')
                        print(f'{obs_prompt_beginning}\n\n{output_obs_prompt}')

                    obs_output, _ = creator.inference(f'{obs_prompt_beginning}\n\n{output_obs_prompt}', model_to_use)

                    if verbose:
                        print('=== OBS STORY OUTPUT ===')
                        print(obs_output)

                    if any([x.lower() in obs_output.lower() for x in items]):
                        obs_prompt_beginning += f"\n\nOne of your last outputs was this: \n\n{obs_output}\n\nThis is incorrect because it mentions one of our key items: \n{items_str}\n\nMake sure your next generation does not include mentioning our key items as that can confuse the reader."
                        print("FAILED OBS PROMPT")
                    else:
                        good_parse = True
                        break

                if not good_parse:
                    failed_parse = True
                    break

                paragraphs.append(obs_output)
                story_so_far += f' {obs_output}'

                sub_paras = "\n\t".join(paragraphs[1:])
                chapter = f'{paragraphs[0]}\n\tMeanwhile, {sub_paras}'
                chapters.append(chapter)
        if failed_parse:
            print("ERROR: FAILED TO PARSE OBS PARAGRAPH.")
            continue

        story = story_so_far

        cost = gpt35.total_cost + gpt16k35.total_cost + gpt4.total_cost
        total_cost += cost
        print(
            f"EXAMPLE COST: {cost:.2f} | TOTAL COST: {total_cost:.2f}")
        gpt35.total_cost = 0.0
        gpt16k35.total_cost = 0.0
        gpt4.total_cost = 0.0

        if verbose:
            print("FINISHED")
            print(tree.print_for_gpt())
            print(story)

            print('\n' * 2)

            for idx, q in enumerate(question):
                print(f'{idx + 1}: {q}')
                for l in locations:
                    print(l)


        dataset.append(
            creator.create_dataset_question_object(
                context=story, questions=question, answers=answers,
                intermediate_trees=[[tree]] * len(question), choices=[locations] * len(question),
                intermediate_data=[[{'events': events, 'beliefs': beliefs, 'actual_locs': actual_locs}]] * len(question)
            )
        )

        __idx += 1


    json.dump(dataset, out_file.open('w'))

    print(f"TOTAL COST: {total_cost} | {total_cost / max_examples} per example.")



if __name__ == "__main__":
    main()
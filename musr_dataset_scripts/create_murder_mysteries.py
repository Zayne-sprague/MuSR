"""
RUN THIS FILE TO CREATE AWESOME MURDER MYSTERIES USING AN LLM :)

Go to the main() function for arguments/control over the dataset creation.

NOTE: Expects your openai api key to be in the environment.  "OPENAI_API_KEY=api_key python script.py" (if you are using
openai LLMs)

NOTE: By default, datasets go into "{ROOT_FOLDER}/datasets/{dataset_name}.json"
"""

import copy
import json
import sys
from pathlib import Path
import random

random.seed(0)

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.madlib.madlib import Madlib
from src.utils.paths import OUTPUT_FOLDER, ROOT_FOLDER

from src.dataset_types.murder_mystery_dataset import MurderMysteryDataset

"""ICL EXAMPLES for creating deductions. See datasetbuilder for more info."""
example1_description = """

Victim: Victoria
Crime scene: Home
Murder weapon: Gun
Suspect: James
Suspect's role in story: Brother
Suspect's motive: Financial gain
"""

example1_tree = LogicTree(
nodes=[
    LogicNode('James is a murderer.', [
        LogicNode('James has a means', [
            LogicNode('James has practiced shooting guns.'),
            LogicNode('James owns guns'),
            LogicNode('If you both own and practice using guns then you have the ability to murder someone.', fact_type=LogicNodeFactType.COMMONSENSE)
        ]),
        LogicNode('James has a motive.', [
            LogicNode('James was violently desperate for cash.'),
            LogicNode('James was violently desperate for Victoria\'s cash'),
            LogicNode(
                'When someone is violently desperate they may go to extreme measures to accomplish a task, including murderer.',
                fact_type=LogicNodeFactType.COMMONSENSE)
        ]),
        LogicNode('James has a opportunity.')
    ])
], prune=False, populate=False
)

example1_node_completion = LogicNode('James has a opportunity.', [
    LogicNode('James has access to Victoria\'s house.'),
    LogicNode('Having access to someones house gives you the opportunity to murder them.', fact_type=LogicNodeFactType.COMMONSENSE)
])

example2_description = """
Story Information:
Victim: Harry
Crime scene: Racetrack
Murder weapon: Shovel
Suspect: Claire
Suspect's role in story: Running buddy
Suspects motive: To prevent someone else harm
"""
example2_tree = LogicTree(
nodes=[
    LogicNode('Claire is a murderer.', [
        LogicNode('Claire has a means.', [
            LogicNode('Claire is a farmer'),
            LogicNode(
                'Farmers typically use gardening tools like shovels in their work.',
                fact_type=LogicNodeFactType.COMMONSENSE)
        ]),
        LogicNode('Claire has a motive.'),
        LogicNode('Claire has an opportunity')
    ])
], prune=False, populate=False
)

example2_node_completion = LogicNode('Claire has a motive.', [
    LogicNode('Claire loves Brian deeply.'),
    LogicNode('Harry threatened Brian.'),
    LogicNode('Deep and passionate love can push people to do extreme things like murder when that loved one is threatened.', fact_type=LogicNodeFactType.COMMONSENSE)
])

example3_description = """
Victim: Jared
Crime scene: Public park bench
Murder weapon: Heroin overdose
Suspect: Jose
Suspect's role in story: Drug user
Suspects motive: Public humiliation
"""
example3_tree = LogicTree(
nodes=[
    LogicNode('Jose is a murderer.', [
        LogicNode('Jose has a means.'),
        LogicNode('Jose has a motive'),
        LogicNode('Jose has an opportunity.')
    ])
], prune=False, populate=False
)

example3_node_completion = LogicNode('Jose has a means.', [
    LogicNode('Jose has access to heroin.'),
    LogicNode('Jose knows how much heroin is needed for an overdose.'),
    LogicNode('Having access to heroin and knowing how much heroin is required to overdose implies you could have intentionally given the victim a dose of lethal heroin providing a means for murder.', fact_type=LogicNodeFactType.COMMONSENSE)
])


example_trees = [example1_tree, example2_tree, example3_tree]
example_node_completions = [example1_node_completion, example2_node_completion, example3_node_completion]
example_descriptions = [example1_description, example2_description, example3_description]


def main():
    # CACHE
    cache.enable()

    # PARAMS (if not with a comment, look at the Murder Mystery dataset class for more info.)

    max_examples = 1
    tree_depth = 3

    max_number_of_suspects = 2
    max_structure_completion_retries = 3
    max_num_suspicious_facts = 1

    use_validators = True

    out_file = OUTPUT_FOLDER / 'custom_murder_mysteries.json'
    if out_file:
        out_file.parent.mkdir(exist_ok=True, parents=True)

    dataset = []

    total_cost = 0

    # Models we foudn helpful to use.  In our finalized dataset, we only used gpt4.
    gpt35 = OpenAIModel(engine='gpt-3.5-turbo', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=1500, num_samples=1, prompt_cost=0.0015/1000, completion_cost=0.002/1000)
    gpt16k35 = OpenAIModel(engine='gpt-3.5-turbo-16k', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.003/1000, completion_cost=0.004/1000)
    gpt4 = OpenAIModel(engine='gpt-4', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.03/1000, completion_cost=0.06/1000)

    model_to_use = gpt16k35

    creator = MurderMysteryDataset()

    madlib = Madlib(
        {
            "male_names": ROOT_FOLDER / 'domain_seed/male_names.json',
            "female_names": ROOT_FOLDER / 'domain_seed/female_names.json',
            "male_relationships": ROOT_FOLDER / 'domain_seed/male_relationships.json',
            "female_relationships": ROOT_FOLDER / 'domain_seed/female_relationships.json',
            "motives": ROOT_FOLDER / 'domain_seed/strong_motives.json',
            "murder_weapons": ROOT_FOLDER / 'domain_seed/murder_weapons.json',
            "relationships": ROOT_FOLDER / 'domain_seed/relationships.json',
            "crime_scenes": ROOT_FOLDER / 'domain_seed/crime_scenes.json',
            'red_herrings': ROOT_FOLDER / 'domain_seed/suspicious_facts.json',
        }
    )

    previously_sampled_items = []

    # Resume from a previous run. (should be a Path object)
    resume_file = None
    if resume_file and resume_file.exists():
        data = json.load(resume_file.open('r'))
        previously_sampled_items.extend([[x["victim"], x["crime_scene"], x["murder_weapon"]] for y in data for x in [y['questions'][0]['intermediate_data'][0]['victim_info']]])
        dataset = data

    # CREATION LOGIC
    for example_idx in range(max_examples):
        print(f"STORY: {example_idx+1}")

        # Setup Scenario (MadLib)
        constant_sampled_items = [['male_names', 'female_names'], 'crime_scenes', 'murder_weapons']
        constant_sampled_names = ['victim', 'crime_scene', 'murder_weapon']
        variable_sampled_items = [['male_names,male_relationships', 'female_names,female_relationships'], 'motives', 'crime_scenes']
        variable_sampled_names = ['suspect', 'role', 'motive', 'alibi']

        description_string = "Victim: {victim}\nCrime Scene: {crime_scene}\nMurder Weapon: {murder_weapon}"
        variable_string = 'Suspect: {suspect}\nRole in story: {role}\nThe suspect\'s motive: {motive}'

        victim_string, victim_dict, sampled = creator.sample_madlib(madlib, constant_sampled_items, previously_sampled=previously_sampled_items, description_string_format=description_string, sampled_item_names=constant_sampled_names)
        victim_dict = victim_dict[0]
        previously_sampled_items = sampled

        suspect_strings, suspect_dicts, _ = creator.sample_madlib(madlib, variable_sampled_items, previously_sampled=[[None,None,None,victim_dict['crime_scene']]], n_samples=max_number_of_suspects, description_string_format=variable_string, sampled_item_names=variable_sampled_names)

        _, suspicious_fact_dicts, _ = creator.sample_madlib(madlib, ['red_herrings'], n_samples=max_num_suspicious_facts * len(suspect_dicts), description_string_format='{red_herrings}')
        random.shuffle(suspicious_fact_dicts)
        for s in suspect_dicts:
            s['red_herrings'] = []
            for n in range(max_num_suspicious_facts):
                s['red_herrings'].append(suspicious_fact_dicts.pop()['red_herrings'])

        scenario = f'{victim_string[0]}\n'
        d = f'{scenario}'.strip()
        for idx, s in enumerate(suspect_strings):
            suspect_dicts[idx]['description'] = f"{scenario}{s}".strip()
            d += f"\n\n{s}\nRed herring: {suspect_dicts[idx]['red_herrings'][0]}"

        # Victim dict should have the victim name, crime scene, and murder weapon (things specific to the victim)
        # Suspect dicts should have the the name of the victim, their role in the story, their suspicious fact and motive.

        suspect_trees = creator.create_suspect_trees(
            model_to_use,
            victim_dict,
            suspect_dicts,
            example_trees,
            example_node_completions,
            example_descriptions,
            depth=tree_depth,
            bf_factor={2: 1.0},
            chance_to_prune=0.0,
            chance_to_prune_all=0.0,
            max_num_of_suspicious_facts=max_num_suspicious_facts,
            max_retries_on_error=max_structure_completion_retries,
            retry_model=model_to_use,
            progress_bar=True,
            use_validators=use_validators,
            model_validator_model=gpt4,
            model_validator_early_escape_model=gpt16k35,
            test_completion_prompt=False
        )

        suspect_trees = creator.create_chapter_trees(suspect_trees, max_num_of_suspicious_facts=max_num_suspicious_facts)

        suspect_trees = creator.create_chapter(model_to_use, suspect_trees, validate_model=model_to_use)

        # Because we only created chapters of the murder, we need an introduction to it.  Here we create a prompt to do that.
        sus_strings = ", ".join([x['suspect_info']['suspect'] for x in suspect_trees])
        intro_prompt = f"Create an intro for this murder mystery.  It should only be 1 or 2 sentences.  Only write the intro nothing else. \n\nScenario:\n{victim_dict['victim']} was killed with a {victim_dict['murder_weapon']} at a {victim_dict['crime_scene']}. Detective Winston is on the case, interviewing suspects. The suspects are {sus_strings}.\n\nOutput:\n"
        intro, _ = creator.inference(intro_prompt, model_to_use)

        # Iterate through the suspects (the curr suspect is the murderer)
        for sidx in range(len(suspect_trees)):
            murderer_idx = sidx

            _suspect_trees = copy.deepcopy(suspect_trees)

            for sidx, s in enumerate(_suspect_trees):
                _suspect_trees[sidx]['used_chapter'] = _suspect_trees[sidx][
                    'innocent_chapter' if sidx != murderer_idx else 'murderer_chapter']
                _suspect_trees[sidx]['used_tree'] = _suspect_trees[sidx][
                    'innocent_tree' if sidx != murderer_idx else 'murderer_tree']
                _suspect_trees[sidx]['is_murderer'] = sidx == murderer_idx

            chapters = [(x['suspect_info']['suspect'], x['used_chapter'].strip()) for x in _suspect_trees]
            random.shuffle(chapters)

            story = f"{intro}\n\n" + "\n\n".join([x[1] for x in chapters])

            choices = [x['suspect_info']["suspect"] for x in _suspect_trees]

            call_cost = gpt35.total_cost + gpt16k35.total_cost + gpt4.total_cost
            total_cost += call_cost
            print(f'EXAMPLE COST: {call_cost:.2f} | TOTAL COST SO FAR: {total_cost:.2f}')
            gpt35.total_cost = 0.0
            gpt16k35.total_cost = 0.0
            gpt4.total_cost = 0.0

            safe_suspects_dict = [{k: v.to_json() if isinstance(v, LogicTree) else v for k, v in x.items()} for x in _suspect_trees]
            dataset.append(
                creator.create_dataset_question_object(
                    context=story,
                    questions=['Who is the most likely murderer?'],
                    answers=[murderer_idx],
                    choices=[choices],
                    intermediate_trees=[[x['used_tree'] for x in _suspect_trees]],
                    intermediate_data=[[{'suspect_info': safe_suspects_dict, 'victim_info': victim_dict, 'story_hash_id': hash(intro)}]]
                )
            )

            if out_file:
                json.dump(dataset, out_file.open('w'))

    if out_file:
        json.dump(dataset, out_file.open('w'))

    print(f"TOTAL COST: {total_cost} | {total_cost / max_examples} per example.")


if __name__ == "__main__":
    main()
"""
RUN THIS FILE TO CREATE AWESOME STORIES USING AN LLM :)

Go to the main() function for arguments/control over the dataset creation.

NOTE: Expects your openai api key to be in the environment.  "OPENAI_API_KEY=api_key python script.py" (if you are using
openai LLMs)

NOTE: By default, datasets go into "{ROOT_FOLDER}/output_datasets/{dataset_name}.json"
"""

import pprint

from jsonlines import jsonlines
import json
import sys
from pathlib import Path
import random
from functools import partial

random.seed(0)

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.madlib.madlib import Madlib
from src.utils.paths import OUTPUT_FOLDER

from src.dataset_types.team_allocation import TeamAllocationDataset

import random
from itertools import combinations


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
                cLogicNode("If you trick someone into looking else where, they cannot see what happens in the other direction.")
            ])
        ])])


example2_description = 'Your dog has just pooped on the neighbours yard.  The neighbour glares in your direction and comes forward... he says "Hey you! What do you think you are letting your dog do on my nice yard right here!"'
example2_tree = LogicTree(
    nodes=[
        LogicNode('Punch the neighbour square in the nose.', [
                LogicNode('It\'ll look cool and this is a pro.', [
                    LogicNode('People think fighting is cool where you live.'),
                    LogicNode('You would be fighting.'),
                    LogicNode('Doing something people think is cool will make you cool too.',
                              fact_type=LogicNodeFactType.COMMONSENSE)
                ]),
                LogicNode('It\'ll look cool unless...', [])
            ]),
        LogicNode('Say, "I am so sorry mr. I am trying to train him."'),
        LogicNode('You feel threatened by the neighbour and think he might hurt you.'),
        LogicNode('The neighbour would leave you alone.')
        ], prune=False, populate=False
)
example2_node_completion_tree = LogicNode('Punch the neighbour square in the nose.',
                                     [LogicNode('It\'ll look cool unless...', [
                                        LogicNode('You could harm your neighbour.'),
                                        LogicNode('You are unprovoked'),
                                        cLogicNode('It\'s not cool if you hurt someone unprovoked.')
                                    ])])


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


example_descriptions = [example1_description, example2_description, example3_description]
example_trees = [example1_tree, example2_tree, example3_tree]
example_node_completions = [example1_node_completion_tree.children[0].children[0], example3_node_completion, example1_node_completion_tree.children[0]]


def main():
    cache.enable()
    creator = TeamAllocationDataset()

    gpt35 = OpenAIModel(engine='gpt-3.5-turbo', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=1500, num_samples=1, prompt_cost=0.0015 / 1000, completion_cost=0.002 / 1000)
    gpt16k35 = OpenAIModel(engine='gpt-3.5-turbo-16k', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.003 / 1000, completion_cost=0.004 / 1000)
    gpt4 = OpenAIModel(engine='gpt-4', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.03 / 1000, completion_cost=0.06 / 1000)

    model_to_use = gpt4

    # PARAMS (if not with a comment, look at the Team Allocation dataset class for more info.)

    tree_depth = 2

    max_examples = 1

    structure_retries = 1


    verbose = False

    out_file = OUTPUT_FOLDER / 'team_allocation.json'
    dataset = []
    previous_samples = [['']]

    # Sample a description/scenario to build the story from
    madlib = creator.build_madlib(
        model_to_use,
        things_to_create_names=['scenario_descriptions'],
        things_to_create_description=[
            "Scenarios where you have to assign people to groups that perform different tasks and the skills required to solve those tasks are fairly different, only give the scenario and DO NOT number them"
        ],
        examples_of_things_to_create=[
            [
                "You and your roommates want to make a video game, how should you assign each of your roommates so that the action video game is made.",
                "A paper deadline is coming up and since you are the supervisor of the lab, you must assign each graduate student efficiently to meet the deadline.",
                "They never said campaigning was easy, you want to be elected for President of the United States, so you have to assign your team effectively to help your chances.",
                "You are planning the perfect heist, how do you build your team such that the heist goes off without a hitch."
            ]
        ]
    )

    total_cost = 0.0

    idx = 0
    max_idx = int(max_examples * 1.5)
    real_idx = 0
    while idx < max_examples and real_idx < max_idx:
        real_idx += 1
        print(f"EXAMPLE {idx + 1}")

        descriptions, _, previous_samples = creator.sample_madlib(madlib, ['scenario_descriptions'],
                                                                  '{scenario_descriptions}',
                                                                  previously_sampled=previous_samples)
        description = descriptions[0]

        # Flesh out the description to include peoples names and skill/task names.
        prompt = f'''
Use the given scenario description to create a list of three people, two tasks, and two skills.  Each skill should be associated with one of the tasks, and it should be that each skill is unique and orthogonal to the other and it's assigned task.

Rules
 1) Never indicate in someones name their title or that they are good at any particular skill or task.  For example, never say "Dr. Bob" as this implies they should be in charge of medical tasks.

Here's an example

Description: A heavy flux of customers walk into the coffee bar, you have to assign workers to the register and others to be baristas to handle the flow.

Output:

People: Sarah; Luis; John;
Tasks: Barista; Cashier
Skills: Can make coffee; Can handle customers

Your turn!

Description: {description}

Output:

        '''.strip()
        output, _ = creator.inference(prompt, model_to_use)

        cost = gpt35.total_cost + gpt16k35.total_cost + gpt4.total_cost
        total_cost += cost
        gpt16k35.total_cost = 0.0
        gpt35.total_cost = 0.0
        gpt4.total_cost = 0.0

        people = []
        skills = []
        tasks = []

        # Parse the output into lists of people/tasks/skills.
        try:
            for line in output.split('\n'):
                if line.startswith('People:'):
                    people.extend([x.strip() for x in line.replace('People:', '').strip().split(';') if x != ''])
                elif line.startswith('Tasks:'):
                    tasks.extend([x.strip() for x in line.replace('Tasks:', '').strip().split(';') if x != ''])
                elif line.startswith('Skills:'):
                    skills.extend([x.strip() for x in line.replace('Skills:', '').strip().split(';') if x != ''])

        except Exception as e:
            # On failure, retry the whole loop.
            print("ERROR")
            print(e)
            continue

        # Create the skill / relationships scores and then generate the fact set F (from paper)
        people_levels, best_pair, all_pairs = creator.build_assignment(people)
        facts = creator.create_facts(people_levels, people, tasks)

        random.shuffle(facts)
        random.shuffle(all_pairs)

        tree = creator.create_fact_trees(
            model_to_use, facts, tasks, description, example_trees, example_node_completions,
            example_descriptions, depth=tree_depth, bf_factor={2:1.0}, chance_to_prune=0.0, chance_to_prune_all=0.0,
            max_retries_on_error=structure_retries, progress_bar=True, test_complete_structure_prompt=False,
            retry_model=model_to_use
        )

        def fact_str(t):
            facts = list(sorted(list(set([x.value for x in t.get_facts()]))))
            facts_str = "\n".join([f'- {x}' for x in facts])
            return facts_str

        facts = fact_str(tree)

        # Create the story.  The list of facts are rather simple and short so we don't need chaptering here.
        prompt = f'''
You will write a short story given the description of a scenario and a list of facts. You must include every fact into the story.  

You should take the role of a manager or leader, someone who can assign people to skills.  

Your job is to find the perfect assignment of each person to a single skill.  You will not say what this perfect assignment is, that is left to the reader to decide.  

Instead, you must write out the narrative as if it were a short story. Make this story coherent and flow nicely.  

Most importantly, make this story interesting! 

Start with an introduction.  
Introduce each person:
- {people[0]}
- {people[1]}
- {people[2]}

And the tasks the manager has to assign them to (mention these by name in the beginning, i.e. "And they all had to be assigned to cleaning and sales" or something.):
- {tasks[0]}
- {tasks[1]}

Scenario description: {description}
Facts that you must include in your story:
{facts}

Output:'''

        if verbose:
            print("--- MAIN STORY PROMPT ---")
            print(prompt)
        output, _ = creator.inference(prompt, model_to_use)
        if verbose:
            print("=== MAIN STORY OUT ===")
            print(output)

        paragraphs = output.split('\n\n')

        # Often times, however, the first paragraph just jumps right into the main story - but we want it to introduce
        # the task a bit more.  Prompt engineering might be able to do this in one call.
        fix_p0_prompt = f'''
I am writing a story that is similar to a word problem where a manager has to assign the right worker to a task.  Here's the full story:

{output}

---

I want you to rewrite the introduction paragraph to introduce each person and the tasks.  Do not assign people to a task (this is the question I want to ask the readers), just introduce them.  Rewrite the introduction:

{paragraphs[0]}

Make sure it includes:

Mentions to these peoples names (do not describe them)
- {people[0]}
- {people[1]}
- {people[2]}

And the tasks the manager has to assign them to (mention these by name in the beginning, i.e. "And they all had to be assigned to cleaning and sales" or something.)
- {tasks[0]}
- {tasks[1]}

It should be short.  No longer than the original introduction.
        '''.strip()

        if verbose:
            print('--- FIX P0 PROMPT ---')
            print(fix_p0_prompt)

        paragraphs[0], _ = creator.inference(fix_p0_prompt, model_to_use, temperature=0.2)

        if verbose:
            print("=== FIXED P0 ===")
            print(paragraphs[0])

        output = "\n\n".join(paragraphs)

        question = "Given the story, how would you uniquely allocate each person to make sure both tasks are accomplished efficiently?"

        choices = [
            f"{tasks[0]}: {all_pairs[0][0][0]}, {tasks[1]}: {all_pairs[0][1][0]} and {all_pairs[0][1][1]}",
            f"{tasks[0]}: {all_pairs[1][0][0]}, {tasks[1]}: {all_pairs[1][1][0]} and {all_pairs[1][1][1]}",
            f"{tasks[0]}: {all_pairs[2][0][0]}, {tasks[1]}: {all_pairs[2][1][0]} and {all_pairs[2][1][1]}",
        ]

        if verbose:
            print(f"EXAMPLE {idx + 1} OUTPUT:")
            print(output)
            print('\n\n')
            print(question)
            print('\n\n')
            print("\n".join([f'{cidx+1} - {x}' for cidx, x in enumerate(choices)]))

        gold_idx = all_pairs.index(best_pair)


        idx += 1
        dataset.append(
            creator.create_dataset_question_object(
                output,
                questions=[question],
                answers=[gold_idx],
                choices=[choices],
                intermediate_trees=[[tree]],
                intermediate_data=[[{'tasks': tasks, 'matrix': people_levels, 'best_pair': best_pair}]]
            )
        )

        cost = gpt4.total_cost + gpt16k35.total_cost + gpt35.total_cost
        total_cost += cost
        gpt4.total_cost = 0.0
        gpt35.total_cost = 0.0
        gpt16k35.total_cost = 0.0

        print(f"Cost of example: {cost} | Total cost so far {total_cost}")

    out_file.parent.mkdir(exist_ok=True, parents=True)
    json.dump(dataset, out_file.open('w'))


if __name__ == "__main__":
    main()

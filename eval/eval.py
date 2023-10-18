import json
import math
import sys
from pathlib import Path
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import collections

random.seed(0)

from src import cache
from src.model import OpenAIModel, HFModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.madlib.madlib import Madlib
from src.utils.paths import OUTPUT_FOLDER

from eval.icl.murder_mystery_solved_ex import murder_mystery_solved_ex
from eval.icl.object_placements_solved_ex import object_placements_solved_ex
from eval.icl.team_allocation_solved_ex import team_allocation_solved_ex


def main():
    """
    This script will run a bunch of models over the datasets created in MuSR.  Furthermore, it can test different
    prompting strategies (at least the ones mentioned in the paper).

    There are also a ton of parameters you can set to control the eval and visualize intermediate answers etc.
    """

    # CACHE
    cache.enable()

    DATASETS_FOLDER = OUTPUT_FOLDER

    gpt4 = OpenAIModel(engine='gpt-4', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.03/1000, completion_cost=0.06/1000)
    gpt3516k = OpenAIModel(engine='gpt-3.5-turbo-16k', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.003/1000, completion_cost=0.004/1000)
    gpt35 = OpenAIModel(engine='gpt-3.5-turbo', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=700, num_samples=1, prompt_cost=0.0015/1000, completion_cost=0.002/1000)

    # NOTE: change the filenames as needed! (to point at whatever thing you want to test.  It is important to keep the system prompt/hint/etc. all the same for each specific type of domain though).
    murder_mysteries = {'name': 'murder mysteries', 'file_name': 'murder_mysteries.json', 'ex': murder_mystery_solved_ex, 'system_prompt': 'You are a helpful assistant that will answer the questions given by the user.', 'hint': 'Before selecting a choice, explain your reasoning step by step. The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.\n\nIf you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established.'}
    object_placements = {'name': 'object placements', 'file_name': 'object_placements.json', 'ex': object_placements_solved_ex, 'skip_ablated': True, 'system_prompt': 'You are a helpful assistant that will answer the questions given by the user.', 'ablation_depth_modifier': 2, 'hint': 'Based on this story, we want to identify where someone believes that a certain object is at the end of the story. In order to do that, you need to read the story and keep track of where they think the object is at each point. When an object is moved, the person may observe its new location if they saw it move.\n\nTo see where an object ends up, they must be able to see the location that it moves to and not be too distracted by what they are doing. If they do not observe the object moving, then they will still believe it to be in the last location where they observed it.', 'hint_before_question': True}
    team_allocation = {'name': 'team allocation', 'file_name': 'team_allocation.json', 'ex': team_allocation_solved_ex, 'system_prompt': 'You are a helpful assistant that will answer the questions given by the user.', 'hint': 'The story should allow you to determine how good each person is at a skill. Roughly, each person is either great, acceptable, or bad at a task. We want to find an optimal assignment of people to tasks that uses their skills as well as possible. In addition, one task will have to have two people assigned to it. The effectiveness of their teamwork (great team, acceptable team, or bad team) also impacts the overall quality of the assignment.\n\nWhen two people need to work on a task and one is bad at it, they don’t necessarily benefit from the other person being good, unless they work well together.\n\nWith different strengths, weaknesses, and interpersonal dynamics at play, you should allocate your team to find the single assignment to ensure that the tasks overall are completed as effectively as possible.\n\n'}

    ablations = [
        # {'prompt': 'regular', 'name': 'regular'},
        # {'prompt': 'cot', 'name': 'cot'},
        {'prompt': 'cot+', 'name': 'cot+'},
        # {'prompt': 'cot+', 'name': 'cot+ 1-shot', 'self_consistency_n': 1, 'use_example': True},
        # {'prompt': 'cot+', 'name': 'cot+ s.c.', 'self_consistency_n': 3},
        # {'prompt': 'cot+', 'name': 'cot+ s.c. 1-shot', 'self_consistency_n': 3, 'use_example': True},
    ]

    datasets_to_test = [
       murder_mysteries,
       object_placements,
       team_allocation
    ]

    models_to_test = [
        {'model': gpt4},
        # {'model': gpt3516k},
        # {'model': gpt35},
        # {'model': HFModel('meta-llama/Llama-2-7b-hf', load_in_4bit=True), 'system_prompt_template': "{system_prompt}\n\n{prompt}"},
        # {'model': HFModel('meta-llama/Llama-2-7b-chat-hf', load_in_4bit=True), 'system_prompt_template': "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{prompt}[/INST]"},
        # {'model': HFModel('meta-llama/Llama-2-13b-chat-hf', load_in_4bit=True), 'system_prompt_template': "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{prompt}[/INST]"},
        # {'model': HFModel('meta-llama/Llama-2-70b-chat-hf', load_in_4bit=True), 'system_prompt_template': "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{prompt}[/INST]"},
        # {'model': HFModel('lmsys/vicuna-7b-v1.5', load_in_4bit=True), 'system_prompt_template': "{system_prompt}\n\nUSER: {prompt}\nASSISTANT: "},
        # {'model': HFModel('lmsys/vicuna-13b-v1.5', load_in_4bit=True), 'system_prompt_template': "{system_prompt}\n\nUSER: {prompt}\nASSISTANT: "},
        # {'model': HFModel('lmsys/vicuna-33b-v1.3', load_in_4bit=True), 'system_prompt_template': "{system_prompt}\n\nUSER: {prompt}\nASSISTANT: "},
    ]

    sample_size = None  # How many examples to test on
    offset = 0 # Offset the dataset
    exclude_contrastive_examples = False  # For murder mysteries, exclude stories that are the same but with the murderer suspect flipped (will only include 1 story per)
    reverse_contrastive_sample = False  # Flip which mystery you are looking at that's unique
    verbose = False  # Print stuff out
    human_verbose = False  # Useful for annotation stuff (just removes COT lingo)
    log_gold_answer = False  # Print the gold answer
    log_tree = False # Print the tree for the answer
    skip_inference = False # Don't actually call the model
    progress_bar = True # Show a progress bar
    randomize = True # Shuffle stuff.

    datasets = {}
    run_data = {}
    out_file = None # Can save results to a json file, should be a path object.

    run_cost = 0.0

    for model_info in models_to_test:
        m = model_info['model']
        model_name = m.model_name if isinstance(m, HFModel) else m.engine

        for d in datasets_to_test:

            for a in ablations:
                total_cost = 0.0
                if isinstance(m, OpenAIModel):
                    m.total_cost = 0.0

                ablation_name = a['name']

                total = 0
                correct = 0

                answered_examples = []

                self_consistency_n = a.get('self_consistency_n', 1)

                if datasets.get(d["name"]):
                    dataset = datasets.get(d['name'])
                else:
                    _dataset = json.load((DATASETS_FOLDER / d.get("file_name", d.get('name', None))).open('r'))
                    if randomize:
                        random.shuffle(_dataset)

                    dataset = []
                    hashes_done = []
                    for _d in _dataset:
                        if exclude_contrastive_examples and _d['questions'][0].get('intermediate_data') and len(
                                _d['questions'][0].get('intermediate_data')) > 0 and \
                                _d['questions'][0]['intermediate_data'][0].get('story_hash_id'):
                            if _d['questions'][0]['intermediate_data'][0]['story_hash_id'] in hashes_done:
                                if reverse_contrastive_sample:
                                    dataset.append(_d)
                                else:
                                    continue
                            elif not reverse_contrastive_sample:
                                dataset.append(_d)
                            hashes_done.append(_d['questions'][0]['intermediate_data'][0]['story_hash_id'])
                        else:
                            dataset.append(_d)

                    dataset = dataset[offset:offset+min(len(dataset), sample_size) if sample_size else sample_size]

                    datasets[d['name']] = dataset

                pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f'RUNNING | {model_name} | {d["name"]} | {ablation_name} | {correct} / {total} | (run cost = {run_cost:.2f}, iteration cost = {total_cost:.2f})', disable=not progress_bar)

                for eidx, example in pbar:

                    answered_questions = []

                    context = example['context']
                    questions = example['questions']

                    for qidx, question in enumerate(questions):

                        answer_outs = []
                        raw_answers = []
                        choices = "\n".join([f'{idx + 1} - {x}' for idx, x in enumerate(question["choices"])])
                        gold_answer = question["answer"] + d.get('answer_index_modifier', 1)

                        for scidx in range(self_consistency_n):

                            ex_str = ''
                            if a.get('use_example') and d.get('ex'):
                                ex_str = 'Here is an example of solving the task:\n\n' + d.get('ex') + '\n\nThis is the end of the example. The real task is below.\n\n---\n\n'

                            prompt_style = a.get('prompt')
                            if prompt_style == 'regular':
                                prompt = f'{ex_str}{context}\n\n{question["question"]}\n\nPick one of the following choices:\n{choices}\n\nYou must pick one option. Finally, the last thing you generate should be "ANSWER: (your answer here, include the choice number)"'
                            elif prompt_style == 'cot':
                                prompt = f'{ex_str}{context}\n\n{question["question"]}\n\nPick one of the following choices:\n{choices}\n\nYou must pick one option. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, include the choice number)"'
                            elif prompt_style == 'cot+':
                                if d.get("hint_before_question"):
                                    prompt = f'{ex_str}{context}\n\n{d["hint"]}\n\n{question["question"]}\n\nPick one of the following choices:\n{choices}\n\nYou must pick one option. Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"'
                                else:
                                    prompt = f'{ex_str}{context}\n\n{question["question"]}\n\nPick one of the following choices:\n{choices}\n\nYou must pick one option. {d["hint"]} Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"'
                            else:
                                if len(question["intermediate_trees"]) == 0 or d.get('skip_ablated'):
                                    continue

                                prompt = f'{ex_str}Answer the following questions given the list of facts per answer choice.\n\n'
                                for c, t in zip(choices.split('\n'), question['intermediate_trees']):
                                    facts = list(set([x.value for x in LogicTree.from_json(t).get_facts(include_cs=a.get('include_cs', False), include_deductions_past_level=-1, no_facts_after_depth=a.get('no_facts_after_depth', 3) + d.get('ablation_depth_modifier', 0))]))
                                    facts = list(sorted(facts)) if d.get('allow_sorted_facts', True) else facts
                                    facts_str = "\n".join([f'- {x}' for x in facts])
                                    prompt += f'Facts for Choice {c}:\n{facts_str}\n\n'
                                prompt += f'Given the list of facts per answer choice answer the following question\n\n{question["question"]}\n\nPick one of the following choices:\n{choices}\n\nYou must pick on option.  After you have found the answer, say it in this format "ANSWER: (your answer here, include the choice number)"'

                            if verbose:
                                print(f'EX: {eidx +1}.{qidx +1}')

                                if human_verbose:
                                    print(prompt.replace(' Explain your reasoning step by step before you answer.', ''))
                                else:
                                    print(prompt)
                            if log_gold_answer:
                                print(gold_answer)
                            if log_tree:
                                for i in question['intermediate_trees']:
                                    print(LogicTree.from_json(i).print_for_gpt(pad_space=1, pad_char='> '))
                            if skip_inference:
                                continue

                            if isinstance(m, OpenAIModel):
                                raw = m.inference(prompt, system_prompt=d.get("system_prompt"))
                                output = raw.choices[0]['message']['content']
                            else:
                                if d.get("system_prompt") and model_info.get("system_prompt_template"):
                                    prompt = model_info.get("system_prompt_template").replace("{system_prompt}", d.get('system_prompt')).replace("{prompt}", prompt)

                                output = m.inference(prompt)

                            if verbose:
                                print("MODEL OUTPUT")
                                print(output)

                            try:
                                lines = [x.split('answer:')[-1].strip() for x in output.lower().split('\n') if 'answer:' in x and len(x.split('answer:')[-1].strip())>0]
                                answer = lines[-1]
                            except Exception as e:
                                answer = ''

                            randomly_selected = False
                            if not any([str(x+1) in answer for x in range(len(question["choices"]))]):
                                answer = random.sample([str(x+1) for x in range(len(question["choices"]))], 1)[0]
                                randomly_selected = True

                            if str(gold_answer) in answer:
                                answer_outs.append(str(gold_answer))
                                raw_answers.append({
                                    'qidx': qidx,
                                    'prompt': prompt,
                                    'output': output,
                                    'model_parsed_answer': answer,
                                    'trees': question['intermediate_trees'],
                                    'data': question['intermediate_data'],
                                    'randomly_selected': randomly_selected,
                                    'gold_answer': gold_answer,
                                    'correct': True
                                })
                            elif any([str(x+1) in answer for x in range(len(question["choices"]))]):
                                answer_outs.append([str(x+1) for x in range(len(question["choices"])) if str(x+1) in answer][0])
                                raw_answers.append({
                                    'qidx': qidx,
                                    'prompt': prompt,
                                    'output': output,
                                    'model_parsed_answer': answer,
                                    'trees': question['intermediate_trees'],
                                    'data': question['intermediate_data'],
                                    'randomly_selected': randomly_selected,
                                    'gold_answer': gold_answer,
                                    'correct': False
                                })
                            else:
                                raise Exception("ERROR: SHOULDN'T HIT")

                            if isinstance(m, OpenAIModel):
                                total_cost += m.total_cost
                                run_cost += m.total_cost

                                m.total_cost = 0.0

                        if len(answer_outs) == 0:
                            continue

                        most_common = collections.Counter(answer_outs).most_common()[0][0]
                        if most_common == str(gold_answer):
                            correct += 1
                            answered_questions.append([x for x in raw_answers if x['correct']][0])
                        elif most_common is None:
                            correct += 1 / len(choices)
                            answered_questions.append([x for x in raw_answers if x['model_parsed_answer'] is None])
                        else:
                            answered_questions.append([x for x in raw_answers if not x['correct'] and x['model_parsed_answer'] is not None])

                        total += 1

                        pbar.set_description(f'RUNNING | {model_name} | {d["name"]} | {ablation_name} | {correct} / {total} | (run cost = {run_cost:.2f}, iteration cost = {total_cost:.2f})')

                    answered_examples.append(answered_questions)

                model_data = run_data.get(f'{model_name}', {})
                dataset_data = model_data.get(d["name"], {})
                ablation_data = dataset_data.get(a["name"], {})
                ablation_data["examples"] = answered_examples
                ablation_data["correct"] = correct
                ablation_data["total"] = total
                dataset_data[a["name"]] = ablation_data
                model_data[d.get('name')] = dataset_data
                run_data[model_name] = model_data

                print(f'RUNNING | {model_name} | {d["name"]} | {ablation_name} | {correct} / {total} | {(correct / max(1,total))*100:.1f}', flush=True)

                if out_file:
                    json.dump(run_data, out_file.open('w'))

        if isinstance(m, HFModel):
            del m

if __name__ == "__main__":
    main()

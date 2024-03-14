from src.utils.paths import OUTPUT_FOLDER
from src.logic_tree.tree import LogicTree

from jsonlines import jsonlines
from pathlib import Path


def load_dataset(file: Path):
    dataset = jsonlines.open(str(file), 'r').read()

    for ex in dataset:
        questions = ex['questions']
        for q in questions:
           q['intermediate_trees'] = [LogicTree.from_json(x) for x in q['intermediate_trees']]

    return dataset


def check_murder_mystery_trees(dataset, specified_depth: int = 3):
    def min_depth(n):
        if len(n.children) == 0:
            return 1
        return 1 + min([min_depth(x) for x in n.children])

    top_level_branches_not_finished = {
    }
    for ex in dataset:
        questions = ex['questions']
        for q in questions:
            trees = q['intermediate_trees']

            for t in trees:
                top_level_children = t.nodes[0].children

                for c in top_level_children:
                    if min_depth(c) + 1 != specified_depth:
                        if 'suspicious' in c.value:
                            val = 'suspicious fact'
                        else:
                            val = ' '.join(c.value.split(' ')[1:])
                        ct = top_level_branches_not_finished.get(val, 0)
                        ct += 1
                        top_level_branches_not_finished[val] = ct
    return top_level_branches_not_finished



if __name__ == "__main__":
    from pprint import pprint

    file = OUTPUT_FOLDER / 'murder_mystery__gpt3516k_v3.json'
    # file = OUTPUT_FOLDER / 'murder_mystery.json'
    dataset = load_dataset(file)

    stats = check_murder_mystery_trees(dataset)

    for idx in range(len(dataset)):
        print(f"EXAMPLE: {idx+1}")
        for tree_idx in [0, 1]:
            print(f"TREE: {tree_idx+1}")
            print(dataset[idx]['questions'][0]['intermediate_trees'][tree_idx].print_for_gpt())
            print('\n'*2)
            print('---' * 10)
        print(f"STORY FOR {idx+1}")
        print(dataset[idx]['context'])

        print('\n'*20)


    # pprint(stats)

'''V1
{'has a means.': 29,
 'has a motive.': 4,
 'has an opportunity.': 69,
 'suspicious fact': 3}
'''

'''V2
{'has a means.': 8,
 'has a motive.': 4,
 'has an opportunity.': 3,
 'suspicious fact': 2}
'''

'''V3
{'has a means.': 1, 'has a motive.': 1}
'''

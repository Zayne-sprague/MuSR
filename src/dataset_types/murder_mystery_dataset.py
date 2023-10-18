from tqdm import tqdm
import sys
from typing import List, Dict, Union, Callable, Tuple
from functools import partial
import random
from copy import deepcopy

random.seed(0)

from src.madlib.madlib import Madlib
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.dataset_builder import DatasetBuilder
from src.model.openai import Model
from src.validators import StructureValidator, Validator, ForbiddenTextValidator, ModelValidator

# We use this to overwrite the original _base_completion_prompt_intro_ in dataset_builder.py
# Specifically, this is used when we are creating the means, motive, and opportunity branches for a suspect.
_mm_completion_prompt_intro_ = '''
Your task is to generate a logic tree for a story, as shown in the example. In this tree, each fact should be deduced from its immediate children. If a deduced fact already has a name, do not overwrite it.

Type of story:

We are creating a murder mystery. A murder mystery needs to have a complex web of evidence that leads to a means, motive, and opportunity for a suspect, which will make them a likely murderer. When writing a murder mystery, the story should take the point of view of the detective.  Evidence should be collected through investigation, including things like interrogation, hearing conversations, reading past criminal records, looking at mail or trash, and other normal modes of detecting evidence.

1. Each fact in the tree must follow via logical deduction from its children.
2. All Fact From Story nodes and the Commonsense Knowledge node must be relevant to the deduction they yield.
3. Each root fact is labeled with a source (Fact from Story or Commonsense Knowledge).
4. A Fact From Story should be a statement about a character, place, or object in the story.
5. Commonsense Knowledge should be a fact that most people will know and agree with. It should not explicitly reference any characters in the story.
6. Commonsense Knowledge should be used as a deduction rule that when the sibling facts are applied to it they yield the parent deduced fact.
7. The tree you generate must match the structure of the tree I give you.

A suspect has a means when they have access to the murder weapon.
A suspect has a motive when the suspect has reason to kill the victim.
A suspect has an opportunity when they were at the crime scene.

You must adhere to the parent nodes facts even if they are outlandish or fantastical. 
'''.strip()

# We use this to overwrite the original _base_completion_prompt_intro_ in dataset_builder.py
# Specifically, this is used when we are creating the suspicious facts tree for each suspect.
_mm_suspicious_prompt_intro_ = '''
Your task is to generate a logic tree for a story, as shown in the example. In this tree, each fact should be deduced from its immediate children. If a deduced fact already has a name, do not overwrite it.

Type of story:

We are creating a murder mystery. A murder mystery needs to have a complex web of evidence that include suspicious facts which are ultimately red herrings (they don't make the person guilty).

Here we are going to create Red Herrings, things that are suspicious, but do not make the suspect guilty.  You will not prove the suspect is guilty.

1. Each fact in the tree must follow via logical deduction from its children.
2. All Fact From Story nodes and the Commonsense Knowledge node must be relevant to the deduction they yield.
3. Each root fact is labeled with a source (Fact from Story or Commonsense Knowledge).
4. A Fact From Story should be a statement about a character, place, or object in the story.
5. Commonsense Knowledge should be a fact that most people will know and agree with. It should not explicitly reference any characters in the story.
6. Commonsense Knowledge should be used as a deduction rule that when the sibling facts are applied to it they yield the parent deduced fact.
7. The tree you generate must match the structure of the tree I give you.
'''.strip()

"""
ICL examples for the suspicious facts.  These are a bit unique and different than the ones used in the MMO branches
because we don't want suspicious facts to be in any way related to the MMO (nor influenced to be so).  
"""
cLogicNode = partial(LogicNode, fact_type=LogicNodeFactType.COMMONSENSE)

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


sf_example_descriptions = [example1_description, example2_description, example3_description]
sf_example_trees = [example1_tree, example2_tree, example3_tree]
sf_example_node_completions = [example1_node_completion_tree.children[0].children[0], example3_node_completion, example1_node_completion_tree.children[0]]



def create_story_prompt__facts_only(description: str, story_tree: LogicTree, pad_char: str = '> '):
    """
    This will create the prompt used for the story generation process.  We pull out leaf explicit facts and use those
    in a list to generate a chapter for the suspect.  We provide one ICL example in the prompt.

    :param description: describe the story
    :param story_tree: Expanded tree used for the chapter
    :param pad_char: (not used)
    :return:
    """
    facts = list(sorted(list(set([x.value for x in story_tree.get_facts()]))))
    random.shuffle(facts)
    facts_str = "\n".join([f'- {x}' for x in facts])

    return f"""
We are creating a murder mystery. A murder mystery needs to have a complex web of evidence for a suspect. When writing a murder mystery, the story should take the point of view of the detective.  Evidence should be collected through investigation, including things like interrogation, hearing conversations, reading past criminal records, looking at mail or trash, and other normal modes of detecting evidence.

We will give you a list of facts to use when writing your story.  You must include all the facts in the list. Never state deduced facts or conclusions. The story should stick to the fact list pretty closely.

You will write a chapter for the murder mystery. This should not introduce the murder or victim, but strictly follow a detective slowly interviewing and detecting the suspect.  Use the list below as a guide.  

Rules:
1. Only write the contents of the chapter, do not write a title or number it.  This text must be easily placed in a larger story.
2. Never say a suspect has a means
3. Never say a suspect has a motive
4. Never say a suspect has an opportunity
5. Never hint at a suspect having or not having a means, motive, or opportunity. 3. Never hint at or explicitly say a deduced fact.
6. Never say a suspect is a murderer.  It's a puzzle that the reader is supposed to guess and deduce!
7. Write the story from the perspective of a detective uncovering clues through various methods that normal detectives use (interrogation, notes, stake-outs, etc.)
8. Never make mental notes, point out suspicious facts, nor make connections.  Let the reader do this.

Include as many sentences as needed to write each fact from the list of facts.  Also include up to 10 sentences of dialogue.

Here is an example:

Suspect and crime information
Victim: Dora
Crime Scene: Remote forest
Murder Weapon: Knife
Suspect: Willard
Role in story: Grounds keeper
The suspect's motive: Religious sacrifice

You are detective Winston.
Facts you must include:
- A witness saw someone with a spaghetti face and green ears
- Willard is a groundskeeper for a local school
- Willard also provides services to nearby residents like house painting, lawn care, etc.
- Willard painted a nearby home green.
- Willard was in a horrible fire when he was a child.
- Willard's family has been in the local area for multiple generations
- Long ago, the local area had religious extremists around them, all participating in occult activities.
- Willard and his immediate family were all handymen of some sort.
- Willard believes in respecting his elders and ancestral history.
- Dora had written about joining a new church nearby.
- A friend of Dora mentioned worrying about Dora's involvement with a new cult-like group of friends.

Output:

Winston took a long drag from his cigarette while reviewing the crime scene photos. He was hardened to common grotesqueries of his line of work, but the murder of Dora sparked something in him... a calling into the void.

The only hard evidence he had in this case was an eyewitness of a monster... a spaghetti face, green-eared monster. That, and the fact that Dora had been exploring some new church... or perhaps a cult, depending on who you asked. Cults had a history of leaving a bad taste in people's mouths here... having some very dark pasts in the local area.

Winston put his cigarette out, placed the photos down, and set out for his following suspect interview with Willard.

The smell of fresh-cut grass washed over Winston as the local groundskeeper of the local elementary shut off his mower.  

"Howdy, Mister," - Willard said with a rich southern twang.

"Howdy..." Winston said carefully, trying to match the southern pleasantries he wasn't all too familiar with.

"You, Willard?" Winston inquired.

"Well, sure I am!" He chuckled, "Been him for a while now, I suppose."  

"You do a lot of work for the school here?"  

"Heh, yes, sir, I do. My family and I have been helping out the local schools and locals with standard chores." noticing Winston motioning for him to continue, Willard explained further, "You know, like painting houses, cutting people's lawn... well heck, just a couple days ago I painted that green house over yonder." Willard said, pointing out across the schoolyard.

Winston couldn't help but notice that Willard had a severe burn scar across the bottom half of his face, which had previously been hidden by sweat and grass shavings.

*Ring* *Ring* - Winston was getting called away, back to the office, so he had to make a quick excuse to get away.

"You know the way back to Highway 45?" Winston asked. "Sure," Willard replied with a grin, "You take two lefts here, then about 12 miles down, you'll hit it." 

Winston smiled, "You sure do know your way around here," he said. "Hah, well, sir, my families have been here for... for a long, long time. I take great pride in my ancestry and family history here..."

With that, Winston got in his car and left.

---

Your turn.

Suspect and crime information
{description}

You are detective Winston.

Facts you must include:
{facts_str}

Output:
    """.strip()


class MurderMysteryDataset(DatasetBuilder):
    """
    Wrapper of the datasetbuilder class for specific Murder Mystery functionality.

    Details specific to Murder Mysteries will be covered here.  Details about generic dataset building can be found in
    DatasetBuilder's file.
    """

    def create_suspect_trees(
            self,
            model: Model,
            victim_info: Dict[str, str],
            suspect_infos: List[Dict[str, str]],
            example_completion_trees: List[LogicTree],
            example_completion_nodes: List[LogicNode],
            example_completion_descriptions: List[str],
            depth: int = 4,
            bf_factor: Dict[int, float] = None,
            chance_to_prune_all: float = 0.45,
            chance_to_prune: float = 0.5,
            max_num_of_suspicious_facts: int = 1,
            max_retries_on_error: int = 1,
            retry_model: Model = None,
            progress_bar: bool = False,
            test_completion_prompt: bool = False,
            use_validators: bool = True,
            model_validator_model: Model = None,
            model_validator_early_escape_model: Model = None
    ):
        """
        Here we create the trees for each suspect.  A suspects tree will have a means, motive, opportunity, and
        suspicious fact(s) branch.

        :param model: See datasetbuilder
        :param victim_info: Dictionary of sampled items for the victim
        :param suspect_infos: Dictionary of sampled items for the suspects
        :param example_completion_trees: See datasetbuilder
        :param example_completion_nodes: See datasetbuilder
        :param example_completion_descriptions: See datasetbuilder
        :param depth: see LogicTree
        :param bf_factor: see LogicTree
        :param chance_to_prune_all: see LogicTree
        :param chance_to_prune: see LogicTree
        :param max_num_of_suspicious_facts: Number of suspicious facts per suspect.
        :param max_retries_on_error: See datasetbuilder
        :param retry_model: See datasetbuilder
        :param progress_bar: See datasetbuilder
        :param test_completion_prompt: See datasetbuilder
        :param use_validators: See datasetbuilder
        :param model_validator_model: For the model validators, which model should we use (confusing but look at model validator for more info)
        :param model_validator_early_escape_model: For the model validators, which early escape model should we use (confusing but look at model validator for more info)
        """

        suspect_trees = []

        victim = victim_info['victim']
        murder_weapon = victim_info['murder_weapon']
        crime_scene = victim_info['crime_scene']

        for suspect_info in suspect_infos:
            suspect_name = suspect_info['suspect']
            motive = suspect_info['motive']
            try:
                description = suspect_info['description']
            except Exception as e:
                print(e)

            # Start with the MMO for each suspect.
            root_node = [
                LogicNode(f'{suspect_name} is the murderer.', [
                    LogicNode(f'{suspect_name} has a means.'),
                    LogicNode(f'{suspect_name} has an opportunity.'),
                    LogicNode(f'{suspect_name} has a motive.')
                ], frozen=True, prunable=False)
            ]

            validators = [StructureValidator()]
            if use_validators:
                # Murder Mystery specific validators
                validators.append(ForbiddenTextValidator(
                    forbidden_words=[
                        ['has a means', crime_scene],
                        ['has a means', 'crime scene'],
                        ['has a means', 'opportunity'],
                        ['has a means', motive],
                        ['has a means', 'motive'],
                        ['has an opportunity', murder_weapon],
                        ['has an opportunity', 'murder weapon'],
                        ['has an opportunity', motive],
                        ['has an opportunity', 'weapon'],
                        ['has a motive', crime_scene],
                        ['has a motive', 'crime scene'],
                        ['has a motive', 'opportunity'],
                        ['has a motive', 'murder weapon'],
                        ['has a motive', murder_weapon],
                        ['has a motive', 'weapon'],
                    ],
                    reason_why="We are trying to create a murder mystery that follows strict logic deduction.  If we prove an opportunity in the means branch of the entailment tree it will contaminate the process later on where we may want to make the suspect innocent by removing their \"opportunity\".  Therefore, all the facts under the means, motive, and opportunity branches must only pertain to the respective means, motive, and opportunity facts."
                ))

                if model_validator_model:
                    validators.extend([
                        ModelValidator(
                            model_validator_model,
                            f"We are writing a murder mystery and to do so we are creating a narrative guide of evidence.  We are proving a motive (reason to kill) right now.  Does this deduction in any way prove or help to prove a means (access to the murder weapon) or an opportunity (access to the crime scene) given the description of the mystery below.\n\n{description}",
                            "We are proving a motive (reason to kill) only.  We do not want to prove an opportunity or means because that could make the reasoning in our murder mystery complicated.",
                            conditional='has a motive',
                            early_escape_model=model_validator_early_escape_model
                        ),
                        ModelValidator(
                            model_validator_model,
                            f"We are writing a murder mystery and to do so we are creating a narrative guide of evidence.  We are proving a means (access to the murder weapon) right now.  Does this deduction in any way prove or help to prove a motive (reason to kill) or an opportunity (access to the crime scene) given the description of the mystery below.\n\n{description}",
                            "We are proving a means (access to the murder weapon) only.  We do not want to prove an opportunity or motive because that could make the reasoning in our murder mystery complicated.",
                            conditional='has a means',
                            early_escape_model=model_validator_early_escape_model
                        ),
                        ModelValidator(
                            model_validator_model,
                            f"We are writing a murder mystery and to do so we are creating a narrative guide of evidence.  We are proving an opportunity (access to the crime scene) right now.  Does this deduction in any way prove or help to prove a motive (reason to kill) or an means (access to the murder weapon) given the description of the mystery below.\n\n{description}",
                            "We are proving an opportunity (access to the crime scene) only.  We do not want to prove an means or motive because that could make the reasoning in our murder mystery complicated.",
                            conditional='has an opportunity',
                            early_escape_model=model_validator_early_escape_model
                        )
                    ])

            # Create a template tree, then fill it out.
            tree = self.complete_structure(
                self.build_structure(
                    depth=depth,
                    bf_factor=bf_factor,
                    chance_to_prune_all=chance_to_prune_all,
                    chance_to_prune=chance_to_prune,
                    root_nodes=root_node
                ),
                model,
                description=description,
                completion_prompt_fn=self.create_completion_prompt(example_completion_trees, example_completion_nodes,
                                                                  example_completion_descriptions, intro=_mm_completion_prompt_intro_, because_clause_after=0, use_complex_facts=False),
                max_retries_on_error=max_retries_on_error,
                inplace=True,
                retry_model=retry_model,
                progress_bar=progress_bar,
                test_prompt=test_completion_prompt,
                validators=validators,
                use_iterative_complete_v2=use_validators
            )

            cf_description = ''

            # Create the supsicious facts for a suspect
            if max_num_of_suspicious_facts:
                root_node = [
                    LogicNode(f'Some suspicious facts about {suspect_name}.', [
                        LogicNode(f'{x} And this is suspicious.') for x in suspect_info['red_herrings']
                    ], frozen=True, prunable=False)
                ]

                validators = [StructureValidator()]
                if use_validators:
                    validators.append(ForbiddenTextValidator(
                        forbidden_words=[
                            crime_scene, motive, murder_weapon, 'opportunity', 'means', 'motive', 'murder weapon', 'crime scene', 'weapon'
                        ],
                        reason_why="These clues are meant to be suspicious and not prove guilt.  Because of that, we cannot mention anything super related to the murder since then it could be counted as actual evidence.  This fact is meant to serve as a red herring for the reader.  It can be close to the murder or related, but should not be able to be used to prove a means, motive, or opportunity."
                    ))
                    if model_validator_model:
                        validators.append(ModelValidator(
                            model_validator_model,
                            f"We are writing a murder mystery and to do so we are creating a narrative guide of evidence.  We are proving a suspicious fact (a fact that is irrelevant to the mystery) right now.  Does this deduction in any way prove or help to prove a motive (reason to kill), an opportunity (access to the crime scene) or a means (access to the murder weapon) given the description of the mystery below.\n\n{description}",
                            "The fact we are proving is suspicious but should not prove guilt.  Which is why we do not want the fact to prove a means (access to murder weapon), an opportunity (access to the crime scene), nor a motive (reason to kill) as this may complicate the reasoning in our mystery.",
                            early_escape_model=model_validator_early_escape_model
                        ))

                cf_description = f'''{suspect_name} is a {suspect_info["role"]}... and they are super suspicious.'''.strip()

                sus_tree = self.complete_structure(
                    self.build_structure(
                        depth=depth,
                        bf_factor=bf_factor,
                        chance_to_prune_all=chance_to_prune_all,
                        chance_to_prune=chance_to_prune,
                        root_nodes=root_node
                    ),
                    model,
                    description=cf_description,
                    completion_prompt_fn=self.create_completion_prompt(sf_example_trees,
                                                                       sf_example_node_completions,
                                                                       sf_example_descriptions,
                                                                       intro=_mm_suspicious_prompt_intro_,
                                                                       because_clause_after=0),
                    max_retries_on_error=max_retries_on_error,
                    inplace=True,
                    retry_model=retry_model,
                    progress_bar=progress_bar,
                    test_prompt=test_completion_prompt,
                    validators=validators,
                    use_iterative_complete_v2=use_validators
                )
                tree.nodes[0].children.extend(sus_tree.nodes[0].children)

            suspect_trees.append({
                'tree': tree,
                'description': description,
                'cf_description': cf_description,
                'suspect_info': suspect_info,
                'victim_info': victim_info
            })

        return suspect_trees

    def create_chapter_trees(
            self,
            suspect_trees,
            max_num_of_suspicious_facts: int = 1
    ):
        """
        Logic to make trees for innocent/guilty suspects.

        A guilty suspect is someone with a means, motive, and opportunity (MMO).
        An innocent suspect is someone without all three.

        Therefore, for each suspect, we generate two trees with 3 main branches.  An innocent tree with a suspicious
        fact branch and 2 branches from the MMO.  And a guilty tree where the only 3 branches are the MMO branches.

        :param suspect_trees: Trees with the MMO and suspicious fact(s) branches.
        :param max_num_of_suspicious_facts: # of suspicious facts (usually 1)
        """

        for sidx, s in enumerate(suspect_trees):
            template = deepcopy(s['tree'])
            t = deepcopy(s['tree'])

            t.nodes[0].children = random.sample([x for x in t.nodes[0].children if 'suspicious' not in x.value.lower()], 2)

            if max_num_of_suspicious_facts:
                t.nodes[0].children.extend([x for x in template.nodes[0].children if 'suspicious' in x.value.lower()])

            suspect_trees[sidx]['innocent_tree'] = t


            suspect_trees[sidx]['murderer_tree'] = deepcopy(s['tree'])
            suspect_trees[sidx]['murderer_tree'].nodes[0].children = [x for x in template.nodes[0].children if any([y in x.value.lower() for y in ['means', 'motive', 'opportunity']])]

            if max_num_of_suspicious_facts:
                suspect_trees[sidx]['murderer_tree'].nodes[0].children.extend(random.sample([x for x in template.nodes[0].children if 'suspicious' in x.value.lower()], max_num_of_suspicious_facts - 1))

        return suspect_trees

    def create_chapter(
            self,
            model: Model,
            suspect_trees: List[Dict[str, any]],
            facts_only: bool = False,
            validate_model: Model = None

    ) -> List[Dict[str, any]]:
        """
        Often it is too difficult for LLMs to generate stories that include all facts from a list if the list is long.
        Therefore, we generate the story in chapters.

        :param model: The model used to generate the story.
        :param suspect_trees: The suspect trees (a list of dictionaries that contain a murderer_tree and innocent_tree)
        :param validate_model: Model used to validate that the facts are entailed by the story. (highly encouraged but
            will take a long time)
        """

        # Create a chapter per suspect.
        for sidx, s in enumerate(suspect_trees):
            description = s['description']

            desc = []
            for line in description.split('\n'):
                if 'motive' in line.lower():
                    continue
                desc.append(line)

            innocent_description = '\n'.join(desc)

            # Start by creating a chapter for when the suspect is the murderer.
            assert len(s['murderer_tree'].nodes[0].children) == 3, 'Bad tree format'
            prompt = create_story_prompt__facts_only(description, s['murderer_tree'])

            output, _ = self.inference(prompt, model)

            unsupported = -1

            # Validate that this chapters facts are all entailed. (or at least try to entail them)
            if validate_model is not None:
                for _ in range(3):
                    new_output, new_unsupported = self.fact_recall_story_validation(output, s['murderer_tree'], validate_model)

                    if output == new_output:
                        break
                    elif unsupported == -1 or unsupported >= new_unsupported:
                        output = new_output
                        unsupported = new_unsupported
                    else:
                        break

            suspect_trees[sidx]['murderer_chapter'] = output

            # Repeat but create an innocent chapter.
            assert len(s['innocent_tree'].nodes[0].children) == 3, 'Bad tree format'
            prompt = create_story_prompt__facts_only(innocent_description, s['innocent_tree'])

            output, _ = self.inference(prompt, model)
            unsupported = -1

            if validate_model is not None:
                for _ in range(3):
                    new_output, new_unsupported = self.fact_recall_story_validation(output, s['innocent_tree'], validate_model)

                    if output == new_output:
                        break
                    elif unsupported == -1 or unsupported >= new_unsupported:
                        output = new_output
                        unsupported = new_unsupported
                    else:
                        break

            suspect_trees[sidx]['innocent_chapter'] = output

        return suspect_trees

    def fact_recall_story_validation(
            self,
            ctx: str,
            tree: LogicTree,
            model: Model
    ):
        """
        Given some story/chapter/context ensure that all the facts from the tree are entailed by the context.

        Entailment is checked via an LLM and prompt.

        :param ctx: Context that should entail all the facts from the tre.
        :param tree: Tree used to create the context.
        :param model: Model to run the entailment check on.
        """

        facts = list(sorted(list(set([x.value for x in tree.get_facts()]))))
        facts_str = "\n".join([f'{fidx+1} - {x}' for fidx, x in enumerate(facts)])

        unsupported = []
        curr_cost = model.total_cost
        pbar = tqdm(facts, desc=f'Validating each fact is supported in the story | cost = {0:.2f}', total=len(facts), disable=True)
        prompt = f'''
Here is a story.

{ctx}

Here are the facts:

{facts_str}

Are the facts supported by the given story?  Answer in this format: "Fact Answer - (Fact idx): (your step-by-step reasoning), ANSWER: Yes" or "ANSWER: No"
        '''.strip()
        output, _ = self.inference(prompt, model, temperature=0.0)

        pbar.set_description(f'Validating each fact is supported in the story | cost = {model.total_cost - curr_cost:.2f}')

        lines = [x for x in output.split('Fact Answer - ') if x != '']
        unsupported_reasons = []

        for f, l in zip(facts, lines):

            if 'ANSWER: Yes'.lower() in l.lower():
                continue
            else:
                unsupported.append(f)
                unsupported_reasons.append(l)

        if len(unsupported) == 0:
            return ctx, len(unsupported)

        facts_str = "\n".join([f'- {x}' for x in facts])
        unsupported_str = "\n".join([f'- {x} This was unsupported because: {r}' for x, r in zip(unsupported, unsupported_reasons)])

        new_story_prompt = f'''
We are editing a story we just wrote.  The story is required to have supporting evidence for a list of facts, but not all facts were supported. 

Your job is to edit the story such that all the facts are supported.

---
Story:
{ctx}

---


Original List Of Facts:
{facts_str}

Unsupported Facts (fix the story so that these facts are now supported):
{unsupported_str}

Remember, all facts should be supported or entailed by the story.  Do not change the story in a way that one of the other facts will be unsupported.  Only add evidence and add information to the story such that the unsupported facts are supported.  

For example, if the story was "Bob is a cat.  I like cool things" and the missing fact was "Cats are objectively cool", you would rewrite the story as "Cats, they are unbelievably cool, and I like cool things. Maybe that's why I like Bob, the cat." 

Output:
        '''

        new_ctx, _ = self.inference(new_story_prompt, model, temperature=0.0)
        return new_ctx, len(unsupported)

import pprint
from pathlib import Path
import json
import random
from typing import List, Dict, Union, Type
import copy


class Madlib:
    """
    A madlib is really just a sampler.  See the __main__ block below for examples on calling.
    """

    def __init__(
            self,
            items: Dict[str, Union[List[str], Path]],
    ):
        self.items = {}
        for k, v in items.items():
            if isinstance(v, Path):
                v = json.load(v.open('r'))
            self.items[k] = v

    def sample(
            self,
            item: str,
            disallow_value_list: List[str] = (),
            num_samples: int = 1
    ) -> List[str]:
        uniq_set = lambda item, vals: [x for x in self.items[item] if x not in vals]
        return random.sample(uniq_set(item, disallow_value_list), num_samples)


if __name__ == "__main__":
    # Example of how to use a "Madlib" - often times the datasets themselves will have more obscure/advanced uses of
    # them wrapped in a function.

    from src.utils.paths import ROOT_FOLDER

    madlib = Madlib(
        {
            "names": ROOT_FOLDER / 'domain_seed/names.json',
            "motives": ROOT_FOLDER / 'domain_seed/motives.json',
            "murder_weapons": ROOT_FOLDER / 'domain_seed/murder_weapons.json',
            "relationships": ROOT_FOLDER / 'domain_seed/relationships.json',
            "crime_scenes": ROOT_FOLDER / 'domain_seed/crime_scenes.json'
        }
    )

    names = madlib.sample('names', num_samples=11)
    victim = names[0]
    suspects = names[1:]

    crime_scene = madlib.sample('crime_scenes', num_samples=1)
    murder_weapon = madlib.sample('murder_weapons', num_samples=1)

    motives = madlib.sample('motives', num_samples=10)
    relationships = madlib.sample('relationships', num_samples=10)

    pprint.pprint({
        'victim': victim,
        'crime_scene': crime_scene,
        'murder_weapons': murder_weapon,
        'suspects': {
            n: {
                'motive': m,
                'relationship': r
            }
            for n, m, r in zip(suspects, motives, relationships)
        },
    })


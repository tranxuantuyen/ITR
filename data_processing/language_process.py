#%%
import benepar, spacy
import os
import json
import re
from nltk import Tree
from nltk.grammar import is_nonterminal, DependencyProduction
from collections import defaultdict
import copy
from tqdm import tqdm
nlp = spacy.load('en_core_web_md')
nlp.add_pipe("benepar", config={"model": "benepar_en3"})
import tree_process as tp
file_path = "/home/s222126678/Documents/meccano/project/MeViS/datasets/mevis/train"
meta_json = os.path.join(file_path, 'meta_expressions.json')
with open(meta_json) as fp:
    meta_data = json.load(fp)


for video in tqdm(meta_data["videos"], disable=False):
    for exp in meta_data["videos"][video]["expressions"]:
        sentence = meta_data["videos"][video]["expressions"][exp]["exp"]
        try:
            tree_dict = tp.get_tree_dict(sentence)
        except:
            print(sentence)
            assert False
        meta_data["videos"][video]["expressions"][exp]["tree"] = tree_dict
        assert len(tree_dict) > 0

with open("tree_meta_valid_u.json", "w") as fp:
    json.dump(meta_data, fp, indent=4)

#%%


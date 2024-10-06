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
file_path = "/home/s222126678/Documents/meccano/project/MeViS/datasets/mevis/valid"
meta_json = os.path.join(file_path, 'meta_expressions.json')
with open(meta_json) as fp:
    meta_data = json.load(fp)


for video in tqdm(meta_data["videos"]):
    for exp in meta_data["videos"][video]["expressions"]:
        sentence = meta_data["videos"][video]["expressions"][exp]["exp"]
        try:
            tree_dict = tp.get_tree_dict(sentence)
            # tree = tp.TreeFull(tree_dict)
            # calculation_procedure = tree.calculattion_procedure()
        except:
            tree_dict = {}
        meta_data["videos"][video]["expressions"][exp]["tree"] = tree_dict

with open("tree_meta_valid_u.json", "w") as fp:
    json.dump(meta_data, fp, indent=4)

#%%


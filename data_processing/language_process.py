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

file_path = "/home/s222126678/Documents/meccano/project/MeViS/datasets/mevis/train"
meta_json = os.path.join(file_path, 'meta_expressions.json')
with open(meta_json) as fp:
    meta_data = json.load(fp)

def is_VP(pattern, text):
    if text.startswith("(VP") == False:
        return False
    occurrences = re.findall(pattern, text)
    return len(occurrences) == 1

def tree_to_dict(tree):
    if isinstance(tree, str):
        return tree
    children = [tree_to_dict(subtree) for subtree in tree]
    
    return {tree.label(): children}

def dict_to_tree(d):
    for label, children in d.items():
        if all(isinstance(child, str) for child in children):
            return Tree(label, children)
        else:
            return Tree(label, [dict_to_tree(child) for child in children])
def remove_until_np_vp(tree):
    for subtree in tree:
        if isinstance(subtree, Tree):
            # If we reach NP or VP, retain the subtree and stop removing further
            if subtree.label() in ['NP', 'VP']:
                return subtree
            else:
                result = remove_until_np_vp(subtree)
                if result is not None:
                    return result
def remove_sbar(tree, current_depth=0):
    # tree = copy.deepcopy(tree_original)
    for i, subtree in enumerate(tree):
        if isinstance(subtree, Tree):
            if subtree.label() == 'SBAR':
                # Look for first VP or NP within SBAR
                tree[i] = remove_until_np_vp(subtree)
                remove_sbar(subtree)
            else:
                # Recursively process child nodes
                remove_sbar(subtree)
    return tree

def extract_np_vp_subtrees(tree, current_depth=0):
    subtrees_with_depth = []
    
    if tree.label() in ('NP', 'VP'):
        subtrees_with_depth.append((tree, current_depth))
    
    for subtree in tree:
        if isinstance(subtree, Tree):  
            subtrees_with_depth.extend(extract_np_vp_subtrees(subtree, current_depth + 1))
    return subtrees_with_depth

for video in tqdm(meta_data["videos"]):
    for exp in meta_data["videos"][video]["expressions"]:
        sentence = meta_data["videos"][video]["expressions"][exp]["exp"]
        if sentence[-1] == ".":
            sentence = sentence[:-1]
        doc = nlp(sentence)
        sent = list(doc.sents)[0]
        input_str = sent._.parse_string
        const_tree = Tree.fromstring(input_str)
        const_tree_reduce = copy.deepcopy(const_tree)
        const_tree_reduce = remove_sbar(const_tree_reduce)
        subtrees_with_distances = extract_np_vp_subtrees(const_tree_reduce)

        tree_dict = {}
        for subtree, depth in subtrees_with_distances:
            if depth not in tree_dict:
                tree_dict[depth] = {}
            if "NP" not in tree_dict[depth]:
                tree_dict[depth]["NP"] = []
            if "VP" not in tree_dict[depth]:
                tree_dict[depth]["VP"] = []
            try:
                tree_dict[depth][subtree.label()].append(" ".join(subtree.flatten())) 
            except:
                pass
        meta_data["videos"][video]["expressions"][exp]["tree"] = tree_dict

with open("tree_meta_train.json", "w") as fp:
    json.dump(meta_data, fp, indent=4)

        # for depth in tree_dict:
        #     if depth == 0:
        #         continue
        #     NP = tree_dict[depth]["NP"] if tree_dict[depth]["NP"] else ["EMPTY"]
        #     VP = tree_dict[depth]["VP"] if tree_dict[depth]["VP"] else ["EMPTY"]
        #     print(f"Depth: {depth}")
        #     print(f"NP: {NP}")
        #     print(f"VP: {VP}")
        #     print("\n")  


#%%
import benepar, spacy
import os
import json
import re
from nltk import Tree
from nltk.grammar import is_nonterminal, DependencyProduction
from collections import defaultdict
import copy
nlp = spacy.load('en_core_web_md')
nlp.add_pipe("benepar", config={"model": "benepar_en3"})
def is_VP(pattern, text):
    if text.startswith("(VP") == False:
        return False
    occurrences = re.findall(pattern, text)
    return len(occurrences) == 1
# sentence = meta_data["videos"][video]["expressions"][exp]["exp"]
# sentence = "The parrot shifting head in front of us"
# sentence = "Feathered creature staying back-side without shifting"
# sentence = "The elephant facing us and then turning around to walk away"
# sentence = "a tall man listens from an old man and move around the crowd"
# sentence = "The initial lizard to consume the food."
# sentence = "The lizard that was the earliest to start eating"
# sentence = "The two monkeys in a crouched position on the left without any movement"
# sentence = "The two monkeys sitting down on the left and remaining still."
# sentence = "The stationary monkey sitting on the ground."
# sentence = "The monkey that hasn't shifted its position while sitting on the ground"
# sentence = "The second car in the row driving straight through the intersection"
sentence = "Bear chasing another bear around by walking in circle"
# sentence = "The second vehicle going straight at the crossroads"
# sentence = "A panda that successfully climbed to a higer place in a tree" 

if sentence[-1] == ".":
    sentence = sentence[:-1]
doc = nlp(sentence)
sent = list(doc.sents)[0]
input_str = sent._.parse_string
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

const_tree = Tree.fromstring(input_str)
def extract_np_vp_subtrees(tree, current_depth=0):
    subtrees_with_depth = []
    
    if tree.label() in ('NP', 'VP'):
        subtrees_with_depth.append((tree, current_depth))
    
    for subtree in tree:
        if isinstance(subtree, Tree):  # Check if it's a subtree
            subtrees_with_depth.extend(extract_np_vp_subtrees(subtree, current_depth + 1))
    
    return subtrees_with_depth

subtrees_with_distances = extract_np_vp_subtrees(const_tree)
def is_substring_in_list(substring, string_list):
    for string in string_list:
        if substring in string:
            return True
    return False

tree_dict = {}
for subtree, depth in subtrees_with_distances:
    if depth not in tree_dict:
        tree_dict[depth] = {}
    if "NP" not in tree_dict[depth]:
        tree_dict[depth]["NP"] = []
    if "VP" not in tree_dict[depth]:
        tree_dict[depth]["VP"] = []
    tree_dict[depth][subtree.label()].append(" ".join(subtree.flatten())) 

for depth in tree_dict:
    if depth == 0:
        continue
    NP = tree_dict[depth]["NP"] if tree_dict[depth]["NP"] else ["EMPTY"]
    VP = tree_dict[depth]["VP"] if tree_dict[depth]["VP"] else ["EMPTY"]
    print(f"Depth: {depth}")
    print(f"NP: {NP}")
    print(f"VP: {VP}")
    print("\n")  
    # print(f"Subtree: {subtree}, Depth: {depth}")

#%%
# tree_dict_vis = copy.deepcopy(tree_dict)
# target = {"S": []}
# previous_key = None
# for key, sub_dict in tree_dict_vis.items():
#     if not previous_key:
#         np_value = sub_dict.get("NP", ["EMPTY"])[0] 
#         vp_value = sub_dict.get("VP", ["EMPTY"])[0]  

#         target["S"].append({
#             f"NP - {np_value}": []
#         })

#         target["S"].append({
#             f"VP - {vp_value}": []
#         })
#         previous_key = key
#         previous_np = np_value
#         previous_vp = vp_value
#     else:
#         np_value = sub_dict.get("NP", ["EMPTY"])[0] 
#         vp_value = sub_dict.get("VP", ["EMPTY"])[0]  
        
    

# Print the target dictionary
# print(target)
# dict_to_tree(target)
# %%
# target = {"S": [{"NP- The second vehicle": [{"NP": ["EMPTY"]}, {"VP": ["EMPTY"]}]}, {"VP - going straight at the crossroads" : [{"NP": ["the crossroads"]}, {"VP": ["EMPTY"]}]}]}
# m = {"S": target}
# %%

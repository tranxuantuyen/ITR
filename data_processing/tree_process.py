import benepar, spacy
from nltk import Tree
from collections import defaultdict
import copy
nlp = spacy.load('en_core_web_md')
nlp.add_pipe("benepar", config={"model": "benepar_en3"})
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



def extract_np_vp_subtrees(tree, bank, current_depth=0, location=0):
    subtrees_with_depth = []
    
    if tree.label() in ('NP', 'VP'):
        text = ' '.join(tree.flatten())
        node_id = '_'.join([str(current_depth), str(location), str(bank.pop(0))])
        subtrees_with_depth.append((text, tree.label(),  node_id))
    
    for i_th, subtree in enumerate(tree):
        if isinstance(subtree, Tree):  
            subtrees_with_depth.extend(extract_np_vp_subtrees(subtree,bank, current_depth + 1, i_th))
    return subtrees_with_depth
def get_tree_dict(sentence):
    bank = list(range(100))
    if sentence[-1] == ".":
        sentence = sentence[:-1]
    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    input_str = sent._.parse_string
    const_tree = Tree.fromstring(input_str)
    const_tree_reduce = copy.deepcopy(const_tree)
    const_tree_reduce = remove_sbar(const_tree_reduce)
    subtrees_with_distances = extract_np_vp_subtrees(const_tree_reduce, bank)
    all_depth = {int(node_id.split('_')[0]) for _,_, node_id in subtrees_with_distances}
    tree_dict = {}
    for subtree, label, node_id in subtrees_with_distances:
        assert node_id not in tree_dict
        tree_dict[node_id] = {}
        tree_dict[node_id]['child'] = []
        tree_dict[node_id]['text']  = subtree
        tree_dict[node_id]['label']  = label
        depth, location, _ = [int(i) for i in node_id.split('_')]
        if depth == 0:
            tree_dict[node_id]['parent'] = "root"
        elif depth == 1:
            tree_dict[node_id]['parent'] = '0_0_0'
        else:
            candidate_parrent = [i[2] for i in subtrees_with_distances if subtree in i[0] and i[2] != node_id]
            tree_dict[node_id]['parent'] = max(candidate_parrent, key=lambda x: int(x.split('_')[0]))
    if '0_0_0' not in tree_dict.keys():
        tree_dict['0_0_0'] = {}
        tree_dict['0_0_0']['child'] = []
        tree_dict['0_0_0']['text'] = ' '.join(const_tree_reduce.flatten())
        tree_dict['0_0_0']['label'] = 'S'
        tree_dict['0_0_0']['parent'] = 'root'
    for key, value in tree_dict.items():
        parent = value['parent']
        if parent in tree_dict:
            tree_dict[parent]['child'].append(key)
    return tree_dict

class Node:
    def __init__(self, node_instance):
        key, value = node_instance
        self.id = key
        self.text = value['text']
        self.label = value['label']
        self.parent = value['parent']
        self.child = value['child']
    def __repr__(self):
        return f"ID: {self.id}, Text: {self.text}, Label: {self.label}, Parent: {self.parent}, Child: {self.child}"
    def get_depth(self):
        return int(self.id.split('_')[0])
    
class TreeFull:
    def __init__(self, tree_dict):
        self.tree_dict = tree_dict
        self.nodes = [Node((key, value)) for key, value in tree_dict.items()]
        # self.root = self.nodes['0_0_0']
    def get_leave_nodes(self):
        return [node for node in self.nodes if len(node.child) == 0]
    def get_non_leave_nodes(self):
        return [node for node in self.nodes if len(node.child) > 0]
    def calculattion_procedure(self):
        non_leave = self.get_non_leave_nodes()
        sorted_non_leave = sorted(non_leave, key=lambda x: x.get_depth(), reverse=True)
        leave = self.get_leave_nodes()
        procedure = leave + sorted_non_leave
        return procedure
    
if __name__ == "__main__":
    sentence = "The quick brown fox jumps over the lazy dog."
    tree_dict = get_tree_dict(sentence)
    tree = TreeFull(tree_dict)
    print(tree.calculattion_procedure())
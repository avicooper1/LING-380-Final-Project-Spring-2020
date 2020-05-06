from stanfordcorenlp import StanfordCoreNLP
import logging
import tools
from nltk import Tree
from functools import reduce
import json
import tqdm

def gen_tikz_qtree(parse):
    parse = parse.replace('(', '[.')
    parse = parse.replace(')', ' ]')
    parse = parse.replace('\n', '')
    return parse

def binarize(tree):
    """
    Recursively turn a tree into a binary tree.
    """
    if isinstance(tree, str):
        return tree
    elif len(tree) == 1:
        return binarize(tree[0])
    else:
        return reduce(lambda x, y: (binarize(x), binarize(y)), tree)

def gen_binary_parse(parse, tree):
    t = Tree.fromstring(parse)
    bt = binarize(t)
    btu = str(bt).replace("'", "").replace(',', '').replace('(', '( ').replace(')', ' )')
    return tree.preprocessing(tree.tokenize(btu))

if __name__ == '__main__':
    sNLP = StanfordCoreNLP('stanford-corenlp-full-2020-04-20', port=9000, memory='8g', logging_level=logging.DEBUG)
    result, sent_bad, sent_good, pre_bad, pre_good = tools.get_blimp_data('principle_A_domain_2')
    from torchtext import datasets
    tree = datasets.nli.ShiftReduceField()
    good_parses = {}
    bad_parses = {}
    for sent in tqdm.tqdm(range(len(sent_bad))):
        bad_parses[sent] = gen_binary_parse(sNLP.parse(sent_bad[sent]), tree)
        good_parses[sent] = gen_binary_parse(sNLP.parse(sent_good[sent]), tree)
    with open('domain2_bad_sent_parses.json', 'w') as fp:
        json.dump(bad_parses, fp)
    with open('domain2_good_sent_parses.json', 'w') as fp:
        json.dump(good_parses, fp)
    sNLP.close()



    # tex_code = ''
    # with open('tex_files/tree_vis.tex', 'r') as f:
    #     tex_code = f.read()
    # with open('tex_files/parsed_tree_vis.tex', 'w') as f:
    #     f.write(tex_code.replace('TREE', parse_tree))
    # sNLP.close()
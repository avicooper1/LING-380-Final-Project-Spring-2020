from stanfordcorenlp import StanfordCoreNLP
import logging

def gen_tikz_qtree(parse):
    parse = parse.replace('(', '[.')
    parse = parse.replace(')', ' ]')
    parse = parse.replace('\n', '')
    return parse

if __name__ == '__main__':
    sNLP = StanfordCoreNLP('stanford-corenlp-full-2020-04-20', port=9000, memory='8g', logging_level=logging.DEBUG)
    text = 'I went to the nearby makolet and bought some yellow potatoes.'
    parse_tree = gen_tikz_qtree(sNLP.parse(text))
    tex_code = ''
    with open('tex_files/tree_vis.tex', 'r') as f:
        tex_code = f.read()
    with open('tex_files/parsed_tree_vis.tex', 'w') as f:
        f.write(tex_code.replace('TREE', parse_tree))
    sNLP.close()
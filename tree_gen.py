from stanfordcorenlp import StanfordCoreNLP
import logging
import tools

def gen_tikz_qtree(parse):
    parse = parse.replace('(', '[.')
    parse = parse.replace(')', ' ]')
    parse = parse.replace('\n', '')
    return parse

if __name__ == '__main__':
    sNLP = StanfordCoreNLP('stanford-corenlp-full-2020-04-20', port=9000, memory='8g', logging_level=logging.DEBUG)
    text = 'I went to the nearby makolet and bought some yellow potatoes.'
    batches = tools.load_wikitext_103(128)
    for batch in batches:
        parse_trees = sNLP.parse(text)
        with open('test.txt', 'a') as f:
            f.write(parse_trees)
        break



    # tex_code = ''
    # with open('tex_files/tree_vis.tex', 'r') as f:
    #     tex_code = f.read()
    # with open('tex_files/parsed_tree_vis.tex', 'w') as f:
    #     f.write(tex_code.replace('TREE', parse_tree))
    # sNLP.close()
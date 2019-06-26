import argparse
import sys
from nltk.corpus import wordnet as wn
from stanfordcorenlp import StanfordCoreNLP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing script to extract example sentences')
    parser.add_argument('--mode', type=str, default='wordnet',
                        help='source of extracting example sentences (wordnet)')
    parser.add_argument('--data', type=str, default='/Users/ishiwatari/Dropbox/PhD/src/PycharmProjects/defgen_pt/data/onto_nora/mini/w2v/train.txt', help='original data')
    parser.add_argument('--out', type=str, default='/Users/ishiwatari/Dropbox/PhD/src/PycharmProjects/defgen_pt/data/onto_nora/mini/eg', help='directory to output files (*.txt, *.eg)')
    parser.add_argument('--corenlp_path', type=str, help="path to stanford-corenlp-full-2018-02-27")
    parser.add_argument('--replace_target_word', action='store_true',
                        help='replace target words with special tokens <TRG>')
    args = parser.parse_args()

def choose_example(examples, word):
    if len(examples) == 0:
        return None

    for eg in examples:
        if word in set(eg.split()): # choose the first example that includes the word
            if args.replace_target_word:
                return eg

    return None

def tokenize(example):
    return ' '.join(nlp.word_tokenize(example))


nlp = StanfordCoreNLP(args.corenlp_path)
out_txt = args.out + '/' + args.data.rsplit('/', 1)[1]
out_eg = args.out + '/' + args.data.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.eg'

with open(out_txt, 'w') as fp_txt:
    with open(out_eg, 'w') as fp_eg:
        for line in open(args.data):
            elems = line.strip().split()
            if elems[2] == 'wordnet':
                word = elems[0].split('%', 1)[0]
                syn = elems[0].split('.', 1)[1]
                examples = wn.synset(syn).examples()
                eg = choose_example(examples, word)
                if eg:
                    eg = tokenize(eg)
                    if args.replace_target_word:
                        eg = eg.replace(word, '<TRG>')
                    fp_txt.write(line)
                    fp_eg.write(elems[0] + '\t' + eg + '\n')

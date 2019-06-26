import argparse
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import translate
import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/noraset/',
                    help='location of the data corpus')
parser.add_argument('--vec', type=str, default='/data/ishiwatari/defgen/vec/onto_nora/GoogleNews-vectors-negative300.txt.filtered',
                    help='location of the word2vec data')
parser.add_argument('--hyp', type=str, default='/data/ishiwatari/defgen/model/sgch_feed_adam/test.out',
                    help='location of the data corpus')
parser.add_argument('--sentence_bleu', type=str, default='./sentence-bleu',
                    help='Compiled binary file of sentece-bleu.cpp')
parser.add_argument('--use_nltk', action='store_true',
                    help='use nlkt instead of to compute sentence BLEU')
parser.add_argument('--mode', type=str,
                    help='task of dataset. Choose one from {edit|eg}.')

args = parser.parse_args()

if args.mode == 'edit':
    corpus = data.EditCorpus(args.data, args.vec, max_vocab_size=23450, mode= 'test', topk=2, ignore_sense_id=True)
elif args.mode == 'eg':
    corpus = data.ExampleCorpus(args)

hyp = []
for line in open(args.hyp, 'r'):
    elems = line.strip().split('\t', 1)
    if len(elems) == 1:
        hyp.append((elems[0], ['']))
    else:
        hyp.append((elems[0], elems[1].split()))

translator = translate.Translator(corpus, sentence_bleu=args.sentence_bleu)
print('Averaged sentence BLEU:\n' + str(translator.bleu(hyp, mode="test", nltk=args.use_nltk)))
import argparse
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import translate
import main
import data

parser = argparse.ArgumentParser(description='Test code for Definition Generation task')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/noraset',
                    help='location of the data corpus')
parser.add_argument('--vec', type=str, default='./data/GoogleNews-vectors-negative300.txt',
                    help='location of the word2vec data')
parser.add_argument('--model', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--words', type=int, default='60',
                    help='number of words to generate')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--p', type=float, default=5.0,
                    help='parameter for batch-ensembling to control peakiniess of the similarity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--sentence_bleu', type=str, default='./sentence-bleu',
                    help='Compiled binary file of sentece-bleu.cpp')
parser.add_argument('--use_nltk', action='store_true',
                    help='use nltk instead of sentence-bleu.cpp to compute bleu')
parser.add_argument('--ignore_sense_id', action='store_true',
                    help='ignore sense ids in {train | valid | test}.txt')
parser.add_argument('--topk', type=int, default=1,
                    help='number of nearest neighbors to use in EDIT model')
parser.add_argument('--gen', action='store_true',
                    help='generate definitions')
parser.add_argument('--att_vis', action='store_true',
                    help='output attention weights')
parser.add_argument('--show_logloss', action='store_true',
                    help='print logloss')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        sys.stderr.write("WARNING: You have a CUDA device, so you should probably run with --cuda\n")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.model, 'rb') as f:
    if args.cuda:
        model = torch.load(f)
        model.cuda()
    else:
        model = torch.load(f, map_location=lambda storage, loc: storage)
        model.cpu()

if model.name == "EDIT":
    corpus = data.EditCorpus(args.data, args.vec, max_vocab_size=model.decoder.out_features-3, mode='test', topk=args.topk, ignore_sense_id=args.ignore_sense_id)
elif model.name[:10] == "EDIT_MULTI" or model.name in {"EDIT_DECODER", "EDIT_TRIPLE", "EDIT_HIERARCHICAL",
                                                       "EDIT_HIERARCHICAL_DIFF", "EDIT_HIRO", "EDIT_COPY"}:
    corpus = data.EditCorpus2(args.data, args.vec, max_vocab_size=model.readout.out_features-3, mode='test', topk=model.topk, ignore_sense_id=args.ignore_sense_id)
elif model.name in {'EDIT_COPYNET', "EDIT_COPYNET_SIMPLE"}:
    corpus = data.EditCorpus3(args.data, args.vec, max_vocab_size=model.readout.out_features-3, mode='test', topk=model.topk,
                              ignore_sense_id=args.ignore_sense_id)
elif model.name in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
    corpus = data.EditCorpus4(args.data, args.vec, max_vocab_size=model.readout.out_features-3, mode='test', topk=model.topk, ignore_sense_id=args.ignore_sense_id)
else:
    corpus = data.NorasetCorpus(args.data, args.vec, max_vocab_size=model.readout.out_features-3, mode='test', ignore_sense_id=args.ignore_sense_id)
translator = translate.Translator(corpus, sentence_bleu=args.sentence_bleu)

if args.gen:
    res = translator.greedy(model, mode="test", max_batch_size=args.batch_size, cuda=args.cuda, p=args.p, max_len=args.words, ignore_duplicates=True, return_att_weights=args.att_vis)
    if args.att_vis:
        results, att_weights_batches = res
        translator.visualize_att_weights(att_weights_batches, "test", model.topk, results, args.model.rsplit('/', 1)[0] + '/att_vis')
    else:
        results = res

    for (word, desc) in results:
        print(word, end='\t')
        new_def = []
        for w in desc:
            if w not in corpus.id2word:
                new_def.append('[' + w + ']')
            else:
                new_def.append(w)
        print(' '.join(new_def), flush=True)

    if not args.use_nltk:
        bleu_cpp = translator.bleu(results, mode="test", nltk=None)
        sys.stderr.write('BLEU (sentence-bleu.cpp): ' + str(bleu_cpp) + '\n')
    bleu_nltk_sent = translator.bleu(results, mode="test", nltk='sentence')
    sys.stderr.write('BLEU (nltk sentence): ' + str(bleu_nltk_sent) + '\n')
    bleu_nltk_corp = translator.bleu(results, mode="test", nltk='corpus')
    sys.stderr.write('BLEU (nltk corpus): ' + str(bleu_nltk_corp) + '\n')

if args.show_logloss:
    # TODO: Fix this ugly way of getting raw references
    if model.name in {'S', 'SG', 'SGCH'}:
        corpus2 = data.EditCorpus3(args.data, args.vec, max_vocab_size=model.readout.out_features - 3, mode='test', topk=1, ignore_sense_id=args.ignore_sense_id)
        translator.corpus.valid = corpus2.valid
        translator.corpus.test = corpus2.test
    returns = translator.eval_log_loss(model, mode='test', max_batch_size=args.batch_size, cuda=args.cuda, p=args.p, return_logloss_matrix=True)
    if model.name in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
        loss, loss_cpy, log_loss_matrix =returns
    else:
        loss, log_loss_matrix = returns
    translator.print_log_loss_matrix(log_loss_matrix, 'test')

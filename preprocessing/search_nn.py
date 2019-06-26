import argparse
import math
import sys
import os
import torch
from torch.autograd import Variable
from nltk.translate import bleu_score

parser = argparse.ArgumentParser(description='Find nearest neighbors in vector space')
parser.add_argument('--data', type=str, default='../data/noraset',
                    help='location of the data corpus')
parser.add_argument('--vec', type=str, default='../data/GoogleNews-vectors-negative300.txt',
                    help='location of the word2vec data')
parser.add_argument('--K', type=int, default=30,
                    help='number of nearest neighbors to find')
parser.add_argument('--L', type=int, default=-1,
                    help='The number of candidates to use for strong_cheat method')
parser.add_argument('--ignore_sense_id', action='store_true',
                    help='ignore senseIds in {train | valid | test}.txt')
parser.add_argument('--filtering', action='store_true',
                    help='output filtered word2vec file. filename: <vec>.filtered')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mode', type=str, default='min_risk',
                    help='policy to retrieve definitions from nearest neighbors. NOTE! DO NOT use "cheat" modes other than TESTING!!! (min_risk | max_vocab | cheat | strong_cheat)')
args = parser.parse_args()

def is_usable(trg_word, nn_word):
    if trg_word.split('%', 1)[0] == nn_word.split('%', 1)[0]:
        return False
    else: return True

def choose_min_risk_def(defs):
    if len(defs) == 1:
        return defs[0]
    else:
        bleu_def = []
        for i, hyp in enumerate(defs):
            refs = []
            for j, def_ref in enumerate(defs):
                if i != j:
                    refs.append(def_ref)
            bleu = bleu_score.sentence_bleu([ref.split() for ref in refs], hyp.split(),
                                            smoothing_function=bleu_score.SmoothingFunction().method2, auto_reweigh=True)
            bleu_def.append((bleu, hyp))
        return sorted(bleu_def, reverse=True)[0][1]

def choose_cheated_def(vocab_trg, covered_vocab, defs_nn):
    # choose a definitions that
    # 1. decrease not covered vocab
    # 2. increase covered_vocab

    criterions = [] # (criterion#1, criterion#2, def, def_split)
    not_covered_vocab = vocab_trg - covered_vocab
    for definition in defs_nn:
        def_split = definition.split()
        vocab = set(def_split)
        criterion1 = len(vocab & not_covered_vocab)
        criterion2 = len(vocab - covered_vocab)
        criterions.append((criterion1, criterion2, definition, def_split))
    sorted_criterions = sorted(criterions, reverse=True)
    return sorted_criterions[0][2], covered_vocab | set(sorted_criterions[0][3])

def choose_max_vocab_def(covered_vocab, defs_nn):
    # choose a definition that increase covered vocab
    criterions = []  # (criterion, def, def_split)
    for definition in defs_nn:
        def_split = definition.split()
        vocab = set(def_split)
        criterion = len(vocab - covered_vocab)
        criterions.append((criterion, definition, def_split))
    sorted_criterions = sorted(criterions, reverse=True)
    return sorted_criterions[0][1], covered_vocab | set(sorted_criterions[0][2])

cos_sim = torch.nn.CosineSimilarity(dim=0)
def choose_nn_from_def(trg_word, definition):
    trg_vec = torch.FloatTensor(word2vec[trg_word])

    criterion = [] # (vec_sim, nn_word, nn_vec)
    for nn_word in def2nns_train[definition]:
        if trg_word != nn_word and len(word2vec[nn_word]) > 0:
            nn_vec = torch.FloatTensor(word2vec[nn_word])
            vec_sim = cos_sim(trg_vec, nn_vec)[0]
            criterion.append((vec_sim, nn_word, nn_vec))

    # choose the word that has the highest sim cos_sim trg_vec
    if len(criterion) > 0:
        vec_sim, nn_word, nn_vec = max(criterion)
        vec_diff = trg_vec - nn_vec
        return nn_word, vec_sim, vec_diff

    else: # if we cannot find any nn_word that have word vector
        return None, None, None

def find_nearest_defs(trg_word, def2nns_train):
    trg_defs = set(word2defs[trg_word])
    defs_split = [def2split[definition] for definition in word2defs[trg_word]]
    trg_vocab = set()
    for definition in defs_split:
        trg_vocab |= set(definition)

    # make a definition list and compute the unigram coverages
    criterions = []  # (vocab_coverage_gain, vocab_coverage, def, covered_vocab)
    for definition in def2nns_train:
        # if the definition is the only one for the target word, do not use it.
        if definition in trg_defs and len(def2nns_train[definition]) == 1 and trg_word in def2nns_train[definition]:
            pass
        else:
            covered_vocab = trg_vocab & set(def2split[definition])
            criterions.append([len(covered_vocab), len(covered_vocab), definition, covered_vocab])
    criterions = sorted(criterions, reverse=True)
    if args.L > 0:
        criterions = criterions[:args.L] # approximated in order to make it faster

    # choose the definitions in a greedy way
    covered_vocab_total = set()
    # defs_chosen = []
    nn_def_sim_diff = []
    while len(nn_def_sim_diff) < args.K:
        best = max(criterions)
        covered_vocab_total |= best[3]
        # defs_chosen.append(best[2])
        definition = best[2]
        criterions.remove(best)

        # randomly choose nn_word given the definition
        nn_word, vec_sim, vec_diff = choose_nn_from_def(trg_word, definition)
        if nn_word != None: # all the nn_word do not have their word vectors (happens when we use ft instead of w2v)
            nn_def_sim_diff.append((nn_word, definition, vec_sim, vec_diff))


        if best[0] > 0:
            for i, (_, vocab_coverage, definition, covered_vocab) in enumerate(criterions):
                criterions[i][0] = len(covered_vocab ^ covered_vocab_total) # update vocab_covarage_gain

    return nn_def_sim_diff

sys.stderr.write('Reading corpora...   ')
sys.stderr.flush()
word2vec = {'<unk>':''}
word2defs = {}
words_test, words_valid, words_train = set(), set(), set()
if args.mode == 'strong_cheat':
    def2nns_train = {}
    def2split = {}
for file in ['test.txt', 'valid.txt', 'train.txt']:
    for line in open(os.path.join(args.data, file), 'r'):
        word, pos, source, definition, syns, hyps = line.strip().split('\t')
        if args.ignore_sense_id:
            word = word.split('%', 1)[0]
        word2vec[word] = ''
        if word not in word2defs:
            word2defs[word] = []
        word2defs[word].append(definition)
        if args.mode == 'strong_cheat':
            def2split[definition] = definition.split()
            if file == 'train.txt':
                if definition not in def2nns_train:
                    def2nns_train[definition] = set()
                def2nns_train[definition].add(word)

        for w in definition.split():
            word2vec[w] = ''
        if file == 'test.txt': words_test.add(word)
        elif file == 'valid.txt': words_valid.add(word)
        elif file == 'train.txt': words_train.add(word)
sys.stderr.write('Done\n')

sys.stderr.write('Reading word2vec file...   ')

vec_num = 0
vec_dim = 0
i2w_test, i2w_valid, i2w_train = [], [], []
vec_test, vec_valid, vec_train = [], [], []
if args.filtering:
    word2vec_raw = {}
for line in open(args.vec, 'r'):
    word, vec = line.strip().split(' ', 1)
    if word in word2vec:
        if args.filtering:
            word2vec_raw[word] = vec

        word2vec[word] = [float(val) for val in vec.split()]
        vec_num += 1
        if vec_dim == 0:
            vec_dim = len(vec)

        if word in words_train:
            i2w_train.append(word)
            vec_train.append(word2vec[word])
        if word in words_valid:
            i2w_valid.append(word)
            vec_valid.append(word2vec[word])
        if word in words_test:
            i2w_test.append(word)
            vec_test.append(word2vec[word])

if args.mode != 'strong_cheat':
    if args.cuda:
        vec_test = torch.FloatTensor(vec_test).cuda()
        vec_valid = torch.FloatTensor(vec_valid).cuda()
        vec_train = torch.FloatTensor(vec_train).cuda()
    else:
        vec_test = torch.FloatTensor(vec_test)
        vec_valid = torch.FloatTensor(vec_valid)
        vec_train = torch.FloatTensor(vec_train)
sys.stderr.write('Done\n')


if args.filtering:
    sys.stderr.write('Writing filtered word2vec file...   ')
    sys.stderr.flush()
    with open(args.vec + '.filtered', 'w') as f:
        f.write(str(vec_num) + ' ' + str(vec_dim) + '\n')
        for word in word2vec_raw:
            if word2vec_raw[word] != '':
                f.write(word + ' ' + word2vec_raw[word] + '\n')
    sys.stderr.write('Done\n')

# find nearest neighbors
A = vec_train
for file in ['valid', 'test', 'train']:
    if file == 'train':
        B = A
        filename = 'train.nn'
        i2w = i2w_train
    elif file == 'valid':
        B = vec_valid
        filename = 'valid.nn'
        i2w = i2w_valid
    elif file == 'test':
        B = vec_test
        filename = 'test.nn'
        i2w = i2w_test

    trgWord_nnWord_sim_def = []
    if args.mode != 'strong_cheat':
        sys.stderr.write('Computing cosine similarity for ' + file + ' data...   ')
        sys.stderr.flush()
        AB = torch.mm(A, B.permute(1,0))
        A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
        B_norm = torch.norm(B, p=2, dim=1, keepdim=True)
        AB = AB / A_norm / B_norm.permute(1,0)
        sys.stderr.write('Done\n')

        # Sort
        sys.stderr.write('Searching nearest neighbors for ' + file + ' data...   ')
        sys.stderr.flush()
        vec_a = []
        vec_b = []
        AB_sorted, id_sorted = AB.sort(dim=0, descending=True)
        for b in range(AB_sorted.size(1)):
            k = 0
            trg_word = i2w[b]
            if args.mode in {'max_vocab', 'cheat'}:
                covered_vocab = set()
            if args.mode == 'cheat': # make vocab set that includes all words in definitions of the target word
                trg_def_split = [word2defs[trg_word][i].split() for i in range(len(word2defs[trg_word]))]
                trg_vocab = set()
                for sent in trg_def_split:
                    trg_vocab |= set(sent)

            for a in range(AB_sorted.size(0)):
                nn_word = i2w_train[id_sorted[a, b]]
                if is_usable(trg_word, nn_word):
                    if args.mode == 'min_risk':
                        definition = choose_min_risk_def(word2defs[nn_word])
                    elif args.mode == 'cheat':
                        definition, covered_vocab = choose_cheated_def(trg_vocab, covered_vocab, word2defs[nn_word])
                    elif args.mode == 'max_vocab':
                        definition, covered_vocab = choose_max_vocab_def(covered_vocab, word2defs[nn_word])

                    sim = AB_sorted[a, b]
                    k+= 1
                    vec_b.append([float(val) for val in word2vec[trg_word]])
                    vec_a.append([float(val) for val in word2vec[nn_word]])
                    trgWord_nnWord_sim_def.append((trg_word, nn_word, sim, definition))
                    if k >= args.K:
                        break
        if args.cuda:
            vec_a = torch.FloatTensor(vec_a).cuda()
            vec_b = torch.FloatTensor(vec_b).cuda()
        else:
            vec_a = torch.FloatTensor(vec_a)
            vec_b = torch.FloatTensor(vec_b)
        diff = vec_b - vec_a

        sys.stderr.write('Done\n')
        sys.stderr.write('Writing nearest neighbors to ' + filename + '...   ')
        sys.stderr.flush()
        with open(os.path.join(args.data, filename), 'w') as f:
            for i, (trg_word, nn_word, sim, definition) in enumerate(trgWord_nnWord_sim_def):
                f.write(trg_word + '\t' + nn_word + '\t' + str(sim) + '\t' + definition + '\t' + ' '.join(
                    [str(val) for val in diff[i]]) + '\n')

    elif args.mode == 'strong_cheat':
        trgWord_nn_def_sim_diff = []
        for i, trg_word in enumerate(i2w):
            nn_def_sim_diff = find_nearest_defs(trg_word, def2nns_train)
            print('L=', args.L, file, (i + 1) / len(i2w) * 100, '% finished')
            trgWord_nn_def_sim_diff.append((trg_word, nn_def_sim_diff))

        sys.stderr.write('Done\n')
        sys.stderr.write('Writing cheated N-best list to ' + filename + '...   ')
        sys.stderr.flush()
        with open(os.path.join(args.data, filename), 'w') as f:
            for trg_word, nn_def_sim_diff in trgWord_nn_def_sim_diff:
                for nn_word, definition, sim, diff in nn_def_sim_diff:
                    f.write(trg_word + '\t' + nn_word + '\t' + str(sim) + '\t' + definition + '\t' + ' '.join([str(val) for val in diff]) + '\n')

    sys.stderr.write('Done\n')





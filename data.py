import os
import sys
import torch
from torch.autograd import Variable
import random
import numpy as np
np.random.seed(505)
from util import cudaw

# from docutils.nodes import description

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class NorasetCorpus(object):
    def __init__(self, data_path, vec_path, max_vocab_size=-1, mode='train', ignore_sense_id=False, train_src='both', args=None):
        # Builds a vocabulary set from train data
        self.special_tokens = ['<unk>', '<bos>', '<eos>']
        self.vec_path = vec_path
        self.max_char_len = 0
        try:
            self.word2id, self.id2word = self.load_vocab(os.path.join(data_path, 'train.txt.vocab'), max_vocab_size)
            sys.stderr.write('Loaded vocab file from ' + os.path.join(data_path, 'train.txt.vocab') + '\n')
            self.char2id, self.id2char = self.load_vocab(os.path.join(data_path, 'train.txt.char'), -1)
            sys.stderr.write('Loaded char vocab file from ' + os.path.join(data_path, 'train.txt.char') + '\n')
        except:
            sys.stderr.write('No vocab files in ' + os.path.join(data_path, 'train.txt.vocab') + '\n')
            sys.stderr.write('               or ' + os.path.join(data_path, 'train.txt.char') + '\n')
            self.word2id, self.id2word, self.char2id, self.id2char = self.build_vocab(os.path.join(data_path, 'train.txt'), max_vocab_size)
            sys.stderr.write('Output vocab file to ' + os.path.join(data_path, 'train.txt.vocab') + '\n')
            sys.stderr.write('                 and ' + os.path.join(data_path, 'train.txt.char') + '\n')

        # Tokenize tokens in text data
        self.max_len = 0
        self.trgWord2descs = {}
        sys.stderr.write('Reading corpora...')
        if mode == 'train':
            self.train, self.train_ntoken, self.ref_train, self.train_orig = self.tokenize(os.path.join(data_path, 'train.txt'), ignore_sense_id, train_src)
            self.valid, self.valid_ntoken, self.ref_valid, self.valid_orig = self.tokenize(os.path.join(data_path, 'valid.txt'), ignore_sense_id, 'both')
        self.test, self.test_ntoken, self.ref_test, self.test_orig  = self.tokenize(os.path.join(data_path, 'test.txt'), ignore_sense_id, 'both')
        sys.stderr.write('  Done\n')

        # Reads vectors
        sys.stderr.write('Reading vector file...')
        self.word2vec = self.read_vector(vec_path, self.trgWord2descs)
        if mode == 'train':
            self.train = self.add_vector(self.train)
            self.valid = self.add_vector(self.valid)
        self.test = self.add_vector(self.test)
        sys.stderr.write('  Done\n')

        # Convert words into char-ids
        if mode == 'train':
            self.train = self.add_chars(self.train)
            self.valid = self.add_chars(self.valid)
        self.test = self.add_chars(self.test)

    def build_vocab(self, path, max_vocab_size):
        assert os.path.exists(path)
        word2freq = {}
        char2freq = {}
        with open(path, 'r') as f:
            for line in f:
                description = line.strip().split('\t')[3]
                char_buf = ''
                for w in description.split():
                    if w in word2freq:
                        word2freq[w] += 1
                    elif w not in self.special_tokens: # ignore special tokens included in training data
                        word2freq[w] = 1

                    if w not in self.special_tokens:
                        for c in w.split('%', 1)[0]: # ignore sense id
                            if c in char2freq:
                                char2freq[c] += 1
                            else:
                                char2freq[c] = 1

        # sort vocabularies in order of frequency and prepend special tokens
        sorted_word2freq = sorted(word2freq.items(), key=lambda x: -x[1])
        for special_token in self.special_tokens:
            sorted_word2freq.insert(0, (special_token, 0))
        id2word = []
        word2id = {}
        with open(path + '.vocab', 'w') as f:
            for i, (word, freq) in enumerate(sorted_word2freq):
                f.write(word + '\t' + str(i) + '\t' + str(freq) + '\n')
                if max_vocab_size == -1 or i < max_vocab_size + len(self.special_tokens):
                    word2id[word] = i
                    id2word.append(word)

        # Do the same things to char vocabs
        sorted_char2freq = sorted(char2freq.items(), key=lambda x: -x[1])
        for special_token in ['<bow>', '<eow>']:
            sorted_char2freq.insert(0, (special_token, 0))
        id2char = []
        char2id = {}
        with open(path + '.char', 'w') as f:
            for i, (char, freq) in enumerate(sorted_char2freq):
                f.write(char + '\t' + str(i) + '\t' + str(freq) + '\n')
                char2id[char] = i
                id2char.append(char)

        return word2id, id2word, char2id, id2char

    def load_vocab(self, path, max_vocab_size):
        assert os.path.exists(path)
        id2word = []
        word2id = {}
        for line in open(path, 'r'):
            word_id = line.strip().split('\t', 2)[:2]
            if max_vocab_size == -1 or int(word_id[1]) < max_vocab_size + len(self.special_tokens):
                word2id[word_id[0]] = int(word_id[1])
                id2word.append(word_id[0])
            else:
                break

        return word2id, id2word

    def tokenize(self, path, ignore_sense_id, train_src):
        assert os.path.exists(path)
        word_desc = []  # [(srcWord0, [trgId0, trgId1, ...]), (srcWord1, [trgId0, trgId1, ...])]
        word_desc_orig = [] # [(srcWord0, [trgWord0, trgWord1, ...]), ... ]
        ntoken = 0
        ref = {}
        if train_src not in set(['wordnet', 'gcide', 'both']):
            sys.stderr.write("train_src has to be one of {'wordnet' | 'gcide' | 'both'}\n")
            exit()
        with open(path, 'r') as f:
            for line in f:
                elems = line.strip().split('\t')
                if train_src in set(['wordnet', 'gcide']) and train_src != elems[2]:
                    continue

                word = elems[0]
                word_wo_id = word.split('%', 1)[0]
                if ignore_sense_id:
                    word = word_wo_id
                if word_wo_id not in ref:
                    ref[word_wo_id] = []
                ref[word_wo_id].append(elems[3])

                if len(word) + 2 > self.max_char_len:
                    self.max_char_len = len(word) + 2
                word_desc.append((word, []))
                description = ['<bos>'] + elems[3].split() + ['<eos>']
                word_desc_orig.append((word,description))
                for w in description:
                    if w in self.word2id:
                        word_desc[-1][1].append(self.word2id[w])
                    else:
                        word_desc[-1][1].append(self.word2id['<unk>'])

                if word not in self.trgWord2descs:
                    self.trgWord2descs[word] = []

                if word_desc[-1][1] not in self.trgWord2descs[word]: # TODO: this must be hyper slow
                    self.trgWord2descs[word].append(word_desc[-1][1])

                ntoken += (len(description) - 1)  # including <eos>, not including <bos>
                if len(description) - 1 > self.max_len:
                    self.max_len = len(description) - 1

        return word_desc, ntoken, ref, word_desc_orig

    def read_vector(self, path, words):
        """Reads word2vec file."""
        assert os.path.exists(path)
        word2vec = {} # {srdWord0: [dim0, dim1, ..., dim299], srcWord1: [dim0, dim1, ..., dim299]}
        words_lemma = set()
        for word in words:
            words_lemma.add(word.split('%', 1)[0])
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                elems = line.strip().split(' ')
                if i == 0 and len(elems) == 2: # The first line in word2vec file
                    continue
                word = elems[0]
                if word in words or word == '<unk>' or word in words_lemma:
                    word2vec[word] = [float(val) for val in elems[1:]]

        if len(word2vec) == 0:
            print('cannot read word2vec file. check file format below:')
            print('word2vec file:', word)
            print('vocab file:', list(words.keys())[:5])
            exit()

        if '<unk>' not in word2vec:
            word2vec['<unk>'] = np.random.uniform(-0.05, 0.05, len(word2vec[list(word2vec.keys())[0]])).tolist()

        return word2vec

    def add_vector(self, word_desc):
        word_vec_desc = []
        for word, description in word_desc:
            if word in self.word2vec:
                word_vec_desc.append((word, self.word2vec[word], description))
            else:
                word_lemma = word.split('%', 1)[0]
                if word_lemma in self.word2vec:
                    word_vec_desc.append((word, self.word2vec[word_lemma], description))
                else:
                    word_vec_desc.append((word, self.word2vec['<unk>'], description))

        return word_vec_desc

    def add_chars(self, word_vec_desc):
        """convert words ([word1, word2, ...]) into char-ids """
        word_char_vec_desc = []
        for word, vec, description in word_vec_desc:
            char_ids = [self.char2id['<bow>']]
            for c in word.split('%', 1)[0]: # ignore sense_id
                if c in self.char2id:
                    char_ids.append(self.char2id[c])
            char_ids.append(self.char2id['<eow>'])
            word_char_vec_desc.append((word, char_ids, vec, description))

        return word_char_vec_desc

    def batch_iterator(self, vec_desc, max_batch_size, cuda=False, shuffle=False, mode='train', ignore_index=-100, seed_feeding=True, ignore_duplicates=False):
        """
        mode: {train (use whole data; volatile=False) | valid (use whole data; volatile=True) | test (don't use duplicate data; volatile=True)}
        """
        data = []
        if ignore_duplicates:
            words = set()
            for i in range(len(vec_desc)):
                word = vec_desc[i][0].split('%', 1)[0]
                if word not in words:
                    data.append(vec_desc[i])
                    words.add(word)
        else:
            data = vec_desc

        if shuffle:
            random.shuffle(data)

        batch_num = -(-len(data) // max_batch_size)
        batch_size = max_batch_size
        for i in range(batch_num):
            # for the last remainder batch, set the batch size smaller than others
            if i == batch_num - 1 and len(data) % max_batch_size != 0:
                batch_size = len(data) % max_batch_size

            batch = data[i*max_batch_size:(i+1)*max_batch_size]
            words = []
            chars = torch.FloatTensor(batch_size, self.max_char_len, len(self.id2char)).zero_() #TODO: Is zero filling ok?
            vecs = torch.FloatTensor(batch_size, len(data[0][2]))
            srcs_T = torch.LongTensor(batch_size, self.max_len)
            if seed_feeding:
                trgs_T = torch.LongTensor(batch_size, self.max_len + 1) # src is 1-element less than trg because we will feed Seed vector later
            else:
                trgs_T = torch.LongTensor(batch_size, self.max_len)

            for j, (word, char, vec, description) in enumerate(batch):
                words.append(word)
                # one-hot repl. of words
                for k, c in enumerate(char):
                    chars[j, k, c] = 1.0

                # padding the source id sequence with <eos>
                padded_desc = description[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description[:-1]))
                # padding the target id sequence with ignore_indices
                if seed_feeding:
                    shifted_desc = [ignore_index] + description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                else:
                    shifted_desc = description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                vecs[j] = torch.FloatTensor(vec)
                srcs_T[j] = torch.Tensor(padded_desc)
                trgs_T[j] = torch.Tensor(shifted_desc)

            # reshape the tensors
            srcs = torch.transpose(srcs_T, 0, 1).contiguous()
            trgs = torch.transpose(trgs_T, 0, 1).contiguous().view(-1)

            if not cuda:
                yield (words, Variable(chars.unsqueeze(1)), Variable(vecs), Variable(srcs, volatile=False if mode =='train' else True), Variable(trgs))
            else:
                yield (words, Variable(chars.unsqueeze(1)).cuda(), Variable(vecs).cuda(), Variable(srcs, volatile=False if mode =='train' else True).cuda(), Variable(trgs).cuda())

    def sample_train_data(self, data_usage):
        sample_size = int(len(self.train) * data_usage / 100)
        self.train = self.train[:sample_size]
        return


class ExampleCorpus(NorasetCorpus):
    def __init__(self, args, mode='train'):
        super(ExampleCorpus, self).__init__(args.data, args.vec, args.vocab_size, ignore_sense_id=args.ignore_sense_id, train_src='both', mode=mode, args=args)
        self.topk = args.topk
        sys.stderr.write('Reading example files...')
        if mode == 'train':
            self.train_word_egs = self.read_examples(os.path.join(args.data, 'train.eg'), self.topk, args.ignore_sense_id)
            self.valid_word_egs = self.read_examples(os.path.join(args.data, 'valid.eg'), self.topk, args.ignore_sense_id)
            self.train = self.add_examples(self.train, self.train_word_egs)
            self.valid = self.add_examples(self.valid, self.valid_word_egs)
        self.test_word_egs = self.read_examples(os.path.join(args.data, 'test.eg'), self.topk, args.ignore_sense_id)
        self.test = self.add_examples(self.test, self.test_word_egs)

    def build_vocab(self, path, max_vocab_size):
        self.special_tokens = ['<TRG>', '<unk>', '<bos>', '<eos>']
        return super(ExampleCorpus, self).build_vocab(path, max_vocab_size)

    def read_examples(self, path, topk, ignore_sense_id):
        assert os.path.exists(path)
        word_egs = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                word, eg = line.strip().split('\t')
                word_lemma = word.split('%', 1)[0]
                if ignore_sense_id:
                    word = word_lemma
                # if word not in word2egs:
                #     word2egs[word] = []
                # word2egs[word].append(eg.split(' '))
                word_egs.append((word, eg.split(' ')))
        # return word2egs
        return word_egs

    def add_examples(self, word_char_vec_desc, word_egs):
        word_char_vec_desc_eg = []
        for i, (word, char, vec, desc) in enumerate(word_char_vec_desc):
            eg_id = []
            for w in ['<bos>'] + word_egs[i][1] + ['<eos>']:
                if w in self.word2id:
                    eg_id.append(self.word2id[w])
                else:
                    eg_id.append(self.word2id['<unk>'])
            word_char_vec_desc_eg.append((word, char, vec, desc, eg_id))
        return word_char_vec_desc_eg

    def batch_iterator(self, word_char_vec_desc_eg, max_batch_size, cuda=False, shuffle=False, mode='train', ignore_index=-100, seed_feeding=True, ignore_duplicates=False):
        """
        mode: {train (use whole data; volatile=False) | valid (use whole data; volatile=True) | test (don't use duplicate data; volatile=True)}
        """
        data = []
        if ignore_duplicates:
            words = set()
            for i in range(len(word_char_vec_desc_eg)):
                word = word_char_vec_desc_eg[i][0].split('%', 1)[0]
                if word not in words:
                    data.append(word_char_vec_desc_eg[i])
                    words.add(word)
        else:
            data = word_char_vec_desc_eg

        if shuffle:
            random.shuffle(data)
        batch_num = -(-len(data) // max_batch_size)
        batch_size = max_batch_size
        for i in range(batch_num):
            # for the last remainder batch, set the batch size smaller than others
            if i == batch_num - 1 and len(data) % max_batch_size != 0:
                batch_size = len(data) % max_batch_size

            batch = data[i*max_batch_size:(i+1)*max_batch_size]
            words = []
            chars = torch.FloatTensor(batch_size, self.max_char_len, len(self.id2char)).zero_()
            vecs = torch.FloatTensor(batch_size, len(data[0][2]))
            srcs_T = torch.LongTensor(batch_size, self.max_len)

            # tensors for EDIT models
            max_len_eg = max([len(batch[i][4]) for i in range(len(batch))])
            eg_T = torch.LongTensor(batch_size, max_len_eg)
            eg_mask_T = torch.FloatTensor(batch_size, max_len_eg).fill_(-float('inf'))

            if seed_feeding:
                trgs_T = torch.LongTensor(batch_size, self.max_len + 1) # src is 1-element less than trg because we will feed Seed vector later
            else:
                trgs_T = torch.LongTensor(batch_size, self.max_len)

            for j, (word, char, vec, description, example) in enumerate(batch):
                words.append(word)
                # one-hot repl. of words
                for k, c in enumerate(char):
                    chars[j, k, c] = 1.0

                # padding the source id sequence with <eos>
                padded_desc = description[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description[:-1]))
                # padding the target id sequence with ignore_indices
                if seed_feeding:
                    shifted_desc = [ignore_index] + description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                else:
                    shifted_desc = description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                vecs[j] = torch.FloatTensor(vec)
                srcs_T[j] = torch.Tensor(padded_desc)
                trgs_T[j] = torch.Tensor(shifted_desc)

                # padding the examples
                padded_eg = example[:-1] + [self.word2id['<eos>']] * (max_len_eg - len(example[:-1]))
                eg_T[j] = torch.Tensor(padded_eg)
                eg_mask_T[j][:len(example)].fill_(0)

            # reshape the tensors
            srcs = torch.transpose(srcs_T, 0, 1).contiguous()
            trgs = torch.transpose(trgs_T, 0, 1).contiguous().view(-1)
            eg = torch.transpose(eg_T, 0, 1).contiguous()
            eg_mask = torch.transpose(eg_mask_T, 0, 1).contiguous()

            yield (words, cudaw(Variable(chars.unsqueeze(1))), cudaw(Variable(vecs)),
                   cudaw(Variable(srcs, volatile=False if mode =='train' else True)),
                   cudaw(Variable(trgs)), cudaw(Variable(eg, volatile=False if mode =='train' else True)),
                   cudaw(Variable(eg_mask, volatile=False if mode == 'train' else True)))

class EditCorpus(NorasetCorpus):
    def __init__(self, data_path, vec_path, max_vocab_size=-1, mode='train', topk=-1, min_sim=0, ignore_sense_id=False, train_src='both'):
        super(EditCorpus, self).__init__(data_path, vec_path, max_vocab_size, ignore_sense_id=ignore_sense_id, train_src=train_src)
        self.topk = topk
        sys.stderr.write('Reading nearest neighbor files...')
        if mode == 'train':
            self.train_word2nns = self.read_nns(os.path.join(data_path, 'train.nn'), topk, min_sim)
            self.valid_word2nns = self.read_nns(os.path.join(data_path, 'valid.nn'), topk, min_sim)
            self.train = self.add_nns(self.train, self.train_word2nns, 'train')
            self.valid = self.add_nns(self.valid, self.valid_word2nns, 'test')
        self.test_word2nns= self.read_nns(os.path.join(data_path, 'test.nn'), topk, min_sim)
        self.test = self.add_nns(self.test, self.test_word2nns, 'test')
        sys.stderr.write('  Done\n')

    def read_nns(self, path, topk, min_sim):
        assert os.path.exists(path)
        word2nn_sim_nnDef_diff = {} # {'information': [('data', 0.54, ['facts', 'and', 'statistics', ...], [Vinformation - Vdata]), ...}

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                word, nn, similarity, desc_nn, diff = line.strip().split('\t')
                similarity = float(similarity)
                diff = [float(val) for val in diff.split()]

                if word not in word2nn_sim_nnDef_diff:
                    word2nn_sim_nnDef_diff[word] = []
                if (topk == -1 or topk > len(word2nn_sim_nnDef_diff[word])) and (min_sim <= similarity):
                    word2nn_sim_nnDef_diff[word].append((nn, similarity, desc_nn, diff))

        return word2nn_sim_nnDef_diff

    def add_nns(self, word_char_vec_desc, word2nns, mode):
        word_char_vec_desc_diff_descNn_sim = []
        for word, char, vec, desc in word_char_vec_desc:
            diff_descNn_sim = []
            for (nn, similarity, desc_nn, diff) in word2nns[word]: # for each nn
                # convert desc_nn to ids
                desc_nn = ['<bos>'] + desc_nn.split() + ['<eos>']
                desc_nn_id = []
                for w in desc_nn:
                    if w in self.word2id:
                        desc_nn_id.append(self.word2id[w])
                    else:
                        desc_nn_id.append(self.word2id['<unk>'])

                if mode == 'train':  # treat each nn as an independent data
                    word_char_vec_desc_diff_descNn_sim.append((word, char, vec, desc, diff, desc_nn_id, similarity))
                else:
                    diff_descNn_sim.append((diff, desc_nn_id, similarity))

            if mode != 'train': # put all the nns, their diff vectors, and their similarities into a single sample
                word_char_vec_desc_diff_descNn_sim.append((word, char, vec, desc, diff_descNn_sim))

        return word_char_vec_desc_diff_descNn_sim

    def batch_iterator(self, word_char_vec_desc_vecDiff_descNn_sim, max_batch_size, cuda=False, shuffle=False, mode='train', ignore_index=-100, seed_feeding=True, ignore_duplicates=False):
        """
        mode: {train (use whole data; volatile=False) | valid (use whole data; volatile=True) | test (don't use duplicate data; volatile=True)}
        """
        data = []
        if ignore_duplicates:
            words = set()
            for i in range(len(word_char_vec_desc_vecDiff_descNn_sim)):
                word = word_char_vec_desc_vecDiff_descNn_sim[i][0].split('%', 1)[0]
                if word not in words:
                    data.append(word_char_vec_desc_vecDiff_descNn_sim[i])
                    words.add(word)
        else:
            data = word_char_vec_desc_vecDiff_descNn_sim

        if mode == 'train':
            if shuffle:
                random.shuffle(data)
            batch_num = -(-len(data) // max_batch_size)
            batch_size = max_batch_size
            for i in range(batch_num):
                # for the last remainder batch, set the batch size smaller than others
                if i == batch_num - 1 and len(data) % max_batch_size != 0:
                    batch_size = len(data) % max_batch_size

                batch = data[i*max_batch_size:(i+1)*max_batch_size]
                words = []
                chars = torch.FloatTensor(batch_size, self.max_char_len, len(self.id2char)).zero_()
                vecs = torch.FloatTensor(batch_size, len(data[0][2]))
                srcs_T = torch.LongTensor(batch_size, self.max_len)

                # tensors for EDIT models
                descs_nn_T = torch.LongTensor(batch_size, self.max_len)
                descs_nn_mask_T = torch.FloatTensor(batch_size, self.max_len).fill_(-float('inf'))
                vecs_diff = torch.FloatTensor(batch_size, len(data[0][2]))
                sims = torch.FloatTensor(batch_size)

                if seed_feeding:
                    trgs_T = torch.LongTensor(batch_size, self.max_len + 1) # src is 1-element less than trg because we will feed Seed vector later
                else:
                    trgs_T = torch.LongTensor(batch_size, self.max_len)

                for j, (word, char, vec, description, vec_diff, description_nn, sim) in enumerate(batch):
                    words.append(word)
                    # one-hot repl. of words
                    for k, c in enumerate(char):
                        chars[j, k, c] = 1.0

                    # padding the source id sequence with <eos>
                    padded_desc = description[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description[:-1]))
                    # padding the target id sequence with ignore_indices
                    if seed_feeding:
                        shifted_desc = [ignore_index] + description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                    else:
                        shifted_desc = description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                    vecs[j] = torch.FloatTensor(vec)
                    srcs_T[j] = torch.Tensor(padded_desc)
                    trgs_T[j] = torch.Tensor(shifted_desc)

                    # padding the description of a nn word
                    padded_desc_nn = description_nn[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description_nn[:-1]))
                    descs_nn_T[j] = torch.Tensor(padded_desc_nn)
                    descs_nn_mask_T[j][:len(description_nn)].fill_(0)
                    vecs_diff[j] = torch.FloatTensor(vec_diff)
                    sims[j] = sim

                # reshape the tensors
                srcs = torch.transpose(srcs_T, 0, 1).contiguous()
                trgs = torch.transpose(trgs_T, 0, 1).contiguous().view(-1)
                descs_nn = torch.transpose(descs_nn_T, 0, 1).contiguous()
                descs_nn_mask = torch.transpose(descs_nn_mask_T, 0, 1).contiguous()

                if not cuda:
                    yield (words, Variable(chars.unsqueeze(1)), Variable(vecs),
                           Variable(srcs, volatile=False if mode =='train' else True),
                           Variable(trgs), Variable(vecs_diff),
                           Variable(descs_nn, volatile=False if mode =='train' else True),
                           Variable(descs_nn_mask, volatile=False if mode == 'train' else True),
                           Variable(sims))
                else:
                    yield (words, Variable(chars.unsqueeze(1)).cuda(), Variable(vecs).cuda(),
                           Variable(srcs, volatile=False if mode =='train' else True).cuda(),
                           Variable(trgs).cuda(), Variable(vecs_diff).cuda(),
                           Variable(descs_nn, volatile=False if mode =='train' else True).cuda(),
                           Variable(descs_nn_mask, volatile=False if mode == 'train' else True).cuda(),
                           Variable(sims).cuda())

        else: # To decode words in an ensembling-like manner, make all the target words inside a batch same.
            for (word, char, vec, description, vecDiff_descNn_sim) in data:
                batch_size = len(vecDiff_descNn_sim)
                words = []
                chars = torch.FloatTensor(batch_size, self.max_char_len, len(self.id2char)).zero_()
                vecs = torch.FloatTensor(batch_size, len(data[0][2]))
                srcs_T = torch.LongTensor(batch_size, self.max_len)

                # tensors for EDIT models
                descs_nn_T = torch.LongTensor(batch_size, self.max_len)
                descs_nn_mask_T = torch.FloatTensor(batch_size, self.max_len).fill_(-float('inf'))
                vecs_diff = torch.FloatTensor(batch_size, len(data[0][2]))
                sims = torch.FloatTensor(batch_size)

                if seed_feeding:
                    trgs_T = torch.LongTensor(1, self.max_len + 1)
                else:
                    trgs_T = torch.LongTensor(1, self.max_len)


                # padded_desc and shifted_desc are same for all the data inside a batch
                padded_desc = description[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description[:-1]))
                if seed_feeding:
                    shifted_desc = [ignore_index] + description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                else:
                    shifted_desc = description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))

                trgs_T[0] = torch.Tensor(shifted_desc)
                for j, (vec_diff, description_nn, sim) in enumerate(vecDiff_descNn_sim):
                    words.append(word)
                    vecs[j] = torch.FloatTensor(vec)
                    srcs_T[j] = torch.Tensor(padded_desc)
                    for k, c in enumerate(char):
                        chars[j, k, c] = 1.0

                    # padding the description of a nn word
                    padded_desc_nn = description_nn[:-1] + [self.word2id['<eos>']] * (
                    self.max_len - len(description_nn[:-1]))
                    descs_nn_T[j] = torch.Tensor(padded_desc_nn)
                    descs_nn_mask_T[j][:len(description_nn)].fill_(0)
                    vecs_diff[j] = torch.FloatTensor(vec_diff)
                    sims[j] = sim

                # reshape the tensors
                srcs = torch.transpose(srcs_T, 0, 1).contiguous()
                trgs = torch.transpose(trgs_T, 0, 1).contiguous().view(-1)
                descs_nn = torch.transpose(descs_nn_T, 0, 1).contiguous()
                descs_nn_mask = torch.transpose(descs_nn_mask_T, 0, 1).contiguous()

                if not cuda:
                    yield (words, Variable(chars.unsqueeze(1)), Variable(vecs),
                           Variable(srcs, volatile=False if mode == 'train' else True),
                           Variable(trgs), Variable(vecs_diff),
                           Variable(descs_nn, volatile=False if mode == 'train' else True),
                           Variable(descs_nn_mask, volatile=False if mode == 'train' else True),
                           Variable(sims))
                else:
                    yield (words, Variable(chars.unsqueeze(1)).cuda(), Variable(vecs).cuda(),
                           Variable(srcs, volatile=False if mode == 'train' else True).cuda(),
                           Variable(trgs).cuda(), Variable(vecs_diff).cuda(),
                           Variable(descs_nn, volatile=False if mode == 'train' else True).cuda(),
                           Variable(descs_nn_mask, volatile=False if mode == 'train' else True).cuda(),
                           Variable(sims).cuda())


class EditCorpus2(NorasetCorpus):
    '''corpus for edit model w/ a multi-encoder'''
    def __init__(self, data_path, vec_path, max_vocab_size=-1, mode='train', topk=-1, min_sim=0, ignore_sense_id=False, train_src='both', raw_nns=False):
        super(EditCorpus2, self).__init__(data_path, vec_path, max_vocab_size, ignore_sense_id=ignore_sense_id, train_src=train_src)
        self.topk = topk
        sys.stderr.write('Reading nearest neighbor files...')
        if mode == 'train':
            self.train_word2nns = self.read_nns(os.path.join(data_path, 'train.nn'), topk, min_sim)
            self.valid_word2nns = self.read_nns(os.path.join(data_path, 'valid.nn'), topk, min_sim)
            self.train = self.add_nns(self.train, self.train_word2nns, 'train_multi_encoder')
            self.valid = self.add_nns(self.valid, self.valid_word2nns, 'test')
        self.test_word2nns= self.read_nns(os.path.join(data_path, 'test.nn'), topk, min_sim)
        self.test = self.add_nns(self.test, self.test_word2nns, 'test', raw_nns=raw_nns)
        sys.stderr.write('  Done\n')

    def read_nns(self, path, topk, min_sim):
        assert os.path.exists(path)
        word2nn_sim_nnDef_diff = {} # {'information': [('data', 0.54, ['facts', 'and', 'statistics', ...], [Vinformation - Vdata]), ...}

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                word, nn, similarity, desc_nn, diff = line.strip().split('\t')
                similarity = float(similarity)
                diff = [float(val) for val in diff.split()]

                if word not in word2nn_sim_nnDef_diff:
                    word2nn_sim_nnDef_diff[word] = []
                if (topk == -1 or topk > len(word2nn_sim_nnDef_diff[word])) and (min_sim <= similarity):
                    word2nn_sim_nnDef_diff[word].append((nn, similarity, desc_nn, diff))

        return word2nn_sim_nnDef_diff

    def add_nns(self, word_char_vec_desc, word2nns, mode, raw_nns=False):
        word_char_vec_desc_diff_descNn_sim = []
        for word, char, vec, desc in word_char_vec_desc:
            diff_descNn_sim = []
            if word not in word2nns:
                continue
            for (nn, similarity, desc_nn, diff) in word2nns[word]: # for each nn
                # convert desc_nn to ids
                desc_nn = desc_nn.split() + ['<eos>']
                if raw_nns: # do not covert nns into ids; only used in Translator.top1_copy()
                    desc_nn_id = desc_nn[1:-1]
                else:
                    desc_nn_id = []
                    for w in desc_nn:
                        if w in self.word2id:
                            desc_nn_id.append(self.word2id[w])
                        else:
                            desc_nn_id.append(self.word2id['<unk>'])
                diff_descNn_sim.append((diff, desc_nn_id, similarity, desc_nn))
            word_char_vec_desc_diff_descNn_sim.append((word, char, vec, desc, diff_descNn_sim))

        return word_char_vec_desc_diff_descNn_sim

    def batch_iterator(self, word_char_vec_desc_vecDiff_descNn_sim, max_batch_size, cuda=False, shuffle=False, mode='train', ignore_index=-100, seed_feeding=True, ignore_duplicates=False):
        """
        mode: {train (use whole data; volatile=False) | valid (use whole data; volatile=True) | test (don't use duplicate data; volatile=True)}
        """
        data = []
        if ignore_duplicates:
            words = set()
            for i in range(len(word_char_vec_desc_vecDiff_descNn_sim)):
                word = word_char_vec_desc_vecDiff_descNn_sim[i][0].split('%', 1)[0]
                if word not in words:
                    data.append(word_char_vec_desc_vecDiff_descNn_sim[i])
                    words.add(word)
        else:
            data = word_char_vec_desc_vecDiff_descNn_sim

        if shuffle:
            random.shuffle(data)

        batch_num = -(-len(data) // max_batch_size)
        batch_size = max_batch_size
        for i in range(batch_num):
            # for the last remainder batch, set the batch size smaller than others
            if i == batch_num - 1 and len(data) % max_batch_size != 0:
                batch_size = len(data) % max_batch_size

            batch = data[i*max_batch_size:(i+1)*max_batch_size]
            words = []
            chars = torch.FloatTensor(batch_size, self.max_char_len, len(self.id2char)).zero_()
            vecs = torch.FloatTensor(batch_size, len(data[0][2]))
            srcs_T = torch.LongTensor(batch_size, self.max_len)

            # tensors for EDIT models
            descs_nn_T = torch.LongTensor(batch_size * self.topk, self.max_len).fill_(0)
            descs_nn_mask_T = torch.FloatTensor(batch_size * self.topk, self.max_len).fill_(-float('inf'))
            vecs_diff = torch.FloatTensor(batch_size * self.topk, len(data[0][2]))
            sims = torch.FloatTensor(batch_size * self.topk)

            if seed_feeding:
                trgs_T = torch.LongTensor(batch_size, self.max_len + 1) # src is 1-element less than trg because we will feed Seed vector later
            else:
                trgs_T = torch.LongTensor(batch_size, self.max_len)


            for j, (word, char, vec, description, diff_descNn_sim) in enumerate(batch):
                words.append(word)
                # one-hot repl. of words
                for k, c in enumerate(char):
                    chars[j, k, c] = 1.0

                # padding the source id sequence with <eos>
                padded_desc = description[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description[:-1]))
                # padding the target id sequence with ignore_indices
                if seed_feeding:
                    shifted_desc = [ignore_index] + description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                else:
                    shifted_desc = description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                vecs[j] = torch.FloatTensor(vec)
                srcs_T[j] = torch.Tensor(padded_desc)
                trgs_T[j] = torch.Tensor(shifted_desc)

                for k, (vec_diff, description_nn, sim, description) in enumerate(diff_descNn_sim):
                    # padding the description of a nn word
                    padded_desc_nn = description_nn[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description_nn[:-1]))
                    descs_nn_T[j * self.topk + k] = torch.Tensor(padded_desc_nn)
                    descs_nn_mask_T[j * self.topk + k][:len(description_nn)].fill_(0)
                    vecs_diff[j * self.topk + k] = torch.FloatTensor(vec_diff)
                    sims[j * self.topk + k] = sim

            # reshape the tensors
            srcs = torch.transpose(srcs_T, 0, 1).contiguous()
            trgs = torch.transpose(trgs_T, 0, 1).contiguous().view(-1)
            descs_nn = torch.transpose(descs_nn_T, 0, 1).contiguous()
            descs_mask_nn = torch.transpose(descs_nn_mask_T, 0, 1).contiguous()

            if not cuda:
                yield (words, Variable(chars.unsqueeze(1)), Variable(vecs),
                       Variable(srcs, volatile=False if mode =='train' else True),
                       Variable(trgs), Variable(vecs_diff),
                       Variable(descs_nn, volatile=False if mode =='train' else True),
                       Variable(descs_mask_nn, volatile=False if mode == 'train' else True),
                       Variable(sims))
            else:
                yield (words, Variable(chars.unsqueeze(1)).cuda(), Variable(vecs).cuda(),
                       Variable(srcs, volatile=False if mode =='train' else True).cuda(),
                       Variable(trgs).cuda(), Variable(vecs_diff).cuda(),
                       Variable(descs_nn, volatile=False if mode =='train' else True).cuda(),
                       Variable(descs_mask_nn, volatile=False if mode == 'train' else True).cuda(),
                       Variable(sims).cuda())

class EditCorpus3(NorasetCorpus):
    '''corpus for edit model w/ COPYNET'''
    def __init__(self, data_path, vec_path, max_vocab_size=-1, mode='train', topk=-1, min_sim=0, ignore_sense_id=False, train_src='both', raw_nns=False):
        super(EditCorpus3, self).__init__(data_path, vec_path, max_vocab_size, ignore_sense_id=ignore_sense_id, train_src=train_src)
        self.topk = topk
        sys.stderr.write('Reading nearest neighbor files...')
        if mode == 'train':
            self.train_word2nns = self.read_nns(os.path.join(data_path, 'train.nn'), topk, min_sim)
            self.valid_word2nns = self.read_nns(os.path.join(data_path, 'valid.nn'), topk, min_sim)
            self.train = self.add_nns(self.train, self.train_word2nns, 'train_multi_encoder')
            self.valid = self.add_nns(self.valid, self.valid_word2nns, 'test')
            self.train = self.add_orig(self.train, self.train_orig)
            self.valid = self.add_orig(self.valid, self.valid_orig)
        self.test_word2nns= self.read_nns(os.path.join(data_path, 'test.nn'), topk, min_sim)
        self.test = self.add_nns(self.test, self.test_word2nns, 'test', raw_nns=raw_nns)
        self.test = self.add_orig(self.test, self.test_orig)
        sys.stderr.write('  Done\n')

    def read_nns(self, path, topk, min_sim):
        assert os.path.exists(path)
        word2nn_sim_nnDef_diff = {} # {'information': [('data', 0.54, ['facts', 'and', 'statistics', ...], [Vinformation - Vdata]), ...}

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                word, nn, similarity, desc_nn, diff = line.strip().split('\t')
                similarity = float(similarity)
                diff = [float(val) for val in diff.split()]

                if word not in word2nn_sim_nnDef_diff:
                    word2nn_sim_nnDef_diff[word] = []
                if (topk == -1 or topk > len(word2nn_sim_nnDef_diff[word])) and (min_sim <= similarity):
                    word2nn_sim_nnDef_diff[word].append((nn, similarity, desc_nn, diff))

        return word2nn_sim_nnDef_diff

    def add_nns(self, word_char_vec_desc, word2nns, mode, raw_nns=False):
        word_char_vec_desc_diff_descNn_sim = []
        for word, char, vec, desc in word_char_vec_desc:
            diff_descNn_sim = []
            if word not in word2nns:
                continue
            for (nn, similarity, desc_nn, diff) in word2nns[word]: # for each nn
                oov_num = 0
                # convert desc_nn to ids
                desc_nn = desc_nn.split()
                if raw_nns: # do not covert nns into ids; only used in Translator.top1_copy()
                    desc_nn_id = desc_nn
                else:
                    desc_nn_id = []
                    for w in desc_nn:
                        if w in self.word2id:
                            desc_nn_id.append(self.word2id[w])
                        else:
                            desc_nn_id.append(self.word2id['<unk>'])
                            oov_num += 1
                diff_descNn_sim.append((diff, desc_nn_id, similarity, desc_nn, oov_num))
            word_char_vec_desc_diff_descNn_sim.append((word, char, vec, desc, diff_descNn_sim))

        return word_char_vec_desc_diff_descNn_sim

    def add_orig(self, word_char_vec_desc_diff_descNn_sim, orig):
        word_char_vec_desc_diff_descNn_sim_orig = []
        for i, (word, char, vec, desc, diff_descNn_sim) in enumerate(word_char_vec_desc_diff_descNn_sim):
            word_char_vec_desc_diff_descNn_sim_orig.append((word, char, vec, desc, diff_descNn_sim, orig[i][1]))
        return word_char_vec_desc_diff_descNn_sim_orig

    def batch_iterator(self, word_char_vec_desc_vecDiff_descNn_sim_orig, max_batch_size, cuda=False, shuffle=False, mode='train', ignore_index=-100, seed_feeding=True, ignore_duplicates=False):
        """
        mode: {train (use whole data; volatile=False) | valid (use whole data; volatile=True) | test (don't use duplicate data; volatile=True)}
        """
        data = []
        if ignore_duplicates:
            words = set()
            for i in range(len(word_char_vec_desc_vecDiff_descNn_sim_orig)):
                word = word_char_vec_desc_vecDiff_descNn_sim_orig[i][0].split('%', 1)[0]
                if word not in words:
                    data.append(word_char_vec_desc_vecDiff_descNn_sim_orig[i])
                    words.add(word)
        else:
            data = word_char_vec_desc_vecDiff_descNn_sim_orig

        if shuffle:
            random.shuffle(data)

        batch_num = -(-len(data) // max_batch_size)
        batch_size = max_batch_size
        for i in range(batch_num):
            # for the last remainder batch, set the batch size smaller than others
            if i == batch_num - 1 and len(data) % max_batch_size != 0:
                batch_size = len(data) % max_batch_size

            batch = data[i*max_batch_size:(i+1)*max_batch_size]
            words = []
            chars = torch.FloatTensor(batch_size, self.max_char_len, len(self.id2char)).zero_()
            vecs = torch.FloatTensor(batch_size, len(data[0][2]))
            srcs_T = torch.LongTensor(batch_size, self.max_len)

            # tensors for EDIT models
            descs_nn_T = torch.LongTensor(batch_size * self.topk, self.max_len - 1).fill_(self.word2id['<eos>']) # self.max_len includes <bos> but descs_nn does not
            descs_nn_copy_T = torch.LongTensor(batch_size * self.topk, self.max_len - 1).fill_(self.word2id['<eos>'])  # assign on-the-fly ids instead of <unk> ids
            descs_nn_mask_T = torch.FloatTensor(batch_size * self.topk, self.max_len - 1).fill_(-float('inf'))
            vecs_diff = torch.FloatTensor(batch_size * self.topk, len(data[0][2]))
            sims = torch.FloatTensor(batch_size * self.topk)


            # assign on-the-fly ids to unks in descs_nn_copy_T
            id2unk_batch = []
            unk2id_batch = []
            for b, (_, _, _, _, diff_descNn_sim, _) in enumerate(batch):
                unk2id = {}  # ids are consistent in k definitions
                id2unk = {}
                for k, (_, description_nn, _, description_orig, _) in enumerate(diff_descNn_sim):
                    for i in range(len(description_nn)):
                        descs_nn_T[b * self.topk + k, i] = description_nn[i]
                        descs_nn_mask_T[b * self.topk + k, i] = 0

                        word = description_orig[i]
                        if description_nn[i] == self.word2id['<unk>']:
                            if word not in unk2id:
                                unk2id[word] = len(unk2id) + len(self.word2id)
                                id2unk[len(unk2id) - 1 + len(self.word2id)] = word # len(unk2id) has just changed
                            descs_nn_copy_T[b * self.topk + k, i] = unk2id[word]
                        else:
                            descs_nn_copy_T[b * self.topk + k, i] = description_nn[i]
                id2unk_batch.append(id2unk)
                unk2id_batch.append(unk2id)


            if seed_feeding:
                trgs_T = torch.LongTensor(batch_size, self.max_len + 1) # src is 1-element less than trg because we will feed Seed vector later
            else:
                trgs_T = torch.LongTensor(batch_size, self.max_len)

            descs_nn_orig = []
            for b, (word, char, vec, description, diff_descNn_sim, description_orig) in enumerate(batch):
                words.append(word)
                vecs[b] = torch.FloatTensor(vec)
                # one-hot repl. of words
                for k, c in enumerate(char):
                    chars[b, k, c] = 1.0

                # pad the source id sequence with <eos>
                padded_desc = description[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description[:-1]))
                srcs_T[b] = torch.Tensor(padded_desc)

                # for target side, replace <unk> ids in description to extended oov_ids if possible
                description_with_unk_ids = []
                # print('debug! description', description)
                # print('debug! self.word2id', self.word2id)
                # print('debug! description_orig', description_orig)
                # print('debug! unk2id_batch', unk2id_batch)
                # if len(description) != len(description_orig):
                #     print('hoge')

                for k in range(len(description)):
                    if description[k] == self.word2id['<unk>'] and description_orig[k] in unk2id_batch[b]:
                        description_with_unk_ids.append(unk2id_batch[b][description_orig[k]])
                    else:
                        description_with_unk_ids.append(description[k])

                if seed_feeding:
                    shifted_desc = [ignore_index] + description_with_unk_ids[1:] + [ignore_index] * (self.max_len - len(description_with_unk_ids[:-1]))
                else:
                    shifted_desc = description_with_unk_ids[1:] + [ignore_index] * (self.max_len - len(description_with_unk_ids[:-1]))
                trgs_T[b] = torch.Tensor(shifted_desc)

                for k, (vec_diff, description_nn, sim, description_nn_orig, oov_num) in enumerate(diff_descNn_sim):
                    vecs_diff[b * self.topk + k] = torch.FloatTensor(vec_diff)
                    sims[b * self.topk + k] = sim
                    descs_nn_orig.append(description_nn_orig)

            # reshape the tensors
            srcs = torch.transpose(srcs_T, 0, 1).contiguous()
            trgs = torch.transpose(trgs_T, 0, 1).contiguous().view(-1)
            descs_nn = torch.transpose(descs_nn_T, 0, 1).contiguous()
            descs_nn_copy = torch.transpose(descs_nn_copy_T, 0, 1).contiguous()
            descs_mask_nn = torch.transpose(descs_nn_mask_T, 0, 1).contiguous()

            if not cuda:
                yield (words, Variable(chars.unsqueeze(1)), Variable(vecs),
                       Variable(srcs, volatile=False if mode =='train' else True),
                       Variable(trgs), Variable(vecs_diff),
                       Variable(descs_nn, volatile=False if mode =='train' else True),
                       Variable(descs_mask_nn, volatile=False if mode == 'train' else True),
                       Variable(descs_nn_copy, volatile=False if mode == 'train' else True),
                       id2unk_batch,
                       Variable(sims))
            else:
                yield (words, Variable(chars.unsqueeze(1)).cuda(), Variable(vecs).cuda(),
                       Variable(srcs, volatile=False if mode =='train' else True).cuda(),
                       Variable(trgs).cuda(), Variable(vecs_diff).cuda(),
                       Variable(descs_nn, volatile=False if mode =='train' else True).cuda(),
                       Variable(descs_mask_nn, volatile=False if mode == 'train' else True).cuda(),
                       Variable(descs_nn_copy, volatile=False if mode == 'train' else True).cuda(),
                       id2unk_batch,
                       Variable(sims).cuda())

class EditCorpus4(NorasetCorpus):
    '''corpus for edit model w/ Cao's copying [Cao+ AAAI17]'''
    def __init__(self, data_path, vec_path, max_vocab_size=-1, mode='train', topk=-1, min_sim=0, ignore_sense_id=False, train_src='both', raw_nns=False):
        super(EditCorpus4, self).__init__(data_path, vec_path, max_vocab_size, ignore_sense_id=ignore_sense_id, train_src=train_src)
        self.topk = topk
        sys.stderr.write('Reading nearest neighbor files...')
        if mode == 'train':
            self.train_word2nns = self.read_nns(os.path.join(data_path, 'train.nn'), topk, min_sim)
            self.valid_word2nns = self.read_nns(os.path.join(data_path, 'valid.nn'), topk, min_sim)
            self.train = self.add_nns(self.train, self.train_word2nns, 'train_multi_encoder')
            self.valid = self.add_nns(self.valid, self.valid_word2nns, 'test')
            self.train = self.add_orig(self.train, self.train_orig)
            self.valid = self.add_orig(self.valid, self.valid_orig)
            self.train = self.add_copyable_labels(self.train)
            self.valid = self.add_copyable_labels(self.valid)
        self.test_word2nns= self.read_nns(os.path.join(data_path, 'test.nn'), topk, min_sim)
        self.test = self.add_nns(self.test, self.test_word2nns, 'test', raw_nns=raw_nns)
        self.test = self.add_orig(self.test, self.test_orig)
        self.test = self.add_copyable_labels(self.test)
        sys.stderr.write('  Done\n')

    def add_copyable_labels(self, word_char_vec_desc_diff_descNn_sim_orig):
        word_char_vec_desc_diff_descNn_sim_orig_copyable = []
        for (word, char, vec, desc, diff_descNn_sim, orig) in word_char_vec_desc_diff_descNn_sim_orig:
            nn_words = [diff_descNn_sim[k][3] for k in range(len(diff_descNn_sim))]
            copyable_words = set()
            for words in nn_words:
                copyable_words |= set(words)
            copyable = [int(orig[i] in copyable_words) for i in range(len(orig))]
            word_char_vec_desc_diff_descNn_sim_orig_copyable.append((word, char, vec, desc, diff_descNn_sim, orig, copyable))
        return word_char_vec_desc_diff_descNn_sim_orig_copyable

    def read_nns(self, path, topk, min_sim):
        assert os.path.exists(path)
        word2nn_sim_nnDef_diff = {} # {'information': [('data', 0.54, ['facts', 'and', 'statistics', ...], [Vinformation - Vdata]), ...}

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                word, nn, similarity, desc_nn, diff = line.strip().split('\t')
                similarity = float(similarity)
                diff = [float(val) for val in diff.split()]

                if word not in word2nn_sim_nnDef_diff:
                    word2nn_sim_nnDef_diff[word] = []
                if (topk == -1 or topk > len(word2nn_sim_nnDef_diff[word])) and (min_sim <= similarity):
                    word2nn_sim_nnDef_diff[word].append((nn, similarity, desc_nn, diff))

        return word2nn_sim_nnDef_diff

    def add_nns(self, word_char_vec_desc, word2nns, mode, raw_nns=False):
        word_char_vec_desc_diff_descNn_sim = []
        for word, char, vec, desc in word_char_vec_desc:
            diff_descNn_sim = []
            if word not in word2nns:
                continue
            for (nn, similarity, desc_nn, diff) in word2nns[word]: # for each nn
                oov_num = 0
                # convert desc_nn to ids
                desc_nn = desc_nn.split()
                if raw_nns: # do not covert nns into ids; only used in Translator.top1_copy()
                    desc_nn_id = desc_nn
                else:
                    desc_nn_id = []
                    for w in desc_nn:
                        if w in self.word2id:
                            desc_nn_id.append(self.word2id[w])
                        else:
                            desc_nn_id.append(self.word2id['<unk>'])
                            oov_num += 1
                diff_descNn_sim.append((diff, desc_nn_id, similarity, desc_nn, oov_num))
            word_char_vec_desc_diff_descNn_sim.append((word, char, vec, desc, diff_descNn_sim))

        return word_char_vec_desc_diff_descNn_sim

    def add_orig(self, word_char_vec_desc_diff_descNn_sim, orig):
        word_char_vec_desc_diff_descNn_sim_orig = []
        for i, (word, char, vec, desc, diff_descNn_sim) in enumerate(word_char_vec_desc_diff_descNn_sim):
            word_char_vec_desc_diff_descNn_sim_orig.append((word, char, vec, desc, diff_descNn_sim, orig[i][1]))
        return word_char_vec_desc_diff_descNn_sim_orig

    def batch_iterator(self, word_char_vec_desc_vecDiff_descNn_sim_orig, max_batch_size, cuda=False, shuffle=False, mode='train', ignore_index=-100, seed_feeding=True, ignore_duplicates=False):
        """
        mode: {train (use whole data; volatile=False) | valid (use whole data; volatile=True) | test (don't use duplicate data; volatile=True)}
        """
        data = []
        if ignore_duplicates:
            words = set()
            for i in range(len(word_char_vec_desc_vecDiff_descNn_sim_orig)):
                word = word_char_vec_desc_vecDiff_descNn_sim_orig[i][0].split('%', 1)[0]
                if word not in words:
                    data.append(word_char_vec_desc_vecDiff_descNn_sim_orig[i])
                    words.add(word)
        else:
            data = word_char_vec_desc_vecDiff_descNn_sim_orig

        if shuffle:
            random.shuffle(data)

        batch_num = -(-len(data) // max_batch_size)
        batch_size = max_batch_size
        for i in range(batch_num):
            # for the last remainder batch, set the batch size smaller than others
            if i == batch_num - 1 and len(data) % max_batch_size != 0:
                batch_size = len(data) % max_batch_size

            batch = data[i*max_batch_size:(i+1)*max_batch_size]
            words = []
            chars = torch.FloatTensor(batch_size, self.max_char_len, len(self.id2char)).zero_()
            vecs = torch.FloatTensor(batch_size, len(data[0][2]))
            srcs_T = torch.LongTensor(batch_size, self.max_len)

            # tensors for EDIT models
            descs_nn_T = torch.LongTensor(batch_size * self.topk, self.max_len - 1).fill_(self.word2id['<eos>']) # self.max_len includes <bos> but descs_nn does not
            descs_nn_copy_T = torch.LongTensor(batch_size * self.topk, self.max_len - 1).fill_(self.word2id['<eos>'])  # assign on-the-fly ids instead of <unk> ids
            descs_nn_mask_T = torch.FloatTensor(batch_size * self.topk, self.max_len - 1).fill_(-float('inf'))
            vecs_diff = torch.FloatTensor(batch_size * self.topk, len(data[0][2]))
            sims = torch.FloatTensor(batch_size * self.topk)


            # assign on-the-fly ids to unks in descs_nn_copy_T
            id2unk_batch = []
            unk2id_batch = []
            for b, (_, _, _, _, diff_descNn_sim, _, _) in enumerate(batch):
                unk2id = {}  # ids are consistent in k definitions
                id2unk = {}
                for k, (_, description_nn, _, description_orig, _) in enumerate(diff_descNn_sim):
                    for i in range(len(description_nn)):
                        descs_nn_T[b * self.topk + k, i] = description_nn[i]
                        descs_nn_mask_T[b * self.topk + k, i] = 0

                        word = description_orig[i]
                        if description_nn[i] == self.word2id['<unk>']:
                            if word not in unk2id:
                                unk2id[word] = len(unk2id) + len(self.word2id)
                                id2unk[len(unk2id) - 1 + len(self.word2id)] = word # len(unk2id) has just changed
                            descs_nn_copy_T[b * self.topk + k, i] = unk2id[word]
                        else:
                            descs_nn_copy_T[b * self.topk + k, i] = description_nn[i]
                id2unk_batch.append(id2unk)
                unk2id_batch.append(unk2id)


            if seed_feeding:
                trgs_T = torch.LongTensor(batch_size, self.max_len + 1) # src is 1-element less than trg because we will feed Seed vector later
                cpys_T = torch.LongTensor(batch_size, self.max_len + 1).fill_(ignore_index)
            else:
                trgs_T = torch.LongTensor(batch_size, self.max_len)
                cpys_T = torch.LongTensor(batch_size, self.max_len).fill_(ignore_index)

            descs_nn_orig = []
            for b, (word, char, vec, description, diff_descNn_sim, description_orig, copyable) in enumerate(batch):
                words.append(word)
                vecs[b] = torch.FloatTensor(vec)
                # one-hot repl. of words
                for k, c in enumerate(char):
                    chars[b, k, c] = 1.0

                # pad the source id sequence with <eos>
                padded_desc = description[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description[:-1]))
                srcs_T[b] = torch.Tensor(padded_desc)

                # for target side, replace <unk> ids in description to extended oov_ids if possible
                description_with_unk_ids = []
                for k in range(len(description)):
                    if description[k] == self.word2id['<unk>'] and description_orig[k] in unk2id_batch[b]:
                        description_with_unk_ids.append(unk2id_batch[b][description_orig[k]])
                    else:
                        description_with_unk_ids.append(description[k])

                if seed_feeding:
                    shifted_desc = [ignore_index] + description_with_unk_ids[1:] + [ignore_index] * (self.max_len - len(description_with_unk_ids[:-1]))
                else:
                    shifted_desc = description_with_unk_ids[1:] + [ignore_index] * (self.max_len - len(description_with_unk_ids[:-1]))
                trgs_T[b] = torch.Tensor(shifted_desc)
                cpys_T[b][1:len(copyable)] = torch.LongTensor(copyable[1:])

                for k, (vec_diff, description_nn, sim, description_nn_orig, oov_num) in enumerate(diff_descNn_sim):
                    vecs_diff[b * self.topk + k] = torch.FloatTensor(vec_diff)
                    sims[b * self.topk + k] = sim
                    descs_nn_orig.append(description_nn_orig)

            # reshape the tensors
            srcs = torch.transpose(srcs_T, 0, 1).contiguous()
            trgs = torch.transpose(trgs_T, 0, 1).contiguous().view(-1)
            cpys = torch.transpose(cpys_T, 0, 1).contiguous().view(-1)
            descs_nn = torch.transpose(descs_nn_T, 0, 1).contiguous()
            descs_nn_copy = torch.transpose(descs_nn_copy_T, 0, 1).contiguous()
            descs_mask_nn = torch.transpose(descs_nn_mask_T, 0, 1).contiguous()

            if not cuda:
                yield (words, Variable(chars.unsqueeze(1)), Variable(vecs),
                       Variable(srcs, volatile=False if mode =='train' else True),
                       Variable(trgs),
                       Variable(cpys),
                       Variable(vecs_diff),
                       Variable(descs_nn, volatile=False if mode =='train' else True),
                       Variable(descs_mask_nn, volatile=False if mode == 'train' else True),
                       Variable(descs_nn_copy, volatile=False if mode == 'train' else True),
                       id2unk_batch,
                       Variable(sims))
            else:
                yield (words, Variable(chars.unsqueeze(1)).cuda(), Variable(vecs).cuda(),
                       Variable(srcs, volatile=False if mode =='train' else True).cuda(),
                       Variable(trgs).cuda(),
                       Variable(cpys).cuda(),
                       Variable(vecs_diff).cuda(),
                       Variable(descs_nn, volatile=False if mode =='train' else True).cuda(),
                       Variable(descs_mask_nn, volatile=False if mode == 'train' else True).cuda(),
                       Variable(descs_nn_copy, volatile=False if mode == 'train' else True).cuda(),
                       id2unk_batch,
                       Variable(sims).cuda())



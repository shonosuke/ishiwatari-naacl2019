import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Filter vectors from word2vec files')
parser.add_argument('vec1', type=str, help='location of the word2vec file')
parser.add_argument('vec2', type=str, help='location of the word2vec file (output of senseRetrofit)')
parser.add_argument('data', type=str, help='directory that contains {train | valid | test}.txt')
parser.add_argument('output', type=str, help='output vector file')
args = parser.parse_args()

words = set()
def read_vector(self, path, words):
    """Reads word2vec file."""
    assert os.path.exists(path)
    word2vec = {} # {srdWord0: [dim0, dim1, ..., dim299], srcWord1: [dim0, dim1, ..., dim299]}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            elems = line.strip().split(' ')
            if i == 0 and len(elems) == 2: # The first line in word2vec file
                continue
            word = elems[0]
            if word in words or word == '<unk>':
                word2vec[word] = [float(val) for val in elems[1:]]

    if len(word2vec) == 0:
        print('cannot read word2vec file. check file format below:')
        print('word2vec file:', word)
        print('vocab file:', list(words.keys())[:5])
        exit()

    if '<unk>' not in word2vec:
        word2vec['<unk>'] = np.random.uniform(-0.05, 0.05, len(word2vec[list(word2vec.keys())[0]])).tolist()

    return word2vec

for file in ['train.txt', 'valid.txt', 'test.txt']:
    path = os.path.join(args.data, file)
    with open(path) as f:
        for line in f:
            word, _, _, definition, _, _ = line.strip().split('\t')
            words.add(word)
            words.add(word.split('%', 1)[0])
            words |= set(definition.split())

with open(args.output, 'w') as fout:
    for path in [args.vec1, args.vec2]:
        with open(path) as fin:
            for line in fin:
                if line.split(' ', 1)[0] in words:
                    fout.write(line)
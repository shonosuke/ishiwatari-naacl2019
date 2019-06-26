# inputs: {train|valid|test}.json
# outputs: {train|valid|test}.{txt|eg}, filtered.vec
# python3 extract_oxford_dic.py --data /disk/ishiwatari/defgen/oxford_dic/orig --vec /disk/ishiwatari/defgen/vec/GoogleNews-vectors-negative300.txt --out /disk/ishiwatari/defgen/oxford_dic

import argparse, sys, json, os, random
import numpy as np

def read_json(path):
    word_set = set()
    data = {'train': None, 'valid': None, 'test': None}
    for filename in ['train', 'valid', 'test']:
        with open(path + '/' + filename + '.json') as f:
            word2desc_egs = {}  # e.g., word2desc_egs[word] = [(desc1, eg1), (desc2, eg2), ...]
            d = json.load(f)
            for [word], desc, example in d:
                if word not in word2desc_egs:
                    word2desc_egs[word] = []
                word2desc_egs[word].append((desc, example))
                word_set |= set([word] + desc + example)

            data[filename] = word2desc_egs
    return data, word_set

def read_vector(path, words, read_as_string=True):
    """Reads word2vec file."""
    assert os.path.exists(path)
    word2vec = {} # {srdWord0: [dim0, dim1, ..., dim299], srcWord1: [dim0, dim1, ..., dim299]}
    words_lemma = set()
    for word in words:
        words_lemma.add(word.split('%', 1)[0])
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if read_as_string:
                if i == 0 and len(line.strip().split()) == 2: # The first line in word2vec file
                    continue
                word, vec = line.strip().split(' ', 1)
                if word in words or word == '<unk>' or word in words_lemma:
                    word2vec[word] = vec
            else:
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
        unk_vec = np.random.uniform(-0.05, 0.05, 300).tolist()
        if read_as_string:
            word2vec['<unk>'] = ' '.join(['%.6f' % (val) for val in unk_vec])
        else:
            word2vec['<unk>'] = unk_vec

    return word2vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing script to convert data from Gadetskys .json to our .txt & .eg format')
    parser.add_argument('--data', type=str, default='/Users/ishiwatari/Dropbox/PhD/data/Gadetsky_ACL2018/orig', help='Path to Gadetsky\'s dataset')
    parser.add_argument('--vec', type=str, default='/Users/ishiwatari/Dropbox/PhD/src/PycharmProjects/defgen_pt/data/GoogleNews-vectors-negative300.txt.filtered', help='word2vec file')
    parser.add_argument('--out', type=str, default='/Users/ishiwatari/Dropbox/PhD/src/PycharmProjects/defgen_pt/data/onto_nora/mini/gadetsky', help='directory to output files (*.txt, *.eg)')
    args = parser.parse_args()
    random.seed('masaru')

    sys.stderr.write('Reading files in ' + args.data + '...\n')
    data, word_set = read_json(args.data)

    sys.stderr.write('Reading ' + args.vec + '...\n')
    word2vec = read_vector(args.vec, word_set)

    with open(args.out + '/filtered.vec', 'w') as fp:
        for word, line in word2vec.items():
            fp.write(word + ' ' + line + '\n')
    sys.stderr.write('Wrote to ' + args.out + '/filtered.vec\n')

    # output
    for filename in ['train', 'valid', 'test']:
        word2descs = {}
        word_desc2id = {}
        with open(args.out + '/' + filename + '.txt', 'w') as fp_txt:
            with open(args.out + '/' + filename + '.eg', 'w') as fp_eg:
                for word in data[filename]:
                    if word not in word2descs:
                        word2descs[word] = set()
                    for (desc, eg) in data[filename][word]:
                        desc_joined = ' '.join(desc)
                        if desc_joined not in word2descs[word]:
                            word_desc2id[(word, desc_joined)] = str(len(word2descs[word]))
                            word2descs[word].add(desc_joined)

                        word_with_id = word + '%oxford.' + word_desc2id[(word, desc_joined)]
                        line_txt = '\t'.join([word_with_id, 'pos', 'oxford', desc_joined, '[]', '[]'])
                        line_eg = '\t'.join([word_with_id, ' '.join(eg)])
                        fp_txt.write(line_txt + '\n')
                        fp_eg.write(line_eg + '\n')

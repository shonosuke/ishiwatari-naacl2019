# input: merged.jsonlines, pretrained w2v file
# outputs: {train|valid|test}.{txt|eg}, filtered.vec
# python3 extract_wikidata.py --data /disk/ishiwatari/defgen/wikidata/merged.jsonlines --vec /disk/ishiwatari/defgen/vec/GoogleNews-vectors-negative300.txt --out /disk/ishiwatari/defgen/wikidata

import argparse, sys, json, os, random
import numpy as np
from collections import OrderedDict

def read_and_split_json_lines(path):
    res = OrderedDict()
    with open(path) as f:
        for line in f:
            j = json.loads(line)

            # Do some preprocessings here:
            # 1. Lower titles and descs
            j['title'] = j['title'].lower()
            j['desc'] = j['desc'].lower()

            # 2. Lower and split examples
            texts_split = []
            for paragraph in j['text']:
                texts_split.append([])
                for sentence in paragraph:
                    texts_split[-1].append(sentence.lower().split())
            j['text'] = texts_split

            res[j['qid']] = j
    return res

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

def convert_vecs_str2np(word2vec_str):
    words = word2vec_str.keys()
    word2vec_np = {}
    for word in words:
        word2vec_np[word] = np.array([float(val) for val in word2vec_str[word].split()])
    return word2vec_np

def compose_vecs(word2vec, words):
    added = np.array(np.zeros(300))
    for word in words:
        if word not in word2vec:
            added = added + word2vec['<unk>']
        else:
            added = added + word2vec[word]
    return added

def divide_data(word_list, valid_ratio=0.05, test_ratio=0.05):
    val_test = random.sample(range(len(word_list)), round(len(word_list) * (valid_ratio + test_ratio)))
    val_ids =  set(val_test[:len(val_test)//2])
    test_ids = set(val_test[len(val_test)//2:])
    train_words, val_words, test_words = set(), set(), set()
    for i, word in enumerate(word_list):
        if i in val_ids:
            val_words.add(word_list[i])
        elif i in test_ids:
            test_words.add(word_list[i])
        else:
            train_words.add(word_list[i])
    return train_words, val_words, test_words

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing script to extract all (including train/valid/test) data and example sentences from merged.jsonlinse')
    parser.add_argument('--data', type=str, default='/Users/ishiwatari/Dropbox/PhD/src/PycharmProjects/defgen_pt/data/merged.jsonlines', help='Shoetsu\'s WikiP2D dataset')
    parser.add_argument('--vec', type=str, default='/Users/ishiwatari/Dropbox/PhD/src/PycharmProjects/defgen_pt/data/GoogleNews-vectors-negative300.txt', help='word2vec file')
    parser.add_argument('--out', type=str, default='/Users/ishiwatari/Dropbox/PhD/src/PycharmProjects/defgen_pt/data/onto_nora/mini/wikidata', help='directory to output files (*.txt, *.eg)')
    parser.add_argument('--max_eg', type=int, default=30, help='max length of examples')
    parser.add_argument('--max_def', type=int, default=30, help='max length of definitions')
    args = parser.parse_args()
    random.seed('masaru')

    # read all files
    sys.stderr.write('Reading ' + args.data +  '...\n')
    data = read_and_split_json_lines(args.data)
    phrase2desc_egs = {} # e.g., word2desc_examples[word] = [(desc1, eg1), (desc2, eg2), ...]
    word_set = set()
    for qid in data:
        for qid_linked in data[qid]['link']:
            if qid_linked in data and data[qid_linked]['desc'] != '':

                # extract a phrase that is linked to another entity
                id_para, id_sent, [pos_begin, pos_end] = data[qid]['link'][qid_linked]
                sentence = data[qid]['text'][id_para][id_sent]
                phrase_linked = '_'.join(sentence[pos_begin:pos_end + 1])

                # extract a title from linked qid
                title = data[qid_linked]['title']
                title_pos_end = title.find('_(')
                if title.find('_(') != -1: # remove the appended (~~~)
                    title = title[:title_pos_end]

                # check if the phrase and title are the same or not
                if phrase_linked == title:
                    example = ' '.join(sentence[:pos_begin] + ['<TRG>'] + sentence[pos_end+1:])
                    desc = data[qid_linked]['desc']
                    if phrase_linked not in phrase2desc_egs:
                        phrase2desc_egs[phrase_linked] = []
                    if desc.count(' ') < args.max_def and example.count(' ') <= args.max_eg:
                        phrase2desc_egs[phrase_linked].append((desc, example))
                        word_set |= set(sentence)

    # read vecs
    sys.stderr.write('Reading ' + args.vec + '...\n')
    word2vec = read_vector(args.vec, word_set)

    # compose phrase vecs
    sys.stderr.write('Composing phrase vectors...\n')
    word2vec = convert_vecs_str2np(word2vec)
    for phrase in phrase2desc_egs:
        if phrase.find('_') != -1:
            words = phrase.split('_')
            word2vec[phrase] = compose_vecs(word2vec, words)

    # write composed vecs
    with open(args.out + '/composed.vec', 'w') as fp:
        for word, vec in word2vec.items():
            line = ' '.join(['%.6f' % (val) for val in vec.tolist()])
            fp.write(word + ' ' + line + '\n')
    sys.stderr.write('Wrote to ' + args.out + '/composed.vec\n')

    # divide data
    sys.stderr.write('Dividing data...\n')
    train, valid, test = divide_data(sorted(list(phrase2desc_egs.keys())))

    # output
    for phrase_subset, filename in zip([train, valid, test], ['train', 'valid', 'test']):
        phrase2descs = {}
        phrase_desc2id = {}
        with open(args.out + '/' + filename + '.txt', 'w') as fp_txt:
            with open(args.out + '/' + filename + '.eg', 'w') as fp_eg:
                for phrase in phrase_subset:
                    if phrase not in phrase2descs:
                        phrase2descs[phrase] = set()
                    for (desc, eg) in phrase2desc_egs[phrase]:
                        if desc not in phrase2descs[phrase]:
                            phrase_desc2id[(phrase, desc)] = str(len(phrase2descs[phrase]))
                            phrase2descs[phrase].add(desc)

                        phrase_with_id = phrase + '%wiki.' + phrase_desc2id[(phrase, desc)]
                        line_txt = '\t'.join([phrase_with_id, 'pos', 'wikidata', desc, '[]', '[]'])
                        line_eg = '\t'.join([phrase_with_id, eg])
                        fp_txt.write(line_txt + '\n')
                        fp_eg.write(line_eg + '\n')

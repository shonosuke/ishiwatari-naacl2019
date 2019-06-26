import argparse
import os
import sys
import regex
import random

parser = argparse.ArgumentParser(description='Extract descriptions from Wordnet database')
parser.add_argument('--data', type=str, default='../data/wordnet',
                    help='location of the data corpus')
parser.add_argument('--dict', type=str, default='../data/wordnet/dict',
                    help='location of wordnet directory')
parser.add_argument('--vec_orig', type=str, default='../data/GoogleNews-vectors-negative300.txt',
                    help='location of the original word2vec data')
parser.add_argument('--vec_retro', type=str, default='../data/GoogleNews-vectors-negative300.txt.sense',
                    help='location of the retrofit word2vec data')
parser.add_argument('--allow_single_offset_split', action='store_true',
                    help='allow the words have same offset to be split to different data')
args = parser.parse_args()
random.seed('masaru')

def def_prepro(definition, pos, lemma=None):
    # remove self references
    if lemma and lemma in definition.split(' '):
        return None

    # delete parenthesis and all the words inside them
    pattern = r"(?<rec>\((?:[^()]+|(?&rec))*\))"
    buf = regex.sub(pattern, '', definition, flags=regex.VERBOSE)

    # remove words after ;
    buf = buf.split(';', 1)[0]

    # if verb, prepend "to"
    if pos == 'v':
        buf = 'to ' + buf

    # lowercasing
    buf = buf.lower()

    # split ','
    buf = buf.replace(', ', ' , ')

    return buf

def divide_data(word_list, valid_ratio=0.05, test_ratio=0.05):
    val_test = random.sample(range(len(word_list)), round(len(word_list) * (valid_ratio + test_ratio)))
    val_ids =  set(val_test[:len(val_test)//2])
    test_ids = set(val_test[len(val_test)//2:])
    train_words, val_words, test_words = set(), set(), set()
    for i, word in enumerate(word_list):
        if i in val_ids:
            val_words.add(i)
        elif i in test_ids:
            test_words.add(i)
        else:
            train_words.add(i)
    return train_words, val_words, test_words

def get_offsets_from_offset(offset, offset2lemmas, lemma2offsets, used_offsets):
    """return offsets that have same lemma with either of the lemma of input offset"""
    new_offsets = set()
    for lemma in offset2lemmas[offset]:
        for new_offset in lemma2offsets[lemma]:
            if new_offset not in used_offsets:
                new_offsets.add(new_offset)
    return new_offsets

def pack_offsets(offset2lemmas, lemma2offsets):
    """pack all the offsets who have common lemmas together"""
    used_offsets = set()
    offset_sets = []
    for offset in sorted(offset2lemmas.keys()):
        if offset not in used_offsets:
            offset_sets.append(set([offset]))
            old_offsets = set([offset])
            used_offsets.add(offset)
            while True:
                new_offsets = set()
                for old_offset in old_offsets:
                    new_offsets |= get_offsets_from_offset(old_offset, offset2lemmas, lemma2offsets, used_offsets)
                if len(new_offsets) > 0:
                    used_offsets |= new_offsets
                    offset_sets[-1] |= (new_offsets)
                    old_offsets = new_offsets
                else:
                    break
        if len(used_offsets) == len(offset2lemmas):
            break
    return offset_sets


sys.stderr.write('Reading data.{adj | adv | verb | noun}...   ')
sys.stderr.flush()
offset2pos_def = {}
words = set() # words in definitions
for file in ['data.adj', 'data.adv', 'data.verb', 'data.noun']:
    with open(os.path.join(args.dict, file), 'r') as f:
        for line in f:
            if line[0] == ' ':
                continue
            else:
                offset, _, pos = line.split(' ', 3)[:3]
                definition = def_prepro(line.split('|', 1)[1].strip(), pos)
                if definition != None and definition != '':
                    offset2pos_def[offset] = (pos, definition)
                    for word in definition.strip().split():
                        if word != '':
                            words.add(word)
sys.stderr.write('Done.\n')


sys.stderr.write('Reading data.{adj | adv | verb | noun}...   ')
sys.stderr.flush()
offset2pos_def = {}
words = set() # words in definitions
for file in ['data.adj', 'data.adv', 'data.verb', 'data.noun']:
    with open(os.path.join(args.dict, file), 'r') as f:
        for line in f:
            if line[0] == ' ':
                continue
            else:
                offset, _, pos = line.split(' ', 3)[:3]
                definition = def_prepro(line.split('|', 1)[1].strip(), pos)
                if definition != None and definition != '':
                    offset2pos_def[offset] = (pos, definition)
                    for word in definition.strip().split():
                        if word != '':
                            words.add(word)
sys.stderr.write('Done.\n')


sys.stderr.write('Reading index.sense...   ')
sys.stderr.flush()
senseId2offset = {}
offset2senseIds = {}
with open(os.path.join(args.dict, 'index.sense'), 'r') as f:
    for line in f:
        # e.g., acidify%2:39:00:: 02201136 1 0
        sense_id, offset = line.strip().split(' ', 2)[:2]
        if offset in offset2pos_def:
            senseId2offset[sense_id] = offset
            if offset not in offset2senseIds:
                offset2senseIds[offset] = []
            offset2senseIds[offset].append(sense_id)
sys.stderr.write('Done.\n')


sys.stderr.write('Filtering retro-fitted word vectors...   ')
sys.stderr.flush()
lemma2senseId_pos_def_vec = {}
with open(args.vec_retro, 'r') as f_in:
    for i, line in enumerate(f_in):
        if i == 0:
            continue
        sense_id, vec = line.split(' ', 1)
        if sense_id in senseId2offset:
            lemma = sense_id.split('%', 1)[0]
            pos, definition = offset2pos_def[senseId2offset[sense_id]]
            if lemma not in lemma2senseId_pos_def_vec:
                lemma2senseId_pos_def_vec[lemma] = []
            lemma2senseId_pos_def_vec[lemma].append((sense_id, pos, definition, line))
sys.stderr.write('Done.\n')


sys.stderr.write('Filtering original word vectors...   ')
sys.stderr.flush()
offset2senseIds_filtered = {} # all data that can be used for experiments (including train/valid/test)
lemma2offsets = {}
offset2lemmas = {}
with open(args.vec_retro + '.filtered', 'w') as f_out1:
    f_out1.write('hoge hoge\n')  # TODO: fix this
    with open(args.vec_orig + '.wn.filtered', 'w') as f_out2:
        f_out2.write('hoge hoge\n')  # TODO: fix this
        with open(args.vec_orig, 'r') as f_in:
            for i, line in enumerate(f_in):
                if i == 0:
                    continue
                lemma, vec = line.split(' ', 1)
                # leave the lemma only if it exists in both original vecs and retro-fitted vecs
                if lemma in lemma2senseId_pos_def_vec:
                    for senseId_pos_def_vec in lemma2senseId_pos_def_vec[lemma]:
                        f_out1.write(senseId_pos_def_vec[3])
                    f_out2.write(line)

                    for (sense_id, pos, definition, line) in lemma2senseId_pos_def_vec[lemma]:
                        offset = senseId2offset[sense_id]
                        if offset not in offset2senseIds_filtered:
                            offset2senseIds_filtered[offset] = []
                        offset2senseIds_filtered[offset].append(sense_id)

                        if lemma not in lemma2offsets:
                            lemma2offsets[lemma] = []
                        lemma2offsets[lemma].append(offset)

                        if offset not in offset2lemmas:
                            offset2lemmas[offset] = []
                        offset2lemmas[offset].append(lemma)

                elif lemma in words: # only used to initialization of word embedding layer
                    f_out1.write(line)
                    f_out2.write(line)
sys.stderr.write('Done.\n')


sys.stderr.write('Dividing definitions into {train | valid | test}.txt ...   ')
sys.stderr.flush()
if not args.allow_single_offset_split:  # get [set(), set(), ...]
    # TODO: packed_data are very unbalanced; len(packed_data[0]) == 30926. should check whether output val data or test data are too large or not
    packed_data = pack_offsets(offset2lemmas, lemma2offsets)
else: # get [(lemma1, [list]), (lemma2, [list], ...)]
    packed_data =  [(lemma, lemma2offsets[lemma]) for lemma in sorted(lemma2offsets)]

train, val, test = divide_data(packed_data)
with open(os.path.join(args.data, 'train.txt'), 'w') as f_train:
    with open(os.path.join(args.data, 'valid.txt'), 'w') as f_val:
        with open(os.path.join(args.data, 'test.txt'), 'w') as f_test:
            for i, elems in enumerate(packed_data):
                if not args.allow_single_offset_split:
                    offsets = elems
                else:
                    lemma, offsets = elems

                for offset in sorted(offsets):
                    pos, definition = offset2pos_def[offset]
                    sense_ids = offset2senseIds_filtered[offset]
                    lines = ''
                    for sense_id in offset2senseIds_filtered[offset]:
                        if (not args.allow_single_offset_split) or (args.allow_single_offset_split and sense_id.split('%',1)[0] == lemma):
                            lines += '\t'.join([sense_id, pos, 'wordnet', definition]) + '\n'
                    if i in train:
                        f_train.write(lines)
                    elif i in val:
                        f_val.write(lines)
                    else:
                        f_test.write(lines)

sys.stderr.write('Done.\n')

from subprocess import Popen, PIPE
import os
import sys
import random
import argparse
import json

def if_known_words(word, known_words):
    words = word.split('%', 1)[0].split('_')
    for word in words:
        if word not in known_words:
            return False
    return True

def read_definition_file(ifp, args, word_with_sense=None, read_known_words_only=None):
    defs = {}
    for line in ifp:
        parts = line.strip().split('\t')
        word = parts[0]
        if args.ignore_sense_id:
            word = word.split('%', 1)[0]

        if read_known_words_only == None or if_known_words(word, read_known_words_only):
            if args.phrase_length == None or len(word.split('_')) == args.phrase_length:
                if len(parts) > 1:
                    definition = parts[-1]
                else:
                    definition = ''
                if word not in defs:
                    defs[word] = []
                if len(defs[word]) < 10: # to avoid the amount of examples being too large (especially when using Wikipedia dataset)
                    defs[word].append(definition)

    # if the test results ignores sense id
    if (not args.ignore_sense_id) and (word.find('%') == -1) and  (word_with_sense != None):
        defs_with_sense_id = {}
        for word in word_with_sense:
            word_wo_sense_id = word.split('%', 1)[0]
            defs_with_sense_id[word] = defs[word_wo_sense_id]
        defs = defs_with_sense_id
    return defs

def get_bleu_score(bleu_path, all_ref_paths, d, hyp_path):
    with open(hyp_path, 'w') as ofp:
        ofp.write(d + '\n')
    read_cmd = ['cat', hyp_path]
    bleu_cmd = [bleu_path] + all_ref_paths
    rp = Popen(read_cmd, stdout=PIPE)
    bp = Popen(bleu_cmd, stdin=rp.stdout, stdout=PIPE, stderr=devnull)
    out, err = bp.communicate()
    if err is None:
        return float(out.strip())
    else:
        print ("get_bleu_score err")
        return None

def unk_ratio(words, known_words):
    known, unk = 0, 0
    for word in words:
        if word in known_words:
            known += 1
        else:
            unk += 1
    return unk / (known + unk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BLEU computation of [Noraset 17]')

    # Model parameters.
    parser.add_argument('--tmp_dir', type=str, default='/tmp',
                        help='tmp directory')
    parser.add_argument('--bleu_path', type=str, default='./sentence-bleu',
                        help='tmp directory')
    parser.add_argument('--mode', type=str, default='average',
                        help='{average|random}')
    parser.add_argument('--ignore_sense_id', action='store_true', help='ignore sense id. for noraset models')
    parser.add_argument('--ref', type=str, help='test.txt')
    parser.add_argument('--hyp', type=str, help='test.out')
    parser.add_argument('--stats', action='store_true', help='stats of UNK words in phrases to be defined')
    parser.add_argument('--json', type=str, help='test.json')
    parser.add_argument('--vec', type=str, default='/disk/ishiwatari/defgen/vec/GoogleNews-vectors-negative300.txt',
                        help='used to calculate the stats for unk words')
    parser.add_argument('--ignore_unk_words', action='store_true', help='ignore UNK words in phrases when calculate statistics')
    parser.add_argument('--phrase_length', type=int, default=None, help='Limit the phrase length to this number. (Ignore all phrases that are longer or shorter than this length)')

    args = parser.parse_args()
    suffix = str(random.random())

    if args.stats:
        known_words = []
        for line in open(args.vec):
            word = line.split(' ', 1)[0]
            known_words.append(word)
        known_words = set(known_words)

    # Read data
    refs, hyps = None, None
    with open(args.ref) as ifp:
        refs = read_definition_file(ifp, args, read_known_words_only=known_words if args.ignore_unk_words else None)
    with open(args.hyp) as ifp:
        hyps = read_definition_file(ifp, args, word_with_sense=refs.keys(), read_known_words_only=known_words if args.ignore_unk_words else None)

    if not args.ignore_sense_id:
        word2senseIds = {}
        for word in list(refs.keys()):
            word_wo_id, sense_id = word.split('%', 1)
            if word_wo_id not in word2senseIds:
                word2senseIds[word_wo_id] = set()
            word2senseIds[word_wo_id].add(sense_id)

    # Check words
    if len(refs) != len(hyps):
        print("Number of words being defined mismatched!")
    words = refs.keys()

    hyp_path = os.path.join(args.tmp_dir, 'hyp' + suffix)
    to_be_deleted = set()
    to_be_deleted.add(hyp_path)

    # Computing BLEU
    devnull = open(os.devnull, 'w')
    score = 0
    count = 0
    total_refs = 0
    total_hyps = 0
    score_nwords = {} # sum(score) only for the target words whose length == 4
    count_nwords = {}
    unkRatio_bleu_len_word_ref_hyp = [] # [(unk_ratio, bleu, len_out, word, ref, hyp), (), .., word, wrefs, whyps.]
    numDef2bleus = {} # number of definition to bleus
    for word in words:
        if word not in refs or word not in hyps:
            continue
        wrefs = refs[word]
        whyps = hyps[word]
        # write out references
        all_ref_paths = []
        for i, d in enumerate(wrefs):
            ref_path = os.path.join(args.tmp_dir, 'ref' + suffix + str(i))
            with open(ref_path, 'w') as ofp:
                ofp.write(d)
                all_ref_paths.append(ref_path)
                to_be_deleted.add(ref_path)
        total_refs += len(all_ref_paths)
        # score for each output
        micro_score = 0
        micro_count = 0
        if args.mode == "average":
            for d in whyps:
                rhscore = get_bleu_score(args.bleu_path, all_ref_paths, d, hyp_path)
                # print (word + '\t' + str(rhscore) + '\t' + d)
                if rhscore is not None:
                    micro_score += rhscore
                    micro_count += 1
        elif args.mode == "random":
            d = random.choice(whyps)
            rhscore = get_bleu_score(args.bleu_path, all_ref_paths, d, hyp_path)
            if rhscore is not None:
                micro_score += rhscore
                micro_count += 1
        total_hyps += micro_count
        score += micro_score / micro_count
        count += 1

        # stats of length
        word_split = word.split('%', 1)[0].split('_')
        length = len(word_split)
        if length not in score_nwords:
            score_nwords[length] = 0
            count_nwords[length] = 0
        score_nwords[length] += micro_score / micro_count
        count_nwords[length] += 1

        if args.stats:
            # stats of #unks
            unk_rat = unk_ratio(word_split, known_words)
            output_lens = [len(output.split()) for output in whyps]
            unkRatio_bleu_len_word_ref_hyp.append((unk_rat, micro_score / micro_count, sum(output_lens) / len(output_lens), word, wrefs, whyps))


            if not args.ignore_sense_id:
                # stats of #definitions
                num_def = len(word2senseIds[word.split('%', 1)[0]])
                if num_def not in numDef2bleus:
                    numDef2bleus[num_def] = []
                numDef2bleus[num_def].append(micro_score / micro_count)
    devnull.close()

    print('average BLEU:', score/count)
    print('---------------------------------')
    # stats of length
    for length in sorted(list(score_nwords.keys())):
        print('BLEU (length of target phrase =', length, '):', score_nwords[length], '/', count_nwords[length], '=', score_nwords[length]/ count_nwords[length])

    if args.stats:
        print('---------------------------------')
        # stats of #unks
        ratio2bleu = [[], [], [], [], [], [], [], [], [], [], [], []]  # (ratio = 0), (0.0 < ratio <= 0.1), (0.1 < ratio <= 0.2), ... (0.9 < ratio <= 1.0)
        ratio2len_out = [[], [], [], [], [], [], [], [], [], [], [], []]
        ratio2word_ref_hyp = [[], [], [], [], [], [], [], [], [], [], [], []]

        for (unkRatio, bleu, out_len, word, ref, hyp) in unkRatio_bleu_len_word_ref_hyp:
            bin = int(unkRatio * 10)
            ratio2bleu[bin].append(bleu)
            ratio2len_out[bin].append(out_len)
            ratio2word_ref_hyp[bin].append((word, ref, hyp))

        for i, bin in enumerate(ratio2bleu):
            if len(bin) > 0:
                print(0.1 * i, '<= unk_ratio <', 0.1 * (i + 1), '\tBLEU:\t', sum(bin),  '/', len(bin), '=', sum(bin) / len(bin), '\tlen(output):\t', sum(ratio2len_out[i]) / len(ratio2len_out[i]))

        # stats of #definitions
        if not args.ignore_sense_id:
            print('---------------------------------')
            for num_def in sorted(list(numDef2bleus.keys())):
                print('#definition = ', num_def, '\tBLEU:\t', sum(numDef2bleus[num_def]), '/', len(numDef2bleus[num_def]),
                      '=', sum(numDef2bleus[num_def]) / len(numDef2bleus[num_def]))

        print('---------------------------------')
        for i, bin in enumerate(ratio2word_ref_hyp):
            if len(bin) > 0:
                word_len, ref_len, hyp_len = [], [], []
                for word, ref, hyp in ratio2word_ref_hyp[i]:
                    word_len.append(len(word.split('_')))
                    for sent in ref:
                        ref_len.append(len(sent.split()))
                    for sent in hyp:
                        hyp_len.append(len(sent.split()))

                print('-----------------', 0.1 * i, '<= unk_ratio <', 0.1 * (i + 1), '-----------------')
                print('--- Len(phrase)', sum(word_len)/len(word_len), '---')
                print('--- Len(ref)', sum(ref_len) / len(ref_len), '---')
                print('--- Len(hyp)', sum(hyp_len) / len(hyp_len), '---')
                for word, ref, hyp in ratio2word_ref_hyp[i]:
                    print(word, ref, hyp)

        # dump json file
        ratio2word_ref_hyp_dic = {}
        for i, bin in enumerate(ratio2word_ref_hyp):
            ratio2word_ref_hyp_dic[i] = bin
        f = open(args.json, 'w')
        json.dump(ratio2word_ref_hyp_dic, f)


    # delete tmp files
    for f in to_be_deleted:
        os.remove(f)

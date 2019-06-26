import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import random
import os
from subprocess import Popen, PIPE
from nltk.translate import bleu_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from util import cudaw


class Translator(object):
    def __init__(self, corpus, sentence_bleu=None, valid_all=False, mode='train'):
        self.corpus = corpus
        self.test = corpus.test
        self.ref= {"test": corpus.ref_test}

        if mode == 'train':
            self.valid = corpus.valid
            self.ref['valid'] = corpus.ref_valid

        if sentence_bleu:
            tmp_dir = "/tmp"
            suffix = str(random.random())
            self.hyp_path = os.path.join(tmp_dir, 'hyp' + suffix)
            self.ref_path = os.path.join(tmp_dir, 'ref' + suffix)
            self.to_be_deleted = set()
            self.to_be_deleted.add(self.hyp_path)
            self.bleu_path = sentence_bleu

        random.seed('masaru')
        if mode == 'train':
            if not valid_all:
                self.random_samples = random.sample(range(min(len(self.valid), len(self.test))), 5)
            else:
                self.random_samples = list(range(len(self.valid)))

    def bleu(self, hyp, mode, nltk=None):
        """mode can be 'valid' or 'test' """
        score = 0
        num_hyp = 0
        if nltk == 'corpus':
            refs = []
        with open(os.devnull, 'w') as devnull:
            for (word, desc) in hyp:
                word_wo_id = word.split('%', 1)[0]
                # compute sentence bleu
                if nltk == 'sentence': #  3~5 point lower than sentence_bleu.cpp
                    if len(desc) == 0:
                        auto_reweigh = False
                    else:
                        auto_reweigh = True
                    bleu = bleu_score.sentence_bleu([ref.split() for ref in self.ref[mode][word_wo_id]], desc,
                                                    smoothing_function=bleu_score.SmoothingFunction().method2,
                                                    auto_reweigh=auto_reweigh)
                    score += bleu
                    num_hyp += 1

                elif nltk == 'corpus':
                    refs.append([ref.split() for ref in self.ref[mode][word_wo_id]])

                else:
                    # write refs to tmp files
                    ref_paths = []
                    for i, ref in enumerate(self.ref[mode][word_wo_id][:30]):
                        ref_path = self.ref_path + str(i)
                        with open(ref_path, 'w') as f:
                            f.write(ref + '\n')
                            ref_paths.append(ref_path)
                            self.to_be_deleted.add(ref_path)

                    # write a hyp to tmp file
                    with open(self.hyp_path, 'w') as f:
                        f.write(' '.join(desc) + '\n')

                    rp = Popen(['cat', self.hyp_path], stdout=PIPE)
                    bp = Popen([self.bleu_path] + ref_paths, stdin=rp.stdout, stdout=PIPE, stderr=devnull)
                    out, err = bp.communicate()
                    bleu = float(out.strip())
                    score += bleu
                    num_hyp += 1

        if nltk == 'corpus':
            bleu = bleu_score.corpus_bleu(refs, [word_desc[1] for word_desc in hyp],
                                          smoothing_function=bleu_score.SmoothingFunction().method2,
                                          emulate_multibleu=True)
            return bleu

        # delete tmp files
        if not nltk:
            for f in self.to_be_deleted:
                os.remove(f)

        return score / num_hyp

    def top1_copy(self, mode="valid", ignore_duplicates=False):
        if mode == "valid":
            data = self.valid
        elif mode == "test":
            data = self.test

        if ignore_duplicates:
            data_new = []
            words = set()
            for i in range(len(data)):
                word = data[i][0].split('%', 1)[0]
                if word not in words:
                    data_new.append(data[i])
                    words.add(word)
            data = data_new

        top1_copied = [(data[i][0], data[i][4][0][1]) for i in range(len(data))]
        return top1_copied

    def draw_att_weights(self, att_mat_np, word, hyp, nns, path):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(att_mat_np, cmap='bone', vmin=0, vmax=1)
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + hyp, rotation=90, fontdict={'size': 8})
        ax.set_yticklabels([''] + nns, fontdict={'size': 8})

        # Show label at every tick
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xlabel('Generated definition')
        ax.set_ylabel('Retrieved definitions')

        # plt.show()
        plt.savefig(path + '/' + word + '.pdf', bbox_inches='tight')
        return

    def visualize_att_weights(self, att_weights_batches, mode, topk, results, path):

        if mode == "valid":
            data = self.valid
        elif mode == "test":
            data = self.test
        # remove duplicates
        data_new = []
        words = set()
        for i in range(len(data)):
            word = data[i][0].split('%', 1)[0]
            if word not in words:
                data_new.append(data[i])
                words.add(word)
        data = data_new

        i = 0
        for batch in att_weights_batches:
            max_src_len = batch.size(2) / topk
            for att_mat in batch:
                word, hyp = results[i]
                hyp_buf = hyp + ['<eos>']
                for j in range(len(hyp_buf)):
                    if hyp_buf[j] not in self.corpus.word2id:
                        hyp_buf[j] = '[' + hyp_buf[j] + ']'
                nns = []
                sliced_att_mat = []
                for k in range(topk):
                    nn = data[i][4][k][3]
                    nns.append(nn)
                    sliced_att_mat.append(att_mat[:(len(hyp_buf)), int(max_src_len * k): int(max_src_len * k + len(nn))])
                att_mat_np = torch.cat(sliced_att_mat, dim=1).permute(1, 0).cpu().data.numpy()
                nns_concat = []
                for nn in nns:
                    for w in nn:
                        if w not in self.corpus.word2id:
                            nns_concat.append('[' + w + ']')
                        else:
                            nns_concat.append(w)
                self.draw_att_weights(att_mat_np, word, hyp_buf, nns_concat, path)
                i += 1
        return 0

    def greedy(self, model, mode="valid", max_batch_size=128, cuda=True, p=1, max_len=60, ignore_duplicates=False, return_att_weights=False):
        if mode == "valid":
            data = self.valid
        elif mode == "test":
            data = self.test
        if model.name == "EDIT": # To perform batch ensembling
            max_batch_size = 1

        results = []
        att_weights_batches = [] # [batch_num, (batch_size, trgLen, srcLen*k)]
        batch_iter = self.corpus.batch_iterator(data, max_batch_size, cuda=cuda, mode=mode, seed_feeding=model.seed_feeding, ignore_duplicates=ignore_duplicates)
        for i, elems in enumerate(batch_iter):
            batch_size = len(elems[0])
            if model.name == "EDIT":
                decoded_words = [[]]
            else:
                decoded_words = [[] for x in range(batch_size)]

            hidden = model.init_hidden(batch_size, model.dhid)
            char_emb = None
            if model.CH:
                char_emb = model.get_char_embedding(elems[1])

            input_word = Variable(torch.LongTensor([[self.corpus.word2id['<bos>']] * batch_size]), volatile=True)
            if cuda:
                input_word = input_word.cuda()

            att_weights_batch = [] # trgLen, (batch_size, srcLen*k)

            # Decode the first word
            if model.name == "EDIT":
                (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, sim) = elems
                nn_emb = model.get_nn_embedding(desc_nn)
                Wh_enc = model.attention.map_enc_states(nn_emb)
                output, hidden = model(input_word, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda,
                                       seed_feeding=model.seed_feeding, char_emb=char_emb, batch_ensemble=True, p=p)
                max_ids = output[-1].max(-1)[1]  # there may be two outputs if we use seed. Here we want the newest one
                input_word[0].data.fill_(max_ids.data[0])
                keep_decoding = [1] * 1
            elif model.name == "EDIT_COPY":
                (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, sim) = elems
                nn_emb = model.get_nn_embedding(desc_nn)
                Wh_enc = model.attention.map_enc_states(nn_emb)
                sim = model.get_nn_sim(vec, vec_diff)
                nn_1hot = model.get_nn_1hot(desc_nn, cuda)
                output, hidden = model(input_word, hidden, vec, vec_diff, nn_1hot, nn_emb, desc_nn_mask, Wh_enc, sim,
                                       cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, p=p)
                max_ids = output[-1].max(-1)[1]  # there may be two outputs if we use seed. Here we want the newest one
                input_word[0] = max_ids
                keep_decoding = [1] * batch_size
            elif model.name in set(["EDIT_MULTI", "EDIT_MULTI_OUTPUT", "EDIT_DECODER", "EDIT_TRIPLE", "EDIT_HIERARCHICAL", "EDIT_HIERARCHICAL_DIFF", "EDIT_HIRO", "EDIT_COPY"]):
                (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, sim) = elems
                nn_emb = model.get_nn_embedding(desc_nn)
                Wh_enc = model.attention.map_enc_states(nn_emb)
                if model.name in set(["EDIT_TRIPLE", "EDIT_HIERARCHICAL_DIFF", "EDIT_COPY"]):
                    sim = model.get_nn_sim(vec, vec_diff)
                returns = model(input_word, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda,
                                       seed_feeding=model.seed_feeding, char_emb=char_emb, batch_ensemble=False, p=p, return_att_weights=return_att_weights)
                if return_att_weights:
                    output, hidden, att_weight = returns
                    att_weights_batch.append(att_weight[-1]) # there would be two att_weights if we use seed. Here we want the newest one
                else:
                    output, hidden = returns

                max_ids = output[-1].max(-1)[1]  # there may be two outputs if we use seed. Here we want the newest one
                input_word[0] = max_ids
                keep_decoding = [1] * batch_size
            elif model.name in {'EDIT_COPYNET', "EDIT_CAO", 'EDIT_CAO_SIMPLE'}:
                if model.name == "EDIT_COPYNET":
                    (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, desc_nn_copy, id2unk, sim) = elems
                else:
                    (word, chars, vec, src, trg_def, trg_cpy, vec_diff, desc_nn, desc_nn_mask, desc_nn_copy, id2unk, sim) = elems
                nn_emb = model.get_nn_embedding(desc_nn)
                # Wh_enc = model.attention.map_enc_states(nn_emb)
                # TODO! commentout ↑ and uncommentout ↓
                att_input = torch.cat([nn_emb, vec_diff.unsqueeze(0).repeat(nn_emb.size(0), 1, 1), torch.abs(vec_diff).unsqueeze(0).repeat(nn_emb.size(0), 1, 1)], dim=2)  # (srcLen, k*b, 4d)
                Wh_enc = model.attention.map_enc_states(att_input)  # (topk*batch, len, dim)

                mapper_cpy_to_cmb = model.get_mapper(desc_nn, desc_nn_copy, cuda)
                returns = model(input_word, hidden, vec, vec_diff, nn_emb, desc_nn_mask, mapper_cpy_to_cmb, Wh_enc, sim,
                                       cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, p=p, return_att_weights=return_att_weights)
                if return_att_weights:
                    if model.name == "EDIT_COPYNET":
                        output, hidden, att_weight = returns
                    else:
                        output, output_cpy, hidden, att_weight = returns
                    att_weights_batch.append(att_weight[-1]) # there would be two att_weights if we use seed. Here we want the newest one
                else:
                    if model.name == "EDIT_COPYNET":
                        output, hidden = returns
                    else:
                        output, output_cpy, hidden = returns

                max_ids = output[-1].max(-1)[1]  # there may be two outputs if we use seed. Here we want the newest one
                keep_decoding = [1] * batch_size
                for k in range(batch_size):
                    word_id = int(max_ids[k].data[0])
                    if word_id >= len(self.corpus.id2word):  # oov word
                        decoded_words[k].append(id2unk[k][word_id])
                        input_word[0][k].data.fill_(self.corpus.word2id['<unk>'])
                    elif self.corpus.id2word[word_id] != '<eos>':  # in-vocab word
                        decoded_words[k].append(self.corpus.id2word[word_id])
                        input_word[0][k].data.fill_(word_id)
                    else:  # <eos>
                        input_word[0][k].data.fill_(self.corpus.word2id['<eos>'])
                        keep_decoding[k] = 0

            elif model.name[:2] == "EG":
                (word, chars, vec, src, trg, eg, eg_mask) = elems
                eg_emb = model.get_nn_embedding(eg)
                Wh_enc = model.attention.map_enc_states(eg_emb)  # (topk*batch, len, dim)
                output, hidden = model(input_word, hidden, vec, eg_emb, eg_mask, Wh_enc, cuda,
                                       seed_feeding=model.seed_feeding, char_emb=char_emb, p=p)
                max_ids = output[-1].max(-1)[1]  # there may be two outputs if we use seed. Here we want the newest one
                input_word[0] = max_ids
                keep_decoding = [1] * batch_size

            else:
                (word, chars, vec, src, trg) = elems
                output, hidden = model(input_word, hidden, vec, cuda, seed_feeding=model.seed_feeding, char_emb=char_emb)
                max_ids = output[-1].max(-1)[1]  # there are two outputs if we use seed. Here we want the newest one
                input_word[0] = max_ids
                keep_decoding = [1] * batch_size  # assign 0 once <eos> has appeared

            if model.name not in {'EDIT_COPYNET', "EDIT_CAO", "EDIT_CAO_SIMPLE"}:
                for k in range(len(keep_decoding)):
                    word_id = max_ids[k].data[0]
                    if self.corpus.id2word[word_id] != '<eos>':
                        decoded_words[k].append(self.corpus.id2word[word_id])
                    else:
                        keep_decoding[k] = 0

            # decode the subsequent words batch by batch
            for j in range(max_len):
                if model.name == "EDIT":
                    output, hidden = model(input_word, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda,
                                           seed_feeding=False, char_emb=char_emb, batch_ensemble=True, p=p)
                elif model.name[:2] == "EG":
                    output, hidden = model(input_word, hidden, vec, eg_emb, eg_mask, Wh_enc, cuda,
                                           seed_feeding=False, char_emb=char_emb, p=p)

                elif model.name == "EDIT_COPY":
                    output, hidden = model(input_word, hidden, vec, vec_diff, nn_1hot, nn_emb, desc_nn_mask, Wh_enc, sim,
                                           cuda, seed_feeding=False, char_emb=char_emb, p=p)
                elif model.name in set(["EDIT_MULTI", "EDIT_MULTI_OUTPUT", "EDIT_DECODER", "EDIT_TRIPLE", "EDIT_HIERARCHICAL", "EDIT_HIERARCHICAL_DIFF", "EDIT_HIRO"]):
                    returns = model(input_word, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda,
                                                     seed_feeding=False, char_emb=char_emb,
                                                     batch_ensemble=False, p=p, return_att_weights=return_att_weights)
                    if return_att_weights:
                        output, hidden, att_weight = returns
                        att_weights_batch.append(att_weight[-1])  # there would be two att_weights if we use seed. Here we want the newest one
                    else:
                        output, hidden = returns
                elif model.name in {'EDIT_COPYNET', "EDIT_CAO", "EDIT_CAO_SIMPLE"}:
                    returns = model(input_word, hidden, vec, vec_diff, nn_emb, desc_nn_mask, mapper_cpy_to_cmb, Wh_enc, sim,
                                    cuda, seed_feeding=False, char_emb=char_emb, p=p, return_att_weights=return_att_weights)
                    if return_att_weights:
                        if model.name == "EDIT_COPYNET":
                            output, hidden, att_weight = returns
                        else:
                            output, output_cpy, hidden, att_weight = returns
                        att_weights_batch.append(att_weight[-1])  # there would be two att_weights if we use seed. Here we want the newest one
                    else:
                        if model.name == "EDIT_COPYNET":
                            output, hidden = returns
                        else:
                            output, output_cpy, hidden = returns

                    max_ids = output[0].max(-1)[1]  # torch.LongTensor of size (batch_size)
                    for k in range(len(keep_decoding)):
                        if keep_decoding[k]:
                            word_id = int(max_ids[k].data[0])
                            if word_id >= len(self.corpus.id2word): # oov word
                                decoded_words[k].append(id2unk[k][word_id])
                                input_word[0][k].data.fill_(self.corpus.word2id['<unk>'])
                            elif self.corpus.id2word[word_id] != '<eos>': # in-vocab word
                                decoded_words[k].append(self.corpus.id2word[word_id])
                                input_word[0][k].data.fill_(word_id)
                            else: # <eos>
                                keep_decoding[k] = 0
                                input_word[0][k].data.fill_(self.corpus.word2id['<eos>'])

                else:
                    output, hidden = model(input_word, hidden, vec, cuda,
                                           seed_feeding=False, char_emb=char_emb) # no seed feeding for second and the later words

                # map id to word
                if model.name not in {'EDIT_COPYNET', "EDIT_CAO", "EDIT_CAO_SIMPLE"}:
                    max_ids = output[0].max(-1)[1]  # torch.LongTensor of size (batch_size)
                    for k in range(len(keep_decoding)):
                        word_id = max_ids[k].data[0]
                        if keep_decoding[k]:
                            if self.corpus.id2word[word_id] != '<eos>':
                                decoded_words[k].append(self.corpus.id2word[word_id])
                            else:
                                keep_decoding[k] = 0
                        else:
                            pass
                    # feedback to the next step
                    if model.name == "EDIT":
                        input_word[0].data.fill_(max_ids.data[0])
                    else:
                        input_word[0] = max_ids
                if max(keep_decoding) == 0:
                    break

            for k in range(len(decoded_words)):
                results.append((word[k], decoded_words[k]))

            if return_att_weights:
                att_weights_batch_reshape = [w.unsqueeze(1) for w in att_weights_batch]
                att_weights_batch_reshape = torch.cat(att_weights_batch_reshape, dim=1) # (b, trgLen, srcLen*k)
                att_weights_batches.append(att_weights_batch_reshape)

        if not ignore_duplicates: # random_samples may be out of index if ignore duplicate data
            print('Decoded ' + mode + ' samples:')
            for sample in self.random_samples:
                sentence = ''
                for word in results[sample][1]:
                    if word not in self.corpus.word2id:
                        sentence += ' [' + word + ']'
                    else:
                        sentence += ' ' + word
                print(results[sample][0] + '\t' + sentence)

        if return_att_weights:
            return results, att_weights_batches
        else:
            return results

    def eval_log_loss(self, model, mode="valid", max_batch_size=128, cuda=True, p=1, return_logloss_matrix=False):
        if mode == 'valid':
            ntoken = self.corpus.valid_ntoken
        elif mode == 'test':
            ntoken = self.corpus.test_ntoken

        # Turn on evaluation mode which disables dropout.
        model.eval()
        if model.name in {"EDIT", "EDIT_DECODER", "EDIT_COPY", "EDIT_COPYNET"}:
            criterion = nn.NLLLoss(size_average=False)
        elif model.name in {'EDIT_CAO', "EDIT_CAO_SIMPLE"}:
            criterion_def = nn.NLLLoss(size_average=False)
            criterion_cpy = nn.NLLLoss(size_average=False)
        else:
            # criterion = nn.CrossEntropyLoss(size_average=False)
            criterion = nn.NLLLoss(size_average=False)

        if return_logloss_matrix:
            logloss_mats = []

        total_loss = 0
        total_loss_gen = 0
        total_loss_cpy = 0
        batch_iter = self.corpus.batch_iterator(self.valid if mode == 'valid' else self.test, max_batch_size, cuda=cuda, mode='valid', seed_feeding=model.seed_feeding)
        for i, elems in enumerate(batch_iter):
            hidden = model.init_hidden(len(elems[0]), model.dhid)

            char_emb = None
            if model.CH:
                char_emb = model.get_char_embedding(elems[1])

            if model.name == "EDIT_COPY":
                (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, sim) = elems
                nn_emb = model.get_nn_embedding(desc_nn)
                Wh_enc = model.attention.map_enc_states(nn_emb)
                sim = model.get_nn_sim(vec, vec_diff)
                nn_1hot = model.get_nn_1hot(desc_nn, cuda)
                output, hidden = model(src, hidden, vec, vec_diff, nn_1hot, nn_emb, desc_nn_mask, Wh_enc, sim,
                                       cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, p=p)

            elif model.name in {"EDIT_COPYNET", "EDIT_COPYNET_SIMPLE"}:
                (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, desc_nn_copy, id2unk, sim) = elems
                nn_emb = model.get_nn_embedding(desc_nn)
                # Wh_enc = model.attention.map_enc_states(nn_emb)
                # TODO! commentout ↑ and uncommentout ↓
                att_input = torch.cat([nn_emb, vec_diff.unsqueeze(0).repeat(nn_emb.size(0), 1, 1), torch.abs(vec_diff).unsqueeze(0).repeat(nn_emb.size(0), 1, 1)], dim=2)  # (srcLen, k*b, 4d)
                Wh_enc = model.attention.map_enc_states(att_input)  # (topk*batch, len, dim)

                mapper_cpy_to_cmb = model.get_mapper(desc_nn, desc_nn_copy, cuda)
                output, hidden = model(src, hidden, vec, vec_diff, nn_emb, desc_nn_mask, mapper_cpy_to_cmb, Wh_enc, sim,
                                       cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, p=p)

            elif model.name in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
                (word, chars, vec, src, trg, trg_cpy, vec_diff, desc_nn, desc_nn_mask, desc_nn_copy, id2unk,
                 sim) = elems
                nn_emb = model.get_nn_embedding(desc_nn)
                # Wh_enc = model.attention.map_enc_states(nn_emb)  # (topk*batch, len, dim)
                att_input = torch.cat([nn_emb, vec_diff.unsqueeze(0).repeat(nn_emb.size(0), 1, 1), torch.abs(vec_diff).unsqueeze(0).repeat(nn_emb.size(0), 1, 1)], dim=2)  # (srcLen, k*b, 4d)
                Wh_enc = model.attention.map_enc_states(att_input)  # (topk*batch, len, dim)
                mapper_cpy_to_cmb = model.get_mapper(desc_nn, desc_nn_copy, cuda)
                output, output_cpy, hidden = model(src, hidden, vec, vec_diff, nn_emb, desc_nn_mask,
                                                       mapper_cpy_to_cmb, Wh_enc, sim, cuda,
                                                       seed_feeding=model.seed_feeding, char_emb=char_emb, p=p)
                loss_gen = criterion_def(output.view(output.size(0) * output.size(1), -1), trg).data
                loss_cpy = criterion_cpy(output_cpy.view(output_cpy.size(0) * output_cpy.size(1), -1), trg_cpy).data
                total_loss += loss_gen
                total_loss_cpy += loss_cpy

            elif model.name[:4] == "EDIT":
                batch_ensemble = True if model.name == "EDIT" else False
                (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, sim) = elems
                nn_emb = model.get_nn_embedding(desc_nn)
                Wh_enc = model.attention.map_enc_states(nn_emb)
                if model.name in set(["EDIT_TRIPLE", "EDIT_HIERARCHICAL_DIFF"]):
                    sim = model.get_nn_sim(vec, vec_diff)
                output, hidden = model(src, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda,
                                       seed_feeding=model.seed_feeding, char_emb=char_emb,
                                       batch_ensemble=batch_ensemble, p=p)

            elif model.name[:2] == "EG":
                (word, chars, vec, src, trg, eg, eg_mask) = elems
                eg_emb = model.get_nn_embedding(eg)
                Wh_enc = model.attention.map_enc_states(eg_emb)  # (topk*batch, len, dim)
                output, hidden = model(src, hidden, vec, eg_emb, eg_mask, Wh_enc, cuda,
                                       seed_feeding=model.seed_feeding, char_emb=char_emb, p=p)

            else:
                (word, chars, vec, src, trg) = elems
                output, hidden = model(src, hidden, vec, cuda, seed_feeding=model.seed_feeding, char_emb=char_emb)

            if model.name not in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
                output_flat = output.view(output.size(0) * output.size(1), -1) # (trgLen, vocab)
                total_loss += criterion(output_flat, trg).data

            if return_logloss_matrix:
                if model.name[:2] == "EG":
                    output = F.log_softmax(output, dim=2)

                trg_ids = trg.view(output.size(0), -1) # (trgLen, b)
                trg_logloss = torch.FloatTensor(trg_ids.size()).fill_(0)
                for j in range(trg_ids.size(0)): # loop over target sentence
                    for b in range(trg_ids.size(1)): # loop over batch
                        if trg_ids[j, b].data[0] != -100:
                            trg_logloss[j, b] = output[j, b, trg_ids[j, b].data[0]].data[0]

                logloss_mats.append(trg_logloss)

        returns = [total_loss[0] / ntoken]
        if model.name in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
            returns.append(total_loss_cpy[0] / ntoken)
        if return_logloss_matrix:
            logloss_matrix = torch.cat(logloss_mats, dim=1).permute(1, 0) # (dataSize, trgLen)
            returns.append(logloss_matrix)

        return returns

    def print_log_loss(self, word, logloss, ref):
        # logloss: (trgLen), ref: [w1, w2, ..., <eos>]
        print(word + ':', end='')
        ref_buf = []
        for w in ref:
            if w not in self.corpus.word2id:
                ref_buf.append('[' + w + ']')
            else:
                ref_buf.append(w)
            print('\t{:>6s}'.format(ref_buf[-1]), end='')
        print('\n' + ' ' * (len(word) + 1), end='')
        for j, loss in enumerate(logloss):
            format = '\t{:>' + str(max(len(ref_buf[j]), 6)) + '.2f}'
            print(format.format(loss), end='')
        print()

    def print_log_loss_matrix(self, logloss_matrix, mode):
        # logloss_matrix (dataSize, trgLen)
        ref = [(self.corpus.test[k][0], self.corpus.test[k][5]) for k in range(len(self.corpus.test))]

        # logloss_matrix: (dataSize, trgLen)
        print('\n' + '-' * 150)
        for i, data in enumerate(logloss_matrix):
            words = ref[i][1][1:]
            logloss = data[1:len(ref[i][1])].tolist()
            self.print_log_loss(ref[i][0], logloss, words)
            print('-' * 150)

import argparse
import time
import math
from builtins import enumerate

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import data
import model
import translate

# import pdb; pdb.set_trace()
def load_checkpoint():
    checkpoint = torch.load(args.save)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def save_checkpoint():
    check_point = {'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
    torch.save(check_point, args.save)

###############################################################################
# Training code
###############################################################################
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_loss_gen = 0
    total_loss_cpy = 0

    batch_iter = corpus.batch_iterator(corpus.train, args.batch_size, cuda=args.cuda, shuffle=True, mode='train', seed_feeding=model.seed_feeding)
    batch_num = -(-len(corpus.train) // args.batch_size)
    start_time = time.time()
    for i, elems in enumerate(batch_iter):
        optimizer.zero_grad()

        hidden = model.init_hidden(len(elems[0]), model.dhid)
        if args.char:
            char_emb = model.get_char_embedding(elems[1])
        else:
            char_emb = None

        if args.mode == "EDIT_COPY":
            (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, sim) = elems
            nn_emb = model.get_nn_embedding(desc_nn)
            Wh_enc = model.attention.map_enc_states(nn_emb)
            sim = model.get_nn_sim(vec, vec_diff)
            nn_1hot = model.get_nn_1hot(desc_nn, args.cuda)
            output, hidden = model(src, hidden, vec, vec_diff, nn_1hot, nn_emb, desc_nn_mask, Wh_enc, sim, args.cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, p=args.p)

        elif args.mode in {"EDIT_COPYNET", "EDIT_COPYNET_SIMPLE"}:
            (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, desc_nn_copy, id2unk, sim) = elems
            nn_emb = model.get_nn_embedding(desc_nn)
            att_input = torch.cat([nn_emb, vec_diff.unsqueeze(0).repeat(nn_emb.size(0), 1, 1), torch.abs(vec_diff).unsqueeze(0).repeat(nn_emb.size(0), 1, 1)], dim=2) # (srcLen, k*b, 4d)
            Wh_enc = model.attention.map_enc_states(att_input) # (topk*batch, len, dim)
            mapper_cpy_to_cmb = model.get_mapper(desc_nn, desc_nn_copy, args.cuda)
            output, hidden = model(src, hidden, vec, vec_diff, nn_emb, desc_nn_mask, mapper_cpy_to_cmb, Wh_enc, sim, args.cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, p=args.p)

        elif args.mode in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
            (word, chars, vec, src, trg_gen, trg_cpy, vec_diff, desc_nn, desc_nn_mask, desc_nn_copy, id2unk, sim) = elems
            nn_emb = model.get_nn_embedding(desc_nn)
            att_input = torch.cat([nn_emb, vec_diff.unsqueeze(0).repeat(nn_emb.size(0), 1, 1), torch.abs(vec_diff).unsqueeze(0).repeat(nn_emb.size(0), 1, 1)], dim=2)  # (srcLen, k*b, 4d)
            Wh_enc = model.attention.map_enc_states(att_input)  # (topk*batch, len, dim)
            # Wh_enc = model.attention.map_enc_states(nn_emb) # (topk*batch, len, dim)
            mapper_cpy_to_cmb = model.get_mapper(desc_nn, desc_nn_copy, args.cuda)
            output_gen, output_cpy, hidden = model(src, hidden, vec, vec_diff, nn_emb, desc_nn_mask, mapper_cpy_to_cmb, Wh_enc, sim, args.cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, p=args.p)
            loss_gen = criterion_def(output_gen.view(output_gen.size(0) * output_gen.size(1), -1), trg_gen)
            loss_cpy = criterion_cpy(output_cpy.view(output_cpy.size(0) * output_cpy.size(1), -1), trg_cpy)
            loss = loss_gen + loss_cpy

        elif args.mode[:4] == "EDIT":
            (word, chars, vec, src, trg, vec_diff, desc_nn, desc_nn_mask, sim) = elems
            nn_emb = model.get_nn_embedding(desc_nn)
            Wh_enc = model.attention.map_enc_states(nn_emb) # (topk*batch, len, dim)
            if args.mode in set(["EDIT_TRIPLE", "EDIT_HIERARCHICAL_DIFF"]):
                sim = model.get_nn_sim(vec, vec_diff) # (b, k)
            output, hidden = model(src, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, args.cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, p=args.p)

        elif args.mode[:2] == "EG":
            (word, chars, vec, src, trg, eg, eg_mask) = elems
            eg_emb = model.get_nn_embedding(eg)
            Wh_enc = model.attention.map_enc_states(eg_emb) # (topk*batch, len, dim)
            output, hidden = model(src, hidden, vec, eg_emb, eg_mask, Wh_enc, args.cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, p=args.p)
        else:
            (word, chars, vec, src, trg) = elems
            output, hidden = model(src, hidden, vec, args.cuda, seed_feeding=model.seed_feeding, char_emb=char_emb)

        if args.mode not in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
            loss = criterion(output.view(output.size(0) * output.size(1), -1), trg)
        loss.backward()

        # # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        if args.mode in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
            total_loss_gen += loss_gen.data
            total_loss_cpy += loss_cpy.data
        else:
            total_loss += loss.data

        if (i+1) % args.log_interval == 0:
            elapsed = time.time() - start_time

            if args.mode in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
                cur_loss_gen = total_loss_gen[0] / args.log_interval
                cur_loss_cpy = total_loss_cpy[0] / args.log_interval
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.07f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | loss2 {:5.2f}'.format(epoch, i+1, batch_num, lr, elapsed * 1000 / args.log_interval,
                                                            cur_loss_gen, math.exp(cur_loss_gen), cur_loss_cpy), flush=True)
            else:
                cur_loss = total_loss[0] / args.log_interval
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.07f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(epoch, i+1, batch_num, lr, elapsed * 1000 / args.log_interval,
                                                            cur_loss, math.exp(cur_loss)), flush=True)
            total_loss = 0
            total_loss_cpy = 0
            total_loss_gen = 0
            start_time = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='./data/noraset',
                        help='location of the data corpus')
    parser.add_argument('--data_usage', type=int, default=100,
                        help='how many train data to be used (0 - 100 [%])')
    parser.add_argument('--vocab_size', type=int, default=-1,
                        help='vocabulary size (-1 = all vocabl)')
    parser.add_argument('--vec', type=str, default='./data/GoogleNews-vectors-negative300.txt',
                        help='location of the word2vec data')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--mode', type=str, default='SEED',
                        help='training method to be used (NORASET, EDIT)')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--init_dist', type=str, default='uniform',
                        help='distribution to be used to initialize (uniform, xavier)')
    parser.add_argument('--init_range', type=float, default=0.05,
                        help='initialize parameters using uniform distribution between -uniform and uniform.')
    parser.add_argument('--init_embedding', action='store_true',
                        help='initialize word embeddings with read vectors from word2vec')
    parser.add_argument('--seed_feeding', action='store_true',
                        help='feed seed embedding at the first step of decoding')
    parser.add_argument('--gated', action='store_true',
                        help='gated function to update hidden states')
    parser.add_argument('--char', action='store_true',
                        help='character embedding')
    parser.add_argument('--fix_embedding', action='store_true',
                        help='fix initialized word embeddings')
    parser.add_argument('--dhid', type=int, default=300,
                        help='dimention of hidden states')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='factor by which learning rate is decayed (lr = lr * factor)')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=500,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--emb_dropout', type=float, default=0.25,
                        help='dropout applied to embedding layer (0 = no dropout)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--opt', type=str, default='adam',
                        help='optimizer (adam, sgd)')
    parser.add_argument('--p', type=float, default=5.0,
                        help='parameter for batch-ensembling to control peakiniess of the similarity')
    parser.add_argument('--sentence_bleu', type=str, default='./sentence-bleu',
                        help='Compiled binary file of sentece-bleu.cpp')
    parser.add_argument('--nltk_bleu', type=str, default=None,
                        help='use nltk instead of sentence-bleu.cpp to compute bleu (sentence, corpus)')
    parser.add_argument('--words', type=int, default='60',
                        help='number of words to generate')
    parser.add_argument('--topk', type=int, default='3',
                        help='number of nearest neighbors to use in EDIT model')
    parser.add_argument('--ignore_sense_id', action='store_true',
                        help='ignore sense ids in {train | valid | test}.txt')
    parser.add_argument('--order', type=str, default='att1st',
                        help='order of computation in EDIT_MULTI_OUTPUT model (att1st, gate1st)')
    parser.add_argument('--att_type', type=str, default='mul',
                        help='Attention type (mul, add, general, dot)')
    parser.add_argument('--train_src', type=str, default='both',
                        help='data source used for training (both, wordnet, gcide)')
    parser.add_argument('--comb_func', type=str, default='weighted_sum',
                        help='way to combine decoded vectors in EDIT_DECODER model (weighted_sum, attention)')
    parser.add_argument('--coverage', action='store_true',
                        help='coverage penalty in EDIT_COPYNET model')
    parser.add_argument('--return_logloss', action='store_true',
                        help='return logloss durin validataion')
    parser.add_argument('--non_copy', action='store_true',
                        help='Close copygate (only to debug EDIT_COPYNET_SIMPLE)')
    parser.add_argument('--non_attention', action='store_true',
                        help='Close copygate (only to debug EDIT_COPYNET_SIMPLE & EDIT_HIERARCHICAL)')
    parser.add_argument('--valid_all', action='store_true',
                        help='Run validation with all data (only for debugging)')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################
    if args.mode == "EDIT":
        corpus = data.EditCorpus(args.data, args.vec, max_vocab_size=args.vocab_size, topk=args.topk, ignore_sense_id=args.ignore_sense_id, train_src=args.train_src)
    elif args.mode[:10] == "EDIT_MULTI" or args.mode in {"EDIT_DECODER", "EDIT_TRIPLE", "EDIT_HIERARCHICAL", "EDIT_HIERARCHICAL_DIFF", "EDIT_HIRO", "EDIT_COPY"}:
        corpus = data.EditCorpus2(args.data, args.vec, max_vocab_size=args.vocab_size, topk=args.topk, ignore_sense_id=args.ignore_sense_id, train_src=args.train_src)
    elif args.mode in {"EDIT_COPYNET", "EDIT_COPYNET_SIMPLE"}:
        corpus = data.EditCorpus3(args.data, args.vec, max_vocab_size=args.vocab_size, topk=args.topk, ignore_sense_id=args.ignore_sense_id, train_src=args.train_src)
    elif args.mode in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
        corpus = data.EditCorpus4(args.data, args.vec, max_vocab_size=args.vocab_size, topk=args.topk, ignore_sense_id=args.ignore_sense_id, train_src=args.train_src)
    elif args.mode in {"EG_HIERARCHICAL", "EG_GADETSKY"}:
        corpus = data.ExampleCorpus(args)
    else:
        corpus = data.NorasetCorpus(args.data, args.vec, max_vocab_size=args.vocab_size, ignore_sense_id=args.ignore_sense_id, train_src=args.train_src)
    if args.data_usage < 100.0:
        corpus.sample_train_data(args.data_usage)
    translator = translate.Translator(corpus, sentence_bleu=args.sentence_bleu, valid_all=args.valid_all)
    eval_batch_size = 10

    ###############################################################################
    # Build the model
    ###############################################################################
    vocab_size = len(corpus.id2word)
    if args.mode == 'BASELINE':
        model = model.RNNModel(args.model, vocab_size, args.emsize, args.dhid, args.nlayers, args.dropout, args.tied,
                               init_range=args.init_range, init_dist=args.init_dist)
    elif args.mode == 'NORASET':
        model = model.NORASET(args.model, vocab_size, args.emsize, args.dhid, args.emsize, args.nlayers, args.dropout, args.tied,
                         init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len)

    elif args.mode == 'EDIT':
        feature_num = args.emsize * 2 + args.dhid* 2 # v(word); v(word-nn); enc_hid (bi-directional)
        model = model.EDIT(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, args.dropout, args.tied,
                         init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, att_type=args.att_type)

    elif args.mode == 'EDIT_DECODER':
        feature_num = args.emsize * 2 + args.dhid* 2 # v(word); v(word-nn); enc_hid (bi-directional)
        model = model.EDIT_DECODER(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type, comb_func=args.comb_func)

    elif args.mode == 'EDIT_MULTI':
        feature_num = args.emsize + (args.dhid * 2 + args.emsize + 1) * args.topk # v(word); {enc_hid(bi-directional); diff; sim} * topk
        model = model.EDIT_MULTI(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type)

    elif args.mode == 'EDIT_TRIPLE':
        feature_num = args.emsize + args.dhid * 2 # v(word); enc_hid(bi-directional);
        model = model.EDIT_TRIPLE(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type)

    elif args.mode == 'EDIT_HIERARCHICAL':
        feature_num = args.emsize + args.dhid # v(word); bUc;
        model = model.EDIT_HIERARCHICAL(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type, non_attention=args.non_attention)

    elif args.mode == 'EDIT_HIERARCHICAL_DIFF':
        feature_num = args.emsize + args.dhid * 2 # v(word); enc_hid(bi-directional);
        model = model.EDIT_HIERARCHICAL_DIFF(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type, non_attention=args.non_attention)

    elif args.mode == 'EDIT_HIRO':
        feature_num = args.emsize + args.dhid * 2  # v(word); enc_hid(bi-directional);
        model = model.EDIT_HIRO(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type)

    elif args.mode == 'EDIT_COPY':
        feature_num = args.emsize # v(word)
        model = model.EDIT_COPY(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type)

    elif args.mode == 'EDIT_COPYNET':
        feature_num = args.emsize # v(word)
        model = model.EDIT_COPYNET(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, corpus.word2id, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type, coverage=args.coverage)

    elif args.mode == "EDIT_COPYNET_SIMPLE":
        feature_num = args.emsize # v(word)
        model = model.EDIT_COPYNET_SIMPLE(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, corpus.word2id, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type, non_copy=args.non_copy, non_attention=args.non_attention)

    elif args.mode == 'EDIT_CAO':
        feature_num = args.emsize # v(word)
        model = model.EDIT_CAO(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, corpus.word2id, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type, coverage=args.coverage)

    elif args.mode == 'EDIT_CAO_SIMPLE':
        feature_num = args.emsize # v(word)
        model = model.EDIT_CAO_SIMPLE(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, corpus.word2id, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type, coverage=args.coverage)

    elif args.mode == 'EDIT_MULTI_ALL':
        feature_num = args.emsize + args.dhid * 2 + (args.emsize + 1) * args.topk # v(word); enc_hid(bi-directional); {diff; sim} * topk
        model = model.EDIT_MULTI_ALL(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len, args.dropout, args.tied,
                                 init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding, CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, topk=args.topk, att_type=args.att_type)

    elif args.mode == 'EDIT_MULTI_OUTPUT':
        feature_num = args.emsize # v(word)
        model = model.EDIT_MULTI_OUTPUT(args.model, vocab_size, args.emsize, args.dhid, feature_num, args.nlayers, corpus.max_len,
                             args.dropout, args.tied,
                             init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding,
                             CH=args.char, gated=args.gated, nchar=len(corpus.id2char),
                             max_char_len=corpus.max_char_len, topk=args.topk, order=args.order, att_type=args.att_type)
    elif args.mode == "EG_HIERARCHICAL":
        model = model.EG_HIERARCHICAL(args, corpus)

    elif args.mode == "EG_GADETSKY":
        model = model.EG_GADETSKY(args, corpus)

    else:
        print('error!')
        exit()

    if args.init_embedding == True:
        model.init_embedding(corpus, fix_embedding=args.fix_embedding)

    if args.cuda:
        model.cuda()

    if args.mode in {"EDIT_DECODER", "EDIT_COPY", "EDIT_COPYNET", "EDIT_COPYNET_SIMPLE"}:
        criterion = nn.NLLLoss()
    elif args.mode in {'EDIT_CAO', "EDIT_CAO_SIMPLE"}:
        criterion_def = nn.NLLLoss()
        criterion_cpy = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = 99999999
    best_bleu = -1
    no_improvement = 0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            if args.mode not in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
                [val_loss] = translator.eval_log_loss(model, mode='valid', max_batch_size=args.batch_size, cuda=args.cuda, p=args.cuda, return_logloss_matrix=args.return_logloss)
            else:
                elems = translator.eval_log_loss(model, mode='valid', max_batch_size=args.batch_size, cuda=args.cuda, p=args.cuda, return_logloss_matrix=args.return_logloss)
                if args.return_logloss:
                    val_loss_gen, val_loss_cpy, logloss_matrix = elems
                else:
                    val_loss_gen, val_loss_cpy = elems
                val_loss = val_loss_gen + val_loss_cpy
            hyp = translator.greedy(model, mode="valid", max_batch_size=args.batch_size, cuda=args.cuda, p=args.p, max_len=corpus.max_len)
            val_bleu_corpus = translator.bleu(hyp, mode="valid", nltk='corpus')
            val_bleu_sentence = translator.bleu(hyp, mode="valid", nltk='sentence')

            if val_loss < best_val_loss:
                save_checkpoint()
                best_val_loss = val_loss
                best_bleu = val_bleu_sentence # we are interested in the best bleu after ppl stop decreasing
                no_improvement = 0

            elif val_bleu_sentence > best_bleu:
                save_checkpoint()
                best_bleu = val_bleu_sentence
                no_improvement = 0

            else:
                no_improvement += 1
                if no_improvement == 4:
                    load_checkpoint()
                    lr *= args.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= args.lr_decay
                elif no_improvement == 8:
                    break

            print('-' * 112)
            if args.mode not in {"EDIT_CAO", "EDIT_CAO_SIMPLE"}:
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ppl {:8.2f} | BLEU(C/S) {:5.2f} /{:5.2f} | not improved: {:d}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_bleu_corpus*100, val_bleu_sentence*100, no_improvement), flush=True)
            else:
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ppl {:8.2f} | BLEU(C/S) {:5.2f} /{:5.2f} | not improved: {:d} | cpy loss {:5.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss_gen, math.exp(val_loss_gen), val_bleu_corpus * 100, val_bleu_sentence * 100, no_improvement, val_loss_cpy), flush=True)
            print('-' * 112)

    except KeyboardInterrupt:
        print('-' * 112)
        print('Exiting from training early')

    # Load the best saved model.
    load_checkpoint()

    # Run on test data.
    hyp = translator.greedy(model, mode="test", max_batch_size=args.batch_size, cuda=args.cuda, p=args.p, max_len=corpus.max_len, ignore_duplicates=args.ignore_sense_id)
    print('-' * 112)
    print('Decoded:')
    for (word, desc) in hyp:
        print(word, end='\t')
        new_def = []
        for w in desc:
            if w not in corpus.id2word:
                new_def.append('[' + w + ']')
            else:
                new_def.append(w)
        print(' '.join(new_def), flush=True)

    test_loss = translator.eval_log_loss(model, mode='test', max_batch_size=args.batch_size, cuda=args.cuda, p=args.cuda, return_logloss_matrix=args.return_logloss)
    test_bleu_cpp = translator.bleu(hyp, mode="test", nltk=args.nltk_bleu)
    test_bleu_corpus = translator.bleu(hyp, mode="test", nltk='corpus')
    test_bleu_sentence = translator.bleu(hyp, mode="test", nltk='sentence')
    print('=' * 112)
    print('| End of training | test BLEU (sent.cpp / corpus.nltk / sent.nltk): {:5.2f}/{:5.2f}/{:5.2f}'.format(test_bleu_cpp * 100, test_bleu_corpus * 100, test_bleu_sentence * 100))
    print('=' * 112)
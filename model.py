import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import sys
import math
from util import cudaw

class Gate(nn.Module):
    def __init__(self, dhid, dfeature, init_range=0.1, init_dist='uniform', dropout=0.5):
        super(Gate, self).__init__()
        self.dhid = dhid
        self.dfeature = dfeature
        self.linear_z = nn.Linear(self.dhid + self.dfeature, self.dhid)
        self.linear_r = nn.Linear(self.dhid + self.dfeature, self.dfeature)
        self.linear_h_tilde = nn.Linear(self.dhid + self.dfeature, self.dhid)
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.init_weights(init_range, init_dist)

    def init_weights(self, init_range, init_dist):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)
        init_w(self.linear_z.weight.data, init_dist)
        init_w(self.linear_r.weight.data, init_dist)
        init_w(self.linear_h_tilde.weight.data, init_dist)
        self.linear_z.bias.data.fill_(0)
        self.linear_r.bias.data.fill_(0)
        self.linear_h_tilde.bias.data.fill_(0)

    def forward(self, h, features):
        z = self.sigmoid(self.linear_z(torch.cat((features, h), dim=1)))
        r = self.sigmoid(self.linear_r(torch.cat((features, h), dim=1)))
        h_tilde = self.tanh(self.linear_h_tilde(torch.cat((torch.mul(r, features), h), dim=1)))
        h_new = torch.mul((1 - z), h) + torch.mul(z, h_tilde)
        h_new = self.drop(h_new)
        return h_new

class Copy_Gate(nn.Module):
    """Copy gate in Search Engine Guided Non-Parametric Neural Machine Translation [Gu+ AAAI18]"""
    def __init__(self, dinp, dhid, init_range=0.1, init_dist='uniform', dropout=0.0):
        super(Copy_Gate, self).__init__()
        self.dinp = dinp
        self.dhid = dhid
        if dhid == 0:
            self.linear = nn.Linear(self.dinp, 1)
        else:
            self.linear_hid = nn.Linear(self.dinp, self.dhid)
            self.linear_out = nn.Linear(self.dhid, 1)
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.init_weights(init_range, init_dist)

    def init_weights(self, init_range, init_dist):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)
        if self.dhid == 0:
            init_w(self.linear.weight.data, init_dist)
            self.linear.bias.data.fill_(0)
        else:
            init_w(self.linear_hid.weight.data, init_dist)
            init_w(self.linear_out.weight.data, init_dist)
            self.linear_hid.bias.data.fill_(0)
            self.linear_out.bias.data.fill_(0)

    def forward(self, input1, input2=None):
        ## inputs: (b, d)
        if isinstance(input2, type(None)):
            input = input1
        else:
            input = torch.cat([input1, input2], dim=-1)
        if self.dhid == 0:
            out = self.linear(input)
        else:
            hidden = self.linear_hid(input)
            hidden = self.drop(hidden)
            out = self.linear_out(hidden) # (b, 1)
        return self.sigmoid(out)

class NORASET(nn.Module):
    """Seed + G in [Noraset+ AAAI17]"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True, CH=False, gated=False, nchar=-1, max_char_len=-1):
        super(NORASET, self).__init__()
        self.name = ''
        if seed_feeding:
            self.name += 'S'
        if gated:
            self.name += 'G'
        if CH:
            self.name += 'CH'
        self.seed_feeding = seed_feeding
        self.gated = gated
        self.CH = CH

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, dinp)
        self.nchar = nchar
        self.dhid = dhid
        if CH:
            self.dfeature = dfeature + 160
        else:
            self.dfeature = dfeature

        if rnn_type in ['LSTM', 'GRU']:
            rnn = []
            for i in range(nlayers):
                if i == 0:
                    rnn.append(getattr(nn, rnn_type + 'Cell')(dinp, self.dhid))
                else:
                    rnn.append(getattr(nn, rnn_type + 'Cell')(self.dhid, self.dhid))
            self.rnn = nn.ModuleList(rnn)

        else:
            raise ValueError( """An invalid option for `--model` was supplied, options are ['LSTM' or 'GRU']""")
        self.readout = nn.Linear(self.dhid, ntoken)

        if tie_weights:
            if self.dhid != dinp:
                raise ValueError('When using the tied flag, dhid must be equal to emsize')
            self.readout.weight = self.encoder.weight

        # Functions and parameters for Gated functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        if gated:
            self.gate = Gate(self.dhid, self.dfeature)

        # Functions and parameters for Char CNNs
        if CH:
            self.conv1 = nn.Conv2d(1, 10, (2, nchar))
            self.conv2 = nn.Conv2d(1, 30, (3, nchar))
            self.conv3 = nn.Conv2d(1, 40, (4, nchar))
            self.conv4 = nn.Conv2d(1, 40, (5, nchar))
            self.conv5 = nn.Conv2d(1, 40, (6, nchar))
            self.relu = nn.ReLU()
            self.max_pool1 = nn.MaxPool2d((max_char_len - 2 + 1, 1))
            self.max_pool2 = nn.MaxPool2d((max_char_len - 3 + 1, 1))
            self.max_pool3 = nn.MaxPool2d((max_char_len - 4 + 1, 1))
            self.max_pool4 = nn.MaxPool2d((max_char_len - 5 + 1, 1))
            self.max_pool5 = nn.MaxPool2d((max_char_len - 6 + 1, 1))

        self.init_weights(init_range=init_range, init_dist=init_dist, gated=gated, CH=CH)
        self.rnn_type = rnn_type + 'Cell'
        self.nlayers = nlayers

    def init_weights(self, init_range=0.1, init_dist='uniform', gated=True, CH=True):
        if init_dist == 'uniform':
            self.encoder.weight.data.uniform_(-init_range, init_range)
            self.readout.weight.data.uniform_(-init_range, init_range)
            if CH:
                self.conv1.weight.data.uniform_(-init_range, init_range)
                self.conv2.weight.data.uniform_(-init_range, init_range)
                self.conv3.weight.data.uniform_(-init_range, init_range)
                self.conv4.weight.data.uniform_(-init_range, init_range)
                self.conv5.weight.data.uniform_(-init_range, init_range)

        elif init_dist == 'xavier':
            nn.init.xavier_uniform(self.encoder.weight.data)
            nn.init.xavier_uniform(self.readout.weight.data)
            if CH:
                nn.init.xavier_uniform(self.conv1.weight.data)
                nn.init.xavier_uniform(self.conv2.weight.data)
                nn.init.xavier_uniform(self.conv3.weight.data)
                nn.init.xavier_uniform(self.conv4.weight.data)
                nn.init.xavier_uniform(self.conv5.weight.data)
        else:
            return False

        self.readout.bias.data.fill_(0)
        if CH:
            self.conv1.bias.data.fill_(0)
            self.conv2.bias.data.fill_(0)
            self.conv3.bias.data.fill_(0)
            self.conv4.bias.data.fill_(0)
            self.conv5.bias.data.fill_(0)

    def stacked_rnn(self, input, hidden):
        h, c = hidden
        for layer in range(self.nlayers):
            if layer == 0:  # the first layer
                h[layer], c[layer] = self.rnn[layer](input, (h[layer], c[layer]))
            else:
                h[layer], c[layer] = self.rnn[layer](h[layer - 1], (h[layer], c[layer]))
            h[layer] = self.drop(h[layer])
        return (h, c)

    def forward(self, input, hidden, vec, cuda, seed_feeding=True, char_emb=None):
        # concat seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))

        if self.CH:
            features = torch.cat([vec, char_emb], dim=1)
        else:
            features = vec

        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda()
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())

        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(emb[i], (h, c))

            # update h in the last layer with a gated function
            if self.gated:
                h[-1] = self.gate(h[-1], features)

            outputs[i] = h[-1]

        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)
        return decoded, (h, c)

    def init_hidden(self, bsz, dhid, bidirectional=False):
        weight = next(self.parameters()).data
        if bidirectional:
            hidden = Variable(weight.new((self.nlayers * 2), bsz, dhid).zero_())
        else:
            hidden = [Variable(weight.new(bsz, dhid).zero_())] * self.nlayers

        if self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            if bidirectional:
                cell = Variable(weight.new((self.nlayers * 2), bsz, dhid).zero_())
            else:
                cell = [Variable(weight.new(bsz, dhid).zero_())] * self.nlayers
            return hidden, cell
        else:
            return hidden

    def init_embedding(self, corpus, fix_embedding=False):
        sys.stderr.write('Initializing word embedding layer...')
        word2vec = corpus.read_vector(corpus.vec_path, corpus.word2id)

        init_emb = []
        for id, word in enumerate(corpus.id2word):
            if word in word2vec:
                init_emb.append(word2vec[word])
            else: # use random initialized embeddings
                init_emb.append([val for val in self.encoder.weight[id].data])
        self.encoder.weight.data = torch.FloatTensor(init_emb)

        if fix_embedding:
            self.encoder.weight.requires_grad = False

        sys.stderr.write('  Done\n')

    def get_char_embedding(self, chars):
        cnn1 = self.max_pool1(self.relu(self.conv1(chars)))
        cnn2 = self.max_pool2(self.relu(self.conv2(chars)))
        cnn3 = self.max_pool3(self.relu(self.conv3(chars)))
        cnn4 = self.max_pool4(self.relu(self.conv4(chars)))
        cnn5 = self.max_pool5(self.relu(self.conv5(chars)))
        char_emb = self.drop(torch.cat((cnn1, cnn2, cnn3, cnn4, cnn5), dim=1).squeeze(2).squeeze(2))
        # print('debug (def getchar_embedding): char_emb.size()=', char_emb.size())
        return char_emb

class Attention(nn.Module):
    def __init__(self, denc, ddec, datt, type='mul', init_range=0.1, init_dist='uniform', dropout=0.0):
        """
        type can be one of the following:

        # [Luong+ 15]
        'general': dot(s, W_dec h)
        'dot': dot(s, h) # Note that dim(s) and dim(h) must be the same

        # [Bahdanau+ 15]
        'add': dot(v, tanh(W_enc s + W_dec h)) # [Bahdanau+ 15]

        # [Britz+ 17]
        'mul': dot(W_enc s,  W_dec h) # [Britz+ 17]

        where s and h denote encoder and decoder states, respectively
        """
        super(Attention, self).__init__()
        self.denc, self.ddec, self.datt = denc, ddec, datt
        self.type = type
        self.drop = nn.Dropout(dropout)

        if self.type == 'general': # No bias following Open-NMT
            self.W_enc = nn.Linear(self.denc, self.datt, bias=False)
        elif self.type == 'add': # Add bias only to W_dec following Open-NMT
            self.W_enc = nn.Linear(self.denc, self.datt, bias=False)
            self.W_dec = nn.Linear(self.ddec, self.datt, bias=True)
            self.v = nn.Linear(self.datt, 1, bias=False)
            self.tanh = nn.Tanh()
        elif self.type == 'mul': # Add bias to W_enc & W_ded following [Britz+ 17]
            self.W_enc = nn.Linear(self.denc, self.datt, bias=True)
            self.W_dec = nn.Linear(self.ddec, self.datt, bias=True)
        self.init_weights(init_range=init_range, init_dist=init_dist)

    def init_weights(self, init_range, init_dist):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)

        if self.type == 'general':
            init_w(self.W_enc.weight.data, init_dist)
        elif self.type == 'add':
            init_w(self.W_enc.weight.data, init_dist)
            init_w(self.W_dec.weight.data, init_dist)
            init_w(self.v.weight.data, init_dist)
        elif self.type == 'mul':
            init_w(self.W_enc.weight.data, init_dist)
            init_w(self.W_dec.weight.data, init_dist)

        if self.type == 'add':
            self.W_dec.bias.data.fill_(0)
        elif self.type == 'mul':
            self.W_enc.bias.data.fill_(0)
            self.W_dec.bias.data.fill_(0)

    def map_enc_states(self, h_enc):
        if self.type in set(['general', 'add', 'mul']):
            return self.W_enc(h_enc).permute(1, 0, 2) # (topk*batch, len, dim)
        elif self.type == 'dot':
            return h_enc.permute(1, 0, 2) # (topk*batch, len, 2dim)

    def att_score(self, h_dec, Wh_enc, topk=1):
        if self.type == 'general':
            h_dec_repeat = h_dec.repeat(topk, 1).unsqueeze(2) # (b, dim) -> (k*b, dim, 1)
            return torch.bmm(Wh_enc, h_dec_repeat).squeeze(2) # (k*b, len)

        elif self.type == 'add':
            Wh_dec = self.W_dec(h_dec).repeat(1, topk).view(h_dec.size(0) * topk, -1).unsqueeze(1)  # (b, dim) -> (k*b, 1, dim)
            sum = Wh_enc + Wh_dec.repeat(1, Wh_enc.size(1), 1) # (k*b, len, dim)
            # Wh_dec = self.W_dec(h_dec) # (b, d)
            # Wh_dec = Wh_dec.unsqueeze(1).expand(Wh_dec.size(0), topk, Wh_dec.size(1)).contiguous().view(-1, Wh_dec.size(1)).unsqueeze(1) # (k*b, 1, dim)
            # sum = Wh_enc + Wh_dec.expand(Wh_dec.size(0), Wh_enc.size(1), Wh_dec.size(2))

            return self.v(self.tanh(sum)).squeeze(2) # (k*b, len)

        elif self.type == 'mul':
            Wh_dec = self.W_dec(h_dec).repeat(1, topk).view(h_dec.size(0) * topk, -1).unsqueeze(2)  # (b, dim) -> (k*b, dim, 1)
            return torch.bmm(Wh_enc, Wh_dec).squeeze(2)  # (k*b, len, dim) * (k*b, dim, 1) -> (k*b, len)

        elif self.type == 'dot':
            pass

    def forward(self, h_dec, h_enc, h_enc_mask=None, Wh_enc=None, topk=1, flatten_enc_states=False, return_att_score=False, att_score=None):
        # h_enc: (len, k*b, 2dim)
        # Wh_enc: (k*b, len, dim)
        # h_dec:  (b, dim)
        if isinstance(Wh_enc, type(None)):
            Wh_enc = self.map_enc_states(h_enc)

        if isinstance(att_score, type(None)):
            att_score = self.att_score(h_dec, Wh_enc, topk) # (topk*batch, len)

        if not isinstance(h_enc_mask, type(None)):
            att_score = att_score + h_enc_mask.permute(1, 0)

        if return_att_score:
            return att_score

        if not flatten_enc_states:
            att_prob = F.softmax(att_score, dim=1) # (topk*batch, len)
            att_result = torch.bmm(att_prob.unsqueeze(1), h_enc.permute(1, 0, 2)).squeeze(1)  # (topk * batch, dim*2)

        else: # softmax over all words in top-k NNs
            batch = h_dec.size(0)
            h_enc_flat = h_enc.permute(1, 0, 2).contiguous().view(batch, -1, h_enc.size(2))  # (len, k*b, 2dim) -> (batch, len*k, 2dim)
            att_score_flat = att_score.view(batch, -1)  # (batch, len*topk)
            att_prob = F.softmax(att_score_flat, dim=1)  # softmax over all words in top-k NNs
            att_result = torch.bmm(att_prob.unsqueeze(1), h_enc_flat).squeeze(1)  # (batch, dim*2)

        att_result = self.drop(att_result)
        return att_result, att_prob

class Attetion_Libovicky(Attention):
    """Eq.8-10 described in [Libovicky & Helcl ACL17]"""
    def __init__(self, denc, ddec, datt, type='mul', init_range=0.1, init_dist='uniform', dropout=0.0, shared_U=False, Uc=True, topk=-1):
        if shared_U and not Uc: # normal attention function
            return Attention(denc, ddec, datt, type=type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        else:
            super(Attention, self).__init__()
        if (not shared_U or Uc) and topk == -1:
            sys.stderr('Specify topk to use unshared Ub or Uc!\n')
            exit()

        self.denc, self.ddec, self.datt = denc, ddec, datt
        self.type = type
        self.drop = nn.Dropout(dropout)
        self.shared_U = shared_U
        self.Uc = Uc
        self.topk = topk

        if self.type == 'general':
            if not self.shared_U:
                self.W_encs = nn.ModuleList([nn.Linear(self.denc, self.datt, bias=False)] * self.topk)
            else:
                self.W_enc = nn.Linear(self.denc, self.datt, bias=False)
        elif self.type == 'add':
            if not self.shared_U:
                self.W_encs = nn.ModuleList([nn.Linear(self.denc, self.datt, bias=False)] * self.topk)
            else:
                self.W_enc = nn.Linear(self.denc, self.datt, bias=False)
            self.W_dec = nn.Linear(self.ddec, self.datt, bias=True)
            self.v = nn.Linear(self.datt, 1, bias=False)
            self.tanh = nn.Tanh()
        elif self.type == 'mul':
            if not self.shared_U:
                self.W_encs = nn.ModuleList([nn.Linear(self.denc, self.datt, bias=True)] * self.topk)
            else:
                self.W_enc = nn.Linear(self.denc, self.datt, bias=True)
            self.W_dec = nn.Linear(self.ddec, self.datt, bias=True)

        if self.Uc:
            if not self.shared_U:
                self.U_cs = nn.ModuleList([nn.Linear(self.denc, self.datt)] * self.topk)
            else:
                self.U_c = nn.Linear(self.denc, self.datt)
        self.init_weights(init_range=init_range, init_dist=init_dist)

    def init_weights(self, init_range, init_dist):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)

        if self.shared_U:
            super(Attetion_Libovicky, self).init_weights(init_range=init_range, init_dist=init_dist)
            if self.Uc:
                init_w(self.U_c.weight.data, init_dist)
                self.U_c.bias.data.fill_(0)

        else: # if Us are not shared
            for W_enc in self.W_encs:
                init_w(W_enc.weight.data, init_dist)

            if self.type == 'add':
                init_w(self.W_dec.weight.data, init_dist)
                init_w(self.v.weight.data, init_dist)
            elif self.type == 'mul':
                init_w(self.W_dec.weight.data, init_dist)

            if self.type == 'add':
                self.W_dec.bias.data.fill_(0)
            elif self.type == 'mul':
                for W_enc in self.W_encs:
                    W_enc.bias.data.fill_(0)
                self.W_dec.bias.data.fill_(0)

            if self.Uc:
                for U_c in self.U_cs:
                    init_w(U_c.weight.data, init_dist)
                    U_c.bias.data.fill_(0)

    def map_enc_states(self, h_enc, cuda):
        if self.shared_U:
            return super(Attetion_Libovicky, self).map_enc_states(h_enc)

        if cuda:
            Wh_encs_T = Variable(torch.FloatTensor(h_enc.size(0), h_enc.size(1), self.W_encs[0].out_features).zero_()).cuda() # (k, b, d)
        else:
            Wh_encs_T = Variable(torch.FloatTensor(h_enc.size(0), h_enc.size(1), self.W_encs[0].out_features).zero_())

        for i in range(self.topk):
            hoge = self.W_encs[i](h_enc[i])
            Wh_encs_T[i] = hoge # (b, d)

        return Wh_encs_T.permute(1, 0, 2) # (b, k, d)

    def forward(self, h_dec, h_enc, Wh_enc=None, return_att_score=False, att_score=None, cuda=False):
        # h_enc: (k, b, d)
        # Wh_enc: (b, k, d)
        # h_dec:  (b, d)
        if isinstance(Wh_enc, type(None)):
            Wh_enc = self.map_enc_states(h_enc, cuda=cuda)

        if isinstance(att_score, type(None)):
            att_score = self.att_score(h_dec, Wh_enc) # (b, k)
        if return_att_score:
            return att_score
        att_prob = F.softmax(att_score, dim=1) # (b, k)

        if not self.shared_U:
            if cuda:
                bUc = Variable(torch.FloatTensor(h_enc.size(0), h_enc.size(1), self.U_cs[0].out_features).zero_()).cuda() # (k, b, d)
            else:
                bUc = Variable(torch.FloatTensor(h_enc.size(0), h_enc.size(1), self.U_cs[0].out_features).zero_())
            for i in range(self.topk):
                bUc[i] = att_prob[:, i].unsqueeze(1) * self.U_cs[i](h_enc[i]) # [(b, k), (b,k), ...]
            att_result = torch.sum(bUc, dim=0) # (b, d)
        else:
            att_result = torch.sum(att_prob.permute(1, 0).unsqueeze(2) * self.U_c(h_enc), dim=0) # (b, d)

        att_result = self.drop(att_result)
        return att_result, att_prob

class Attention_Gadetsky(nn.Module):
    def __init__(self, denc, datt, dout, init_range=0.1, init_dist='uniform', dropout=0.5):
        super(Attention_Gadetsky, self).__init__()
        self.denc, self.datt, self.dout = denc, datt, dout # no decoder states

        self.W_enc = nn.Linear(self.denc, self.datt)
        self.W_mask = nn.Linear(self.datt, self.dout)

        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.init_weights(init_range=init_range, init_dist=init_dist)

    def init_weights(self, init_range, init_dist):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)

            init_w(self.W_enc.weight.data, init_dist)
            init_w(self.W_mask.weight.data, init_dist)
            self.W_enc.bias.data.fill_(0)
            self.W_mask.bias.data.fill_(0)

    def forward(self, h_enc, eg_mask):
        # eg_mask: (len, b); binary mask with 1 & 0
        ann_output = self.tanh(self.W_enc(self.drop(h_enc))).permute(1, 0, 2) # (batch, len, dim)
        ann_output_masked = eg_mask.permute(1, 0).unsqueeze(2) * ann_output  # (batch, len, dim)
        length = eg_mask.sum(0).unsqueeze(1) # (b, 1)

        ann_output_average = ann_output_masked.sum(1) / length # (b, dim)
        vec_mask = self.sigmoid(self.W_mask(ann_output_average))

        return vec_mask

class EDIT(NORASET):
    """Our EDIT model that based on S + G + CH model [Noraset+ AAAI17]"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, att_type='mul'):
        super(EDIT, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, dropout, tie_weights, init_range, init_dist, seed_feeding=seed_feeding, CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len)
        self.name = 'EDIT'
        self.dhid_nn = dhid
        self.max_len = max_len
        self.attention = Attention(self.dhid * 2, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.softmax = nn.Softmax()

        # Encoder for descriptions of nearest neighbor words
        if rnn_type in ['LSTM', 'GRU']:
            self.nnEncoder = getattr(nn, rnn_type)(dinp, self.dhid_nn, nlayers, dropout=dropout, bidirectional=True)

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))

        # build feature vector [vec; vec_diff; char_emb]
        features = torch.cat([vec, vec_diff], dim=1)
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda()
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())


        for i in range(emb.size(0)):
            h, c = self.stacked_rnn(emb[i], (h, c))
            att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc)
            h[-1] = self.gate(h[-1], torch.cat((features, att_result), dim=1))
            outputs[i] = h[-1]

        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)
        if batch_ensemble:
            weighted_sim = sim * p # (batch)
            sim_weights = self.softmax(weighted_sim.unsqueeze(0)).permute(1,0) # (batch, 1)
            decoded_flat = decoded.view(-1, decoded.size(2)) # ((len*batch), vocab)
            probs = self.softmax(decoded_flat).view(decoded.size()) # (len, batch, vocab)
            weighted_sum = torch.log((probs * sim_weights).sum(dim=1).unsqueeze(1)) # (len, 1, vocab)
            return weighted_sum, (h, c)

        return decoded, (h, c)

    def get_nn_embedding(self, desc_nn):
        # encode description of the nearest neighbor word
        emb_nn = self.drop(self.encoder(desc_nn))
        hidden_nn = self.init_hidden(desc_nn.size(1), self.dhid_nn, bidirectional=True)
        nn_emb, hidden_nn = self.nnEncoder(emb_nn, hidden_nn)
        nn_emb = self.drop(nn_emb)
        return nn_emb

class EDIT_MULTI(EDIT):
    """Our EDIT model that based on S + G + CH model [Noraset+ AAAI17]"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul'):
        super(EDIT_MULTI, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_MULTI'
        self.topk = topk
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))

        # build feature vector [vec; sim_flat; diff_flat; char_emb]
        sim_flat = sim.view(input.size(1), -1)  # (batch, topk)
        diff_flat = vec_diff.view(input.size(1), -1)  # (batch, dim*topk)
        features = torch.cat([vec, sim_flat, diff_flat], dim=1)
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda()
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())
        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(emb[i], (h, c))

            # compute attention vector
            att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk)
            att_result_flat = att_result.view(h[-1].size(0), -1) # (batch, dim*2 * topk)
            h[-1] = self.gate(h[-1], torch.cat((features, att_result_flat), dim=1))
            outputs[i] = h[-1]

        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)
        return decoded, (h, c)

class EDIT_MULTI_DOUBLE(EDIT):
    """Our EDIT model that based on S + G + CH model [Noraset+ AAAI17]"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul'):
        super(EDIT_MULTI_DOUBLE, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_MULTI'
        self.topk = topk
        self.attention_topk = Attention(self.dhid * 2, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec; sim_flat; diff_flat; char_emb]
        sim_flat = sim.view(input.size(1), -1)  # (batch, topk)
        diff_flat = vec_diff.view(input.size(1), -1)  # (batch, dim*topk)
        features = torch.cat([vec, sim_flat, diff_flat], dim=1)
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda()
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())
        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(emb[i], (h, c))

            # attention over source words
            att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk) # (k*b, 2dim)

            # attention over att_results
            att_emb = att_result.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, 2dim); Note that att_result.view(k, b, 2dim) is wrong
            att_result_topk, _ = self.attention_topk(h[-1], att_emb)  # (b, 2dim)

            # gated function
            h[-1] = self.gate(h[-1], torch.cat((features, att_result_topk), dim=1))
            outputs[i] = h[-1]

        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)
        return decoded, (h, c)

class EDIT_TRIPLE(EDIT):
    """Our EDIT model that based on S + G + CH model [Noraset+ AAAI17]"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul'):
        super(EDIT_TRIPLE, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_TRIPLE'
        self.topk = topk
        self.attention_topk = Attention(self.dhid * 4, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        # self.attention_emb = Attention(self.dhid, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, nn_sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None):
        # Note that nn_sim is computed w/ self.attention_emb, but not the original similarities read from *.nn files!

        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec; char_emb]
        features = vec
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda()
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())
        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(emb[i], (h, c))

            # attention over source words
            att_result, _ = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk) # (k*b, 2dim)

            # attention over k vectors
            att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, 2dim); Note that att_result.view(k, b, 2dim) is wrong
            vec_diff_reshape = vec_diff.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, dim)
            att2_input = torch.cat([att_result_reshape, vec_diff_reshape, torch.abs(vec_diff_reshape)], dim=2) # (k, b, 4d)
            att2_score = self.attention_topk(h[-1], att2_input, return_att_score=True) # (1*b, k)

            # combine two attention scores and compute final attended vector
            att_score_comb = nn_sim + att2_score
            att_prob = F.softmax(att_score_comb, dim=1) # (1*b, k)
            att_result_topk = torch.bmm(att_prob.unsqueeze(1), att_result_reshape.permute(1, 0, 2)).squeeze(1)  # (b, 2d)

            # gated function
            h[-1] = self.gate(h[-1], torch.cat((features, att_result_topk), dim=1))
            outputs[i] = h[-1]

        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)
        return decoded, (h, c)

    def get_nn_sim(self, vec, vec_diff):
        # compute attention scores Att(w_trg, w_nn): (1*b, k)
        vec_nn = vec.repeat(1, self.topk).view(-1, vec.size(1)) - vec_diff # (k*b, d)
        vec_nn_reshape = vec_nn.view(-1, self.topk, vec_nn.size(1)).permute(1, 0, 2)# (k, 1*b, d)
        Wh_enc = self.attention_emb.map_enc_states(vec_nn_reshape) # (k, 1*b, d) -> (1*b, k, d)
        att_score = self.attention_emb(vec, None, Wh_enc=Wh_enc, return_att_score=True) # (1*b, k)
        return att_score

class EDIT_MULTI_ALL(EDIT):
    """Our EDIT model that based on S + G + CH model [Noraset+ AAAI17]"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul'):
        super(EDIT_MULTI_ALL, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_MULTI'
        self.topk = topk
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))

        # build feature vector [vec; sim_flat; diff_flat; char]
        sim_flat = sim.view(input.size(1), -1)  # (batch, topk)
        diff_flat = vec_diff.view(input.size(1), -1)  # (batch, dim*topk)
        features = torch.cat([vec, sim_flat, diff_flat], dim=1)
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda()
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())
        for i in range(emb.size(0)):
            h, c = self.stacked_rnn(emb[i], (h, c))
            att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk, flatten_enc_states=True)
            h[-1] = self.gate(h[-1], torch.cat((features, att_result), dim=1))
            outputs[i] = h[-1]

        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)
        return decoded, (h, c)

class EDIT_MULTI_OUTPUT(EDIT):
    """combine nn_encoder at the last linear layer"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, order='att1st', att_type='mul'):
        super(EDIT_MULTI_OUTPUT, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_MULTI_OUTPUT'
        self.topk = topk
        self.logsoftmax = nn.LogSoftmax()
        self.readout = nn.Linear(self.dhid + self.topk * (1 + self.dhid * 2 + self.dhid), ntoken) # [h'(300); (w_nn(600); sim(1); diff(300)) * topk]
        self.init_weights()
        self.order = order

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        length = emb.size(0)
        batch = emb.size(1)

        # build feature vector [vec; char_emb]
        features = vec
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            att_results = Variable(torch.FloatTensor(length, batch, self.dhid_nn * 2 * self.topk).zero_()).cuda() # (len, batch, dim*2*topk)
            outputs = Variable(torch.FloatTensor(length, batch, h[-1].size(1)).zero_()).cuda() # (len, batch, dim)
        else:
            att_results = Variable(torch.FloatTensor(length, batch, self.dhid_nn * 2 * self.topk).zero_()) # (len, batch, dim*2*topk)
            outputs = Variable(torch.FloatTensor(length, batch, h[-1].size(1)).zero_())

        for i in range(length):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(emb[i], (h, c))

            if self.order == 'att1st':
                att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk)
            h[-1] = self.gate(h[-1], features)

            outputs[i] = h[-1] # (len, batch, dim)

            if self.order == 'gate1st':
                att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk)
            att_results[i] = att_result.view(batch, -1)

        diff_flat = vec_diff.view(input.size(1), -1).unsqueeze(0).repeat(length, 1, 1) # (len, batch, dim*topk)
        diff_flat = self.drop(diff_flat)
        sim_flat = sim.view(input.size(1), -1).unsqueeze(0).repeat(length, 1, 1)  # (len, batch, topk)
        decode_features = torch.cat((diff_flat, sim_flat, att_results, outputs), dim=2) # (len, batch, dim*topk + topk + dim*2*topk + dim)
        decoded = self.readout(decode_features).view(outputs.size(0), outputs.size(1), self.readout.out_features) # (len, batch, vocab)
        return decoded, (h, c)

class EDIT_DECODER(EDIT):
    """combine decoder states after gated function"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul', comb_func='weighted_sum'):
        super(EDIT_DECODER, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_DECODER'
        self.topk = topk
        self.comb_func = comb_func
        if self.comb_func == 'attention':
            self.attention_topk = Attention(ntoken, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=0.0)

    def init_hidden(self, bsz, dhid, bidirectional=False):
        # needs decoder hidden states to be (b*k, dim)
        return super(EDIT_DECODER, self).init_hidden(bsz * self.topk, dhid, bidirectional=bidirectional)

    def get_nn_embedding(self, desc_nn):
        # needs encoder hidden states to be the same as EDIT model
        emb_nn = self.drop(self.encoder(desc_nn))
        hidden_nn = super(EDIT_DECODER, self).init_hidden(desc_nn.size(1), self.dhid_nn, bidirectional=True)
        nn_emb, hidden_nn = self.nnEncoder(emb_nn, hidden_nn)
        nn_emb = self.drop(nn_emb)
        return nn_emb

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec_repeat; vec_diff; char_emb_repeat] (k*b, d + d + 160)
        vec_repeat = vec.repeat(1, self.topk).view(batch * self.topk, -1) # (b, dim) -> (k*b, dim)
        features = torch.cat([vec_repeat, vec_diff], dim=1)
        if self.CH:
            char_emb_repeat = char_emb.repeat(1, self.topk).view(batch * self.topk, -1) # (b, 160) -> (k*b, 160)
            features = torch.cat([features, char_emb_repeat], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda() # (len, k*b, dim)
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())
        emb_repeat = emb.repeat(1, 1, self.topk).view(emb.size(0), batch * self.topk, -1) # (len, b, dim) -> (len, k*b, dim)
        for i in range(emb.size(0)):
            h, c = self.stacked_rnn(emb_repeat[i], (h, c))
            att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc)
            h[-1] = self.gate(h[-1], torch.cat((features, att_result), dim=1))
            outputs[i] = h[-1] # (k*b, dim)

        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features) # (len, k*b, vocab)

        # top-k ensembling
        if self.comb_func == 'weighted_sum':
            weighted_sim = sim * p # (k*b)
            sim_weights = F.softmax(weighted_sim.view(-1, self.topk), dim=1).view(-1).unsqueeze(0).unsqueeze(2) # (1, k*b, 1)
            probs = F.softmax(decoded, dim=-1) # (len, k*b, vocab)
            weighted_probs = probs * sim_weights
            weighted_probs_split = weighted_probs.permute(1, 0, 2).unsqueeze(1).contiguous().view(batch, self.topk, probs.size(2), probs.size(0)) # (b, k, vocab, len)
            weighted_sum = torch.log(weighted_probs_split.sum(dim=1)).permute(2, 0, 1).contiguous()  # (len, b, vocab)

        elif self.comb_func == 'attention':
            if cuda:
                weighted_sum = Variable(torch.FloatTensor(decoded.size(0), batch, decoded.size(2)).zero_()).cuda() # (len, b, vocab)
            else:
                weighted_sum = Variable(torch.FloatTensor(decoded.size(0), batch, decoded.size(2)).zero_())

            for i in range(decoded.size(0)): # TODO: Parallelize the for loop by modifing Attention class!
                decoded_reshape = decoded[i].view(batch, self.topk, -1).permute(1, 0, 2) # (k*b, vocab) -> (k, b, vocab)
                att_result_topk, _ = self.attention_topk(outputs[i], decoded_reshape)  # (b, vocab)
                weighted_sum[i] = torch.log(att_result_topk)

        return weighted_sum, (h, c)

class EDIT_HIERARCHICAL(EDIT):
    """Hierarchical attention [Libovicky & Helcl ACL17]"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul', non_attention=False):
        super(EDIT_HIERARCHICAL, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_HIERARCHICAL'
        self.topk = topk
        self.attention_topk = Attetion_Libovicky(self.dhid * 2, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout, shared_U=False, Uc=True, topk=self.topk)
        # self.attention_emb = Attention(self.dhid, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.logsoftmax = nn.LogSoftmax()
        self.non_attention = non_attention

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, nn_sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None, return_att_weights=False):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec; char_emb]
        features = vec
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda()
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())

        if return_att_weights:
            if cuda:
                att_weights = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).zero_()).cuda()
            else:
                att_weights = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).zero_())

        att_result_topk = cudaw(Variable(torch.FloatTensor(batch, emb.size(2)).zero_()))
        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(emb[i], (h, c))

            if not self.non_attention:
                # attention over source words
                att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk) # (k*b, 2dim)

                # attention over k vectors
                att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, 2dim); Note that att_result.view(k, b, 2dim) is wrong
                att_result_topk, att_prob_topk = self.attention_topk(h[-1], att_result_reshape, cuda=cuda)

            # gated function
            h[-1] = self.gate(h[-1], torch.cat((features, att_result_topk), dim=1))
            outputs[i] = h[-1]

            # save attention weights (Jan. 22)
            if return_att_weights:
                att_weight = att_prob * att_prob_topk.view(1, -1).permute(1,0) # (k*b, srcLen)
                att_weights[i] = att_weight.view(batch, -1) # (b, srcLen*k)

        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)

        if return_att_weights:
            return decoded, (h, c), att_weights
        else:
            return decoded, (h, c)

class EDIT_HIERARCHICAL_DIFF(EDIT):
    """Triple attention w/ Hierarchical attention [Libovicky & Helcl ACL17]"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul', non_attention=False):
        super(EDIT_HIERARCHICAL_DIFF, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_HIERARCHICAL_DIFF'
        self.topk = topk
        self.attention_topk = Attetion_Libovicky(self.dhid * 4, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout, shared_U=False, Uc=True, topk=self.topk)
        self.attention_emb = Attention(self.dhid, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.logsoftmax = nn.LogSoftmax()
        self.non_attention = non_attention

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, nn_sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None, return_att_weights=False):
        # Note that nn_sim is computed w/ self.attention_emb, but not the original similarities read from *.nn files!
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec; char_emb]
        features = vec
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda()
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())

        att_result_topk = cudaw(Variable(torch.FloatTensor(batch, 2 * emb.size(2)).zero_()))
        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(emb[i], (h, c))

            if not self.non_attention:
                # attention over source words
                att_result, _ = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk) # (k*b, 2dim)

                # attention over k vectors
                att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, 2dim)
                vec_diff_reshape = vec_diff.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, dim)
                att2_input = torch.cat([att_result_reshape, vec_diff_reshape, torch.abs(vec_diff_reshape)], dim=2) # (k, b, 4d)
                att2_score = self.attention_topk(h[-1], att2_input, return_att_score=True, cuda=cuda) # (1*b, k)

                # combine two attention scores and compute final attended vector
                att_score_comb = nn_sim + att2_score
                att_prob = F.softmax(att_score_comb, dim=1)  # (b, k)
                att_result_topk = torch.bmm(att_prob.unsqueeze(1), att_result_reshape.permute(1, 0, 2)).squeeze(1)  # (b, 2d)

            # gated function
            h[-1] = self.gate(h[-1], torch.cat((features, att_result_topk), dim=1))
            outputs[i] = h[-1]


        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)
        return decoded, (h, c)

    def get_nn_sim(self, vec, vec_diff):
        # compute attention scores Att(w_trg, w_nn): (1*b, k)
        vec_nn = vec.repeat(1, self.topk).view(-1, vec.size(1)) - vec_diff # (k*b, d)
        vec_nn_reshape = vec_nn.view(-1, self.topk, vec_nn.size(1)).permute(1, 0, 2)# (k, 1*b, d)
        att_score = self.attention_emb(vec, vec_nn_reshape, return_att_score=True) # (1*b, k)
        return att_score

class EDIT_HIRO(EDIT):
    """Hiro's aggregation function"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul'):
        super(EDIT_HIRO, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_HIRO'
        self.topk = topk
        self.attention_emb = Attention(self.dhid, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, Wh_enc, nn_sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None):
        # Note that nn_sim is computed w/ self.attention_emb, but not the original similarities read from *.nn files!
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec; char_emb]
        features = vec
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()).cuda()
        else:
            outputs = Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())
        desc_nn_mask_reshape = torch.clamp(desc_nn_mask.permute(1, 0) + 1, min=0) # binary mask whose size is (k*b, len)
        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(emb[i], (h, c))

            # attention over source words; c_k
            att_score = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk, return_att_score=True) # (k*b, len)
            att_prob = F.softmax(att_score, dim=1)  # (k*b, len)
            att_result = torch.bmm(att_prob.unsqueeze(1), nn_emb.permute(1, 0, 2)).squeeze(1)  # (k*b, 2dim)

            # compute top-k weights based on length-normalized attention score; s_k
            att_score_masked = desc_nn_mask_reshape * att_score # -inf -> nan
            att_score_masked[desc_nn_mask_reshape.byte() ^ 1] = 0 # nan -> 0
            att_score_sum = torch.sum(att_score_masked, dim=1) # (k*b)
            desc_nn_len = torch.sum(desc_nn_mask_reshape, dim=1) # (k*b)
            att_score_ave = att_score_sum / desc_nn_len # (k*b)

            # sum_k(c_k * s_k)
            att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2)  # (k, b, 2d)
            att_score_ave_reshape = att_score_ave.view(-1, self.topk).permute(1, 0).unsqueeze(2) # (k, b, 1)
            weighted_att_result = att_result_reshape * att_score_ave_reshape
            att_result_topk = torch.sum(weighted_att_result, dim=0) # (b, 2d)

            # gated function
            h[-1] = self.gate(h[-1], torch.cat((features, att_result_topk), dim=1))
            outputs[i] = h[-1]

        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)
        return decoded, (h, c)

    def get_nn_sim(self, vec, vec_diff):
        # compute attention scores Att(w_trg, w_nn): (1*b, k)
        vec_nn = vec.repeat(1, self.topk).view(-1, vec.size(1)) - vec_diff # (k*b, d)
        vec_nn_reshape = vec_nn.view(-1, self.topk, vec_nn.size(1)).permute(1, 0, 2)# (k, 1*b, d)
        att_score = self.attention_emb(vec, vec_nn_reshape, return_att_score=True) # (1*b, k)
        return att_score

class EDIT_COPY(EDIT):
    """Copy mechanism similar to SEG-NMT [Gu+ AAAI18]"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul'):
        super(EDIT_COPY, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_COPY'
        self.topk = topk
        self.attention_topk = Attetion_Libovicky(self.dhid * 4, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout, shared_U=False, Uc=True, topk=self.topk)
        self.attention_emb = Attention(self.dhid, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.logsoftmax = nn.LogSoftmax()
        self.copy_gate = Copy_Gate(self.dhid * 2, self.dhid)

    def forward(self, input, hidden, vec, vec_diff, nn_1hot, nn_emb, desc_nn_mask, Wh_enc, nn_sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None):
        # Note that nn_sim is computed w/ self.attention_emb, but not the original similarities read from *.nn files!
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec; char_emb]
        features = vec
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if cuda:
            h_gen = Variable(torch.FloatTensor(emb.size(0), batch, h[-1].size(1)).zero_()).cuda()
            h_copy = Variable(torch.FloatTensor(emb.size(0), batch, self.dhid).zero_()).cuda()
            p_copies = Variable(torch.FloatTensor(emb.size(0), batch, nn_1hot.size(2)).zero_()).cuda()
        else:
            h_gen = Variable(torch.FloatTensor(emb.size(0), batch, h[-1].size(1)).zero_())
            h_copy = Variable(torch.FloatTensor(emb.size(0), batch, self.dhid).zero_())
            p_copies = Variable(torch.FloatTensor(emb.size(0), batch, nn_1hot.size(2)).zero_())

        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(emb[i], (h, c))

            # attention over source words
            att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk) # (k*b, 2dim), (k*b, len)

            # attention over k vectors
            att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, 2dim); Note that att_result.view(k, b, 2dim) is wrong
            vec_diff_reshape = vec_diff.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, dim)
            att2_input = torch.cat([att_result_reshape, vec_diff_reshape, torch.abs(vec_diff_reshape)], dim=2) # (k, b, 4d)
            att2_score = self.attention_topk(h[-1], att2_input, return_att_score=True, cuda=cuda) # (1*b, k)

            # combine two attention scores and compute final attended vector
            att_score_comb = nn_sim + att2_score
            att_result_topk, att_prob_topk = self.attention_topk(h[-1], att2_input, att_score=att_score_comb, cuda=cuda) # (b, d), (b, k)

            # compute p_copy based on attention probs. # TODO: check
            p_copy_scat = nn_1hot * att_prob.permute(1, 0).unsqueeze(2) # (len, k*b, v) * (len, k*b, 1)
            p_copy_scat_reshape = p_copy_scat.permute(1, 0, 2).contiguous().view(self.topk, -1, p_copy_scat.size(2)) # (len, k*b, v) -> (k*b, len, v) -> (k, len*b, v)
            att_prob_topk_reshape = att_prob_topk.permute(1, 0).repeat(1, p_copy_scat.size(0)).unsqueeze(2) # (k, len*b, 1)
            p_copy = torch.sum(p_copy_scat_reshape * att_prob_topk_reshape, dim=0) # weighted sum over k; (len*b, v)
            p_copy_reshape = p_copy.view(batch, nn_1hot.size(0), -1).permute(1, 0, 2) # (len, b, v)
            p_copy_sum = torch.sum(p_copy_reshape, dim=0) # sum over length; (b, v)

            # gated function
            h[-1] = self.gate(h[-1], features)

            h_gen[i] = h[-1]
            h_copy[i] = att_result_topk
            p_copies[i] = p_copy_sum

        # decoded: (len, batch, vocab)
        decoded = self.readout(h_gen.view(h_gen.size(0) * h_gen.size(1), h_gen.size(2))).view(h_gen.size(0), h_gen.size(1), self.readout.out_features)
        p_gens = F.softmax(decoded, dim=2)  # (len, b, v)

        # combine p_gen and p_copy
        copy_weight = self.copy_gate(h_gen, h_copy)
        prob_comb = copy_weight * p_copies + (1 - copy_weight) * p_gens
        result = torch.log(prob_comb)

        return result, (h, c)

    def get_nn_sim(self, vec, vec_diff):
        # compute attention scores Att(w_trg, w_nn): (1*b, k)
        vec_nn = vec.repeat(1, self.topk).view(-1, vec.size(1)) - vec_diff # (k*b, d)
        vec_nn_reshape = vec_nn.view(-1, self.topk, vec_nn.size(1)).permute(1, 0, 2)# (k, 1*b, d)
        att_score = self.attention_emb(vec, vec_nn_reshape, return_att_score=True) # (1*b, k)
        return att_score

    def get_nn_1hot(self, desc_nn, cuda):
        # (len, k*b) -> (len, k*b, vocab)
        if cuda:
            onehot = Variable(torch.FloatTensor(desc_nn.size(0), desc_nn.size(1), self.readout.out_features).zero_()).cuda()
        else:
            onehot = Variable(torch.FloatTensor(desc_nn.size(0), desc_nn.size(1), self.readout.out_features).zero_())
        onehot = onehot.scatter_(dim=2, index=desc_nn.unsqueeze(2), value=1.) # (len, k*b, vocab)
        return onehot

class EDIT_COPYNET(EDIT):
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, word2id, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul', coverage=True):
        super(EDIT_COPYNET, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, att_type=att_type)
        self.name = 'EDIT_COPYNET'
        self.topk = topk
        self.attention = Attention(self.dhid * 4, self.dhid * 2, self.dhid, type='add', init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.attention_topk = Attetion_Libovicky(self.dhid * 4, self.dhid, self.dhid, type=att_type,
                                                 init_range=init_range, init_dist=init_dist, dropout=dropout,
                                                 shared_U=False, Uc=True, topk=self.topk)
        self.copy_gate = Copy_Gate(self.dhid * 2, self.dhid)
        self.coverage = coverage
        if coverage:
            self.coverage_weight = nn.Linear(1, 1, bias=False)
            self.init_coverage(init_range, init_dist)

        self.word2id = word2id

        # add context vector as input of RNN
        if rnn_type in ['LSTM', 'GRU']:
            rnn = []
            for i in range(nlayers):
                if i == 0:
                    rnn.append(getattr(nn, rnn_type + 'Cell')(dinp + self.dhid, self.dhid)) # input: dim(embedding) + dim(context)
                else:
                    rnn.append(getattr(nn, rnn_type + 'Cell')(self.dhid, self.dhid))
            self.rnn = nn.ModuleList(rnn)

    def init_coverage(self, init_range, init_dist):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)
        init_w(self.coverage_weight.weight.data, init_dist)

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, mapper_cpy_to_cmb, Wh_enc, nn_sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None, return_att_weights=False):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        if return_att_weights:
            if cuda:
                att_weights = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).zero_()).cuda()
            else:
                att_weights = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).zero_())

        # build feature vector [vec; char_emb]
        features = vec
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        vocab_oov = mapper_cpy_to_cmb.size(2) - self.readout.out_features
        if cuda:
            prob_oov = Variable(torch.FloatTensor(batch, emb.size(0), vocab_oov).fill_(0)).cuda() # (b, trgLen, vocab_oov)
            prob_cmbs = Variable(torch.FloatTensor(emb.size(0), batch, mapper_cpy_to_cmb.size(2)).fill_(0)).cuda() # (trgLen, b, vocab_total)
            hs = Variable(torch.FloatTensor(emb.size(0), batch, h[-1].size(1)).fill_(0)).cuda() # (trgLen, b, d)
            context = Variable(torch.FloatTensor(batch, self.dhid).fill_(0)).cuda()
            contexts = Variable(torch.FloatTensor(emb.size(0), batch, self.dhid).fill_(0)).cuda() # (trgLen, b, d)
            # score_cpy = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).fill_(0)).cuda() # (trgLen, b, srcLen*k)
            probs_cpy = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).fill_(0)).cuda()  # (trgLen, b, srcLen*k)
            weights_cpy = Variable(torch.FloatTensor(emb.size(0), batch).fill_(0)).cuda() # (trgLen, b)

            # TODO: remove this and retrain models
            self.coverage = True
            if self.coverage:
                coverage_vector = Variable(torch.FloatTensor(batch, nn_emb.size(0) * self.topk).fill_(0)).cuda() # (b, srcLen*k)
        else:
            prob_oov = Variable(torch.FloatTensor(batch, emb.size(0), vocab_oov).fill_(0))
            prob_cmbs = Variable(torch.FloatTensor(emb.size(0), batch, mapper_cpy_to_cmb.size(2)).fill_(0))
            hs = Variable(torch.FloatTensor(emb.size(0), batch, h[-1].size(1)).fill_(0))
            context = Variable(torch.FloatTensor(batch, self.dhid).fill_(0))
            contexts = Variable(torch.FloatTensor(emb.size(0), batch, self.dhid).fill_(0))
            # score_cpy = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).fill_(0))
            probs_cpy = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).fill_(0))
            weights_cpy = Variable(torch.FloatTensor(emb.size(0), batch).fill_(0))  # (trgLen, b)
            if self.coverage:
                coverage_vector = Variable(torch.FloatTensor(batch, nn_emb.size(0) * self.topk).fill_(0))  # (b, srcLen*k)

        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(torch.cat([emb[i], context], dim=1), (h, c))
            # gated function
            h[-1] = self.gate(h[-1], features)
            hs[i] = h[-1]

            # attention over retrieved sentences w/ coverage penalty
            att_score = self.attention(torch.cat([h[-1], emb[i]], dim=1), nn_emb, desc_nn_mask, Wh_enc, topk=self.topk, return_att_score=True) # (k*b, srcLen)
            score_cpy = att_score.view(batch, -1)  # (b, srcLen*k)
            prob_cpy = F.softmax(score_cpy, dim=1)
            probs_cpy[i] = prob_cpy
            if self.coverage:
                coverage_penalty = self.coverage_weight(coverage_vector.view(-1, 1)).view(coverage_vector.size()) # (k*b, srcLen)
                score_cpy = score_cpy - coverage_penalty
            att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk, att_score=score_cpy.view(att_score.size()))  # (k*b, 2dim), (k*b, srcLen)

            # attention over k vectors
            att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, 2dim)
            vec_diff_reshape = vec_diff.view(batch, self.topk, -1).permute(1, 0, 2)  # (k, b, dim)
            att_topk_input = torch.cat([att_result_reshape, vec_diff_reshape, torch.abs(vec_diff_reshape)], dim=2)  # (k, b, 4d)
            context, att_prob_topk = self.attention_topk(h[-1], att_topk_input, cuda=cuda) # (b, d), (b, k)
            contexts[i] = context

            # weight of copy mode
            weight_cpy = self.copy_gate(context, h[-1]).squeeze(1) # (b)
            weights_cpy[i] = weight_cpy

            # update coverage vector
            if self.coverage:
                coverage_vector = coverage_vector + (weight_cpy.unsqueeze(1) * prob_cpy) # (b, srcLen*k)

            if return_att_weights:
                att_weights[i] = prob_cpy * weight_cpy.unsqueeze(1)  # (b, srcLen*k)

        # hs: (trgLen, b, d)
        score_gen = self.readout(hs.view(hs.size(0) * hs.size(1), hs.size(2)))\
            .view(hs.size(0), hs.size(1), -1).permute(1, 0, 2)  # (b, trgLen, vocab)

        prob_gen = F.softmax(score_gen, dim=2) # (b, trgLen, vocab_in)
        prob_gen_to_cmb = torch.cat([prob_gen, prob_oov], dim=2) # (b, trgLen, vocab_total)
        # prob_cpy = F.softmax(score_cpy.permute(1, 0, 2), dim=2) # (b, trgLen, srcLen*k)
        probs_cpy_reshape = probs_cpy.permute(1, 0, 2)

        # copy gate
        weights_cpy_reshape = weights_cpy.permute(1, 0).unsqueeze(2)  # (b, trgLen, 1)
        weights_gen = 1 - weights_cpy_reshape # (b, trgLen, 1)
        prob_cpy_to_cmbs = []
        for i in range(probs_cpy_reshape.size(1)):
            # (b, 1, srcLen*k) * (b, srclen*k, vocab_total) -> (b, 1, vocab_total)
            prob_cpy_to_cmbs.append(torch.bmm(probs_cpy_reshape[:, i, :].unsqueeze(1), mapper_cpy_to_cmb))
        prob_cpy_to_cmb = torch.cat(prob_cpy_to_cmbs, dim=1) # (b, trgLen, vocab_total)

        # debug; check nan
        # if math.isnan(prob_cpy_to_cmb.max().data[0]):
        #     print('nan in prob_cpy_to_cmb!!!!!!!!!!!!!!!!!!!!!!!!!!!', flush=True)
        #     print(prob_cpy_to_cmb, flush=True)

        if vocab_oov > 0: # if num of oov != 0
            prob_cmb = prob_gen_to_cmb * weights_gen + prob_cpy_to_cmb * weights_cpy_reshape
        else:
            prob_cmb = prob_gen

        prob_cmb = prob_cmb + 1e-40 # avoid some elements in prob_cmb from being zero
        result = torch.log(prob_cmb.permute(1, 0, 2)) # (trgLen, b, vocab_total)
        # debug: check -inf in result
        # if result.data.min() == float('-inf'):
        #     print('-inf in result!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', flush=True)
        #     print(result)

        if return_att_weights:
            return result, (h, c), att_weights
        return result, (h, c)

    def get_mapper(self, desc_nn, desc_nn_copy, cuda):
        # return a one-hot mapper that maps score_cpy (b, len*k) to score_all (b, vocab_total)
        # desc_nn_copy: (len, k*b)

        batch_size = int(desc_nn_copy.size(1) / self.topk)
        total_vocab_size = max(len(self.word2id), int(torch.max(desc_nn_copy) + 1))
        total_nn_length = int(desc_nn.size(0) * self.topk)
        if cuda:
            mapper = Variable(torch.FloatTensor(batch_size, total_nn_length, total_vocab_size).fill_(0)).cuda() # (b, len*k, total_vocab)
        else:
            mapper = Variable(torch.FloatTensor(batch_size, total_nn_length, total_vocab_size).fill_(0))

        desc_nn_copy_reshape = desc_nn_copy.permute(1, 0).contiguous().view(batch_size, -1).unsqueeze(2) # (b, len*k, 1)
        mapper.scatter_(2, desc_nn_copy_reshape, 1) # (b, len*k, vocab_total)

        return mapper

class EDIT_COPYNET_SIMPLE(EDIT_COPYNET):
    """Shared Uc in the 2nd attention, No coverage, No gated fucntion, linear copy gate"""
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, word2id, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul', non_copy=False, non_attention=False):
        super(EDIT_COPYNET_SIMPLE, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, word2id, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, topk=topk, att_type=att_type, coverage=False)
        self.copy_gate = Copy_Gate(self.dhid, 0)  # no hidden layer
        self.attention_topk = Attetion_Libovicky(self.dhid * 4, self.dhid * 2, self.dhid, type='add', init_range=init_range, init_dist=init_dist, dropout=dropout, shared_U=True, Uc=True, topk=self.topk)
        # self.readout = nn.Linear(self.dhid * 3, ntoken) # input: [hidden; y_prev; context]
        self.readout = nn.Linear(self.dhid, ntoken)  # input: hidden; stop using y_prev & context
        self.init_readout(init_range, init_dist)
        self.non_copy = non_copy
        self.non_attention = non_attention

    def init_readout(self, init_range, init_dist):
        if init_dist == 'uniform':
            self.readout.weight.data.uniform_(-init_range, init_range)
        elif init_dist == 'xavier':
            nn.init.xavier_uniform(self.readout.weight.data)
        self.readout.bias.data.fill_(0)

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, mapper_cpy_to_cmb, Wh_enc, nn_sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None, return_att_weights=False):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec; char_emb]
        if self.gated:
            features = vec
            if self.CH:
                features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        vocab_oov = mapper_cpy_to_cmb.size(2) - self.readout.out_features
        prob_oov = cudaw(Variable(torch.FloatTensor(batch, emb.size(0), vocab_oov).fill_(0))) # (b, trgLen, vocab_oov)
        hs = cudaw(Variable(torch.FloatTensor(emb.size(0), batch, h[-1].size(1)).fill_(0))) # (trgLen, b, d)
        context = cudaw(Variable(torch.FloatTensor(batch, self.dhid).fill_(0)))
        contexts = cudaw(Variable(torch.FloatTensor(emb.size(0), batch, self.dhid).fill_(0))) # (trgLen, b, 2dim)
        att_probs = cudaw(Variable(torch.FloatTensor(emb.size(0), self.topk * batch, nn_emb.size(0)).fill_(0))) # (trgLen, k*b, srcLen)
        att_probs_topk = cudaw(Variable(torch.FloatTensor(emb.size(0), batch, self.topk).fill_(0))) # (trgLen, b, k)

        nn_emb_reshape = nn_emb.permute(1, 0, 2).contiguous().view(batch, self.topk * nn_emb.size(0), nn_emb.size(2)) # (b, srcLen*k, 2dim)
        vec_diff_reshape = torch.cat([vec_diff.view(batch, self.topk, -1).permute(1, 0, 2), torch.abs(vec_diff).view(batch, self.topk, -1).permute(1, 0, 2)], dim=2)  # (k, b, 2d)
        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(torch.cat([emb[i], context], dim=1), (h, c))
            if self.gated:
                h[-1] = self.gate(h[-1], features)
            hs[i] = h[-1]

            # attention over sentences
            att_query = torch.cat([h[-1], emb[i]], dim=1)
            att_result, att_probs[i] = self.attention(att_query, nn_emb, desc_nn_mask, Wh_enc, topk=self.topk) # (k*b, 2d), (k*b, srcLen)

            # attention over k context vectors
            att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, 2d)
            att_topk_input = torch.cat([att_result_reshape, vec_diff_reshape], dim=2)  # (k, b, 4d)

            if not self.non_attention: # if self.non_attention, set context vector to zero
                context, att_probs_topk[i] = self.attention_topk(att_query, att_topk_input, cuda=cuda)  # (b, d), (b, k)
                contexts[i] = context

        # compute prob_gen
        # hs: (trgLen, b, d)

        # stop feeding previous word and context vector to readout layer
        # readout_input = torch.cat([hs, emb, contexts], dim=2) # (trgLen, b, 4d)
        # readout_input_reshape = readout_input.view(hs.size(0) * hs.size(1), readout_input.size(2)) # (??, 4d)
        readout_input_reshape = hs.view(hs.size(0) * hs.size(1), hs.size(2))

        score_gen = self.readout(readout_input_reshape) # (??, vocab)
        score_gen = score_gen.view(hs.size(0), hs.size(1), -1).permute(1, 0, 2) # (b, trgLen, vocab)
        prob_gen = F.softmax(score_gen, dim=2)

        # concat prob_gen with a vocab_oov dimensional vector
        if vocab_oov > 0:  # if num of oov != 0
            prob_gen = torch.cat([prob_gen, prob_oov], dim=2)  # (b, trgLen, vocab_total)

        # do not copy
        if self.non_copy:
            prob_cmb = prob_gen

        # compute prob_cpy
        else:
            att_probs_topk_reshape = att_probs_topk.view(att_probs_topk.size(0), -1).unsqueeze(2) # (trgLen, k*b, 1)
            prob_cpy = (att_probs * att_probs_topk_reshape) # (trgLen, k*b, srcLen)
            prob_cpy_reshape = prob_cpy.permute(1, 0, 2).contiguous().view(batch, self.topk, prob_cpy.size(0), prob_cpy.size(2)) # (b, k, trgLen, srcLen)
            prob_cpy_reshape = prob_cpy_reshape.permute(0, 2, 1, 3).contiguous().view(batch, prob_cpy.size(0), 1, -1).squeeze(2)  # (b, trgLen, srcLen*k)

            if return_att_weights:
                att_weights = prob_cpy_reshape.permute(1, 0, 2) # (trgLen, b, srcLen*k)

            # map prob_cpy to vocab_total dimensions
            prob_cpy_to_cmb = torch.bmm(prob_cpy_reshape, mapper_cpy_to_cmb) # (b, trgLen, vocab_total)

            # compute copy gate
            weights_cpy = self.copy_gate(hs).permute(1, 0, 2) # (b, trgLen, 1)
            weights_gen = 1 - weights_cpy  # (b, trgLen, 1)
            prob_cmb = (prob_gen * weights_gen) + (prob_cpy_to_cmb * weights_cpy)

        prob_cmb = prob_cmb + 1e-40  # avoid some elements in prob_cmb from being zero
        result = torch.log(prob_cmb.permute(1, 0, 2))  # (trgLen, b, vocab_total)

        if return_att_weights:
            return result, (h, c), att_weights
        return result, (h, c)

class EDIT_CAO(EDIT_COPYNET):
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, word2id, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul', coverage=True):
        super(EDIT_CAO, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, word2id, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, topk=topk, att_type=att_type, coverage=coverage)
        self.name = 'EDIT_CAO'
        self.readout = nn.Linear(self.dhid * 3, ntoken)  # input: [hidden; y_prev; context]
        self.init_readout(init_range, init_dist)

    def init_readout(self, init_range, init_dist):
        if init_dist == 'uniform':
            self.readout.weight.data.uniform_(-init_range, init_range)
        elif init_dist == 'xavier':
            nn.init.xavier_uniform(self.readout.weight.data)
        self.readout.bias.data.fill_(0)

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, mapper_cpy_to_cmb, Wh_enc, nn_sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None, return_att_weights=False):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        if return_att_weights:
            if cuda:
                att_weights = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).zero_()).cuda()
            else:
                att_weights = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).zero_())

        # build feature vector [vec; char_emb]
        features = vec
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        vocab_oov = mapper_cpy_to_cmb.size(2) - self.readout.out_features
        if cuda:
            prob_oov = Variable(
                torch.FloatTensor(batch, emb.size(0), vocab_oov).fill_(0)).cuda()  # (b, trgLen, vocab_oov)
            prob_cmbs = Variable(torch.FloatTensor(emb.size(0), batch, mapper_cpy_to_cmb.size(2)).fill_(
                0)).cuda()  # (trgLen, b, vocab_total)
            hs = Variable(torch.FloatTensor(emb.size(0), batch, h[-1].size(1)).fill_(0)).cuda()  # (trgLen, b, d)
            context = Variable(torch.FloatTensor(batch, self.dhid).fill_(0)).cuda()
            contexts = Variable(torch.FloatTensor(emb.size(0), batch, self.dhid).fill_(0)).cuda()  # (trgLen, b, d)
            # score_cpy = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).fill_(0)).cuda() # (trgLen, b, srcLen*k)
            probs_cpy = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).fill_(
                0)).cuda()  # (trgLen, b, srcLen*k)
            weights_cpy = Variable(torch.FloatTensor(emb.size(0), batch).fill_(0)).cuda()  # (trgLen, b)

            if self.coverage:
                coverage_vector = Variable(
                    torch.FloatTensor(batch, nn_emb.size(0) * self.topk).fill_(0)).cuda()  # (b, srcLen*k)
        else:
            prob_oov = Variable(torch.FloatTensor(batch, emb.size(0), vocab_oov).fill_(0))
            prob_cmbs = Variable(torch.FloatTensor(emb.size(0), batch, mapper_cpy_to_cmb.size(2)).fill_(0))
            hs = Variable(torch.FloatTensor(emb.size(0), batch, h[-1].size(1)).fill_(0))
            context = Variable(torch.FloatTensor(batch, self.dhid).fill_(0))
            contexts = Variable(torch.FloatTensor(emb.size(0), batch, self.dhid).fill_(0))
            # score_cpy = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).fill_(0))
            probs_cpy = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).fill_(0))
            weights_cpy = Variable(torch.FloatTensor(emb.size(0), batch).fill_(0))  # (trgLen, b)
            if self.coverage:
                coverage_vector = Variable(
                    torch.FloatTensor(batch, nn_emb.size(0) * self.topk).fill_(0))  # (b, srcLen*k)

        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(torch.cat([emb[i], context], dim=1), (h, c))
            # gated function
            h[-1] = self.gate(h[-1], features)
            hs[i] = h[-1]

            # attention over retrieved sentences w/ coverage penalty
            att_score = self.attention(torch.cat([h[-1], emb[i]], dim=1), nn_emb, desc_nn_mask, Wh_enc, topk=self.topk, return_att_score=True)  # (k*b, srcLen)
            score_cpy = att_score.view(batch, -1)  # (b, srcLen*k)
            prob_cpy = F.softmax(score_cpy, dim=1)
            probs_cpy[i] = prob_cpy
            if self.coverage:
                coverage_penalty = self.coverage_weight(coverage_vector.view(-1, 1)).view(
                    coverage_vector.size())  # (k*b, srcLen)
                score_cpy = score_cpy - coverage_penalty
            att_result, att_prob = self.attention(h[-1], nn_emb, desc_nn_mask, Wh_enc, topk=self.topk,
                                                  att_score=score_cpy.view(
                                                      att_score.size()))  # (k*b, 2dim), (k*b, srcLen)

            # attention over k vectors
            att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2)  # (k, b, 2dim)
            vec_diff_reshape = vec_diff.view(batch, self.topk, -1).permute(1, 0, 2)  # (k, b, dim)
            att_topk_input = torch.cat([att_result_reshape, vec_diff_reshape, torch.abs(vec_diff_reshape)],
                                       dim=2)  # (k, b, 4d)
            context, att_prob_topk = self.attention_topk(h[-1], att_topk_input, cuda=cuda)  # (b, d), (b, k)
            contexts[i] = context

            # weight of copy mode
            weight_cpy = self.copy_gate(context, h[-1]).squeeze(1)  # (b)
            weights_cpy[i] = weight_cpy

            # update coverage vector
            if self.coverage:
                coverage_vector = coverage_vector + (weight_cpy.unsqueeze(1) * prob_cpy)  # (b, srcLen*k)

            if return_att_weights:
                att_weights[i] = prob_cpy * weight_cpy.unsqueeze(1)  # (b, srcLen*k)

        # hs: (trgLen, b, d)
        readout_input = torch.cat([hs, emb, contexts], dim=2)  # (trgLen, b, 4d)
        readout_input_reshape = readout_input.view(hs.size(0) * hs.size(1), readout_input.size(2))  # (??, 4d)
        score_gen = self.readout(readout_input_reshape).view(hs.size(0), hs.size(1), -1).permute(1, 0, 2)  # (b, trgLen, vocab)

        prob_gen = F.softmax(score_gen, dim=2)  # (b, trgLen, vocab_in)
        if vocab_oov > 0:
            prob_gen_to_cmb = torch.cat([prob_gen, prob_oov], dim=2)  # (b, trgLen, vocab_total)
        else:
            prob_gen_to_cmb = prob_gen
        probs_cpy_reshape = probs_cpy.permute(1, 0, 2)

        # copy gate
        weights_cpy_reshape = weights_cpy.permute(1, 0).unsqueeze(2)  # (b, trgLen, 1)
        weights_gen = 1 - weights_cpy_reshape  # (b, trgLen, 1)
        prob_cpy_to_cmbs = []
        for i in range(probs_cpy_reshape.size(1)):
            # (b, 1, srcLen*k) * (b, srclen*k, vocab_total) -> (b, 1, vocab_total)
            prob_cpy_to_cmbs.append(torch.bmm(probs_cpy_reshape[:, i, :].unsqueeze(1), mapper_cpy_to_cmb))
        prob_cpy_to_cmb = torch.cat(prob_cpy_to_cmbs, dim=1)  # (b, trgLen, vocab_total)

        if vocab_oov > 0:  # if num of oov != 0
            prob_cmb = prob_gen_to_cmb * weights_gen + prob_cpy_to_cmb * weights_cpy_reshape
        else:
            prob_cmb = prob_gen

        prob_cmb = prob_cmb + 1e-40  # avoid some elements in prob_cmb from being zero
        result_gen = torch.log(prob_cmb.permute(1, 0, 2))  # (trgLen, b, vocab_total)
        result_cpy = torch.log(torch.cat([weights_gen, weights_cpy_reshape], dim=2).permute(0, 1, 2) + 1e-40) # (trgLen, b, 2)

        if return_att_weights:
            return result_gen, result_cpy, (h, c), att_weights
        return result_gen, result_cpy, (h, c)

class EDIT_CAO_SIMPLE(EDIT_COPYNET_SIMPLE):
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, word2id, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, gated=False, nchar=-1, max_char_len=-1, topk=-1, att_type='mul', coverage=True):
        super(EDIT_CAO_SIMPLE, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, word2id, dropout=dropout, tie_weights=tie_weights, init_range=init_range, init_dist=init_dist, seed_feeding=seed_feeding,  CH=CH, gated=gated, nchar=nchar, max_char_len=max_char_len, topk=topk, att_type=att_type)
        self.name = 'EDIT_CAO_SIMPLE'

    def forward(self, input, hidden, vec, vec_diff, nn_emb, desc_nn_mask, mapper_cpy_to_cmb, Wh_enc, nn_sim, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None, return_att_weights=False):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec; char_emb]
        if self.gated:
            features = vec
            if self.CH:
                features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        vocab_oov = mapper_cpy_to_cmb.size(2) - self.readout.out_features
        prob_oov = cudaw(Variable(torch.FloatTensor(batch, emb.size(0), vocab_oov).fill_(0)))  # (b, trgLen, vocab_oov)
        hs = cudaw(Variable(torch.FloatTensor(emb.size(0), batch, h[-1].size(1)).fill_(0)))  # (trgLen, b, d)
        context = cudaw(Variable(torch.FloatTensor(batch, self.dhid).fill_(0)))
        contexts = cudaw(Variable(torch.FloatTensor(emb.size(0), batch, self.dhid).fill_(0)))  # (trgLen, b, 2dim)
        att_probs = cudaw(Variable(
            torch.FloatTensor(emb.size(0), self.topk * batch, nn_emb.size(0)).fill_(0)))  # (trgLen, k*b, srcLen)
        att_probs_topk = cudaw(Variable(torch.FloatTensor(emb.size(0), batch, self.topk).fill_(0)))  # (trgLen, b, k)

        nn_emb_reshape = nn_emb.permute(1, 0, 2).contiguous().view(batch, self.topk * nn_emb.size(0),
                                                                   nn_emb.size(2))  # (b, srcLen*k, 2dim)
        vec_diff_reshape = torch.cat([vec_diff.view(batch, self.topk, -1).permute(1, 0, 2),
                                      torch.abs(vec_diff).view(batch, self.topk, -1).permute(1, 0, 2)],
                                     dim=2)  # (k, b, 2d)
        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(torch.cat([emb[i], context], dim=1), (h, c))
            if self.gated:
                h[-1] = self.gate(h[-1], features)
            hs[i] = h[-1]

            # attention over sentences
            att_query = torch.cat([h[-1], emb[i]], dim=1)
            att_result, att_probs[i] = self.attention(att_query, nn_emb, desc_nn_mask, Wh_enc,
                                                      topk=self.topk)  # (k*b, 2d), (k*b, srcLen)

            # attention over k context vectors
            att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2)  # (k, b, 2d)
            att_topk_input = torch.cat([att_result_reshape, vec_diff_reshape], dim=2)  # (k, b, 4d)
            context, att_probs_topk[i] = self.attention_topk(att_query, att_topk_input, cuda=cuda)  # (b, d), (b, k)
            contexts[i] = context

        # compute prob_gen
        # hs: (trgLen, b, d)
        # readout_input = torch.cat([hs, emb, contexts], dim=2)  # (trgLen, b, 4d)
        # readout_input_reshape = readout_input.view(hs.size(0) * hs.size(1), readout_input.size(2))  # (??, 4d)
        readout_input_reshape = hs.view(hs.size(0) * hs.size(1), hs.size(2))
        score_gen = self.readout(readout_input_reshape)  # (??, vocab)
        score_gen = score_gen.view(hs.size(0), hs.size(1), -1).permute(1, 0, 2)  # (b, trgLen, vocab)
        prob_gen = F.softmax(score_gen, dim=2)

        # compute prob_cpy
        att_probs_topk_reshape = att_probs_topk.view(att_probs_topk.size(0), -1).unsqueeze(2)  # (trgLen, k*b, 1)
        prob_cpy = (att_probs * att_probs_topk_reshape)  # (trgLen, k*b, srcLen)
        prob_cpy_reshape = prob_cpy.permute(1, 0, 2).contiguous().view(batch, self.topk, prob_cpy.size(0), prob_cpy.size(2))  # (b, k, trgLen, srcLen)
        prob_cpy_reshape = prob_cpy_reshape.permute(0, 2, 1, 3).contiguous().view(batch, prob_cpy.size(0), 1, -1).squeeze(2)  # (b, trgLen, srcLen*k)

        if return_att_weights:
            att_weights = prob_cpy_reshape.permute(1, 0, 2)  # (trgLen, b, srcLen*k)

        # concat prob_gen with a vocab_oov dimensional vector
        if vocab_oov > 0:  # if num of oov != 0
            prob_gen = torch.cat([prob_gen, prob_oov], dim=2)  # (b, trgLen, vocab_total)

        # map prob_cpy to vocab_total dimensions
        prob_cpy_to_cmb = torch.bmm(prob_cpy_reshape, mapper_cpy_to_cmb) # (b, trgLen, vocab_total)

        # compute copy gate
        weights_cpy = self.copy_gate(hs).permute(1, 0, 2)  # (b, trgLen, 1)
        weights_gen = 1 - weights_cpy  # (b, trgLen, 1)

        prob_cmb = (prob_gen * weights_gen) + (prob_cpy_to_cmb * weights_cpy)
        prob_cmb = prob_cmb + 1e-40  # avoid some elements in prob_cmb from being zero
        result = torch.log(prob_cmb.permute(1, 0, 2))  # (trgLen, b, vocab_total)
        result_cpy = torch.log(torch.cat([weights_gen, weights_cpy], dim=2).permute(0, 1, 2) + 1e-40) # (trgLen, b, 2)

        if return_att_weights:
            return result, result_cpy, (h, c), att_weights
        return result, result_cpy, (h, c)

class EG_HIERARCHICAL(EDIT):
    """Proposed model & Ni's model"""
    def __init__(self, args, corpus):
        self.feature_num = args.emsize + args.dhid  # v(word); bUc;
        super(EG_HIERARCHICAL, self).__init__(args.model, len(corpus.id2word), args.emsize, args.dhid, self.feature_num, args.nlayers, corpus.max_len, dropout=args.dropout, tie_weights=args.tied, init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding,  CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, att_type=args.att_type)
        self.name = 'EG_HIERARCHICAL'
        self.topk = 1 # TODO: Enable use multi examples
        self.attention_topk = Attetion_Libovicky(self.dhid * 2, self.dhid, self.dhid, type=args.att_type, init_range=args.init_range, init_dist=args.init_dist, dropout=args.dropout, shared_U=False, Uc=True, topk=1)
        # self.attention_emb = Attention(self.dhid, self.dhid, self.dhid, type=att_type, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.logsoftmax = nn.LogSoftmax()
        self.non_attention = args.non_attention

        if not args.gated: # Ni's model.
            if args.model in ['LSTM', 'GRU']: # feed attention vector to RNN at every time step
                rnn = []
                for i in range(args.nlayers):
                    if i == 0:
                        rnn.append(getattr(nn, args.model + 'Cell')(self.dhid * 2, self.dhid))
                    else:
                        rnn.append(getattr(nn, args.model + 'Cell')(self.dhid, self.dhid))
                self.rnn = nn.ModuleList(rnn)

            self.liniear_comb_hiddens = nn.Linear(self.dhid + 160, self.dhid) # Combine char emb and context encoder states
            self.readout = nn.Linear(self.dhid * 2, len(corpus.id2word)) # feed attention vector to readout state
            self.init_weights_ni()

    def init_weights_ni(self, init_range=0.1, init_dist='uniform'):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)

        init_w(self.liniear_comb_hiddens.weight.data, init_dist)
        init_w(self.readout.weight.data, init_dist)

        self.liniear_comb_hiddens.bias.data.fill_(0)
        self.readout.bias.data.fill_(0)

    def forward(self, input, hidden, vec, eg_emb, eg_mask, Wh_enc, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None, return_att_weights=False):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # build feature vector [vec; char_emb]
        if self.gated:
            features = vec
            if self.CH:
                features = torch.cat([features, char_emb], dim=1)

        # decode words
        h, c = hidden
        if self.gated:
            outputs = cudaw(Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()))
        else: #Ni's model. input attention vector to readout layer
            outputs = cudaw(Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1) * 2).zero_()))


        if return_att_weights:
            if cuda:
                att_weights = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).zero_()).cuda()
            else:
                att_weights = Variable(torch.FloatTensor(emb.size(0), batch, nn_emb.size(0) * self.topk).zero_())

        att_result_topk = cudaw(Variable(torch.FloatTensor(batch, emb.size(2)).zero_()))
        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            if not self.gated: # Ni's model
                h, c = self.stacked_rnn(torch.cat((emb[i], att_result_topk), dim=1), (h, c))
            else:
                h, c = self.stacked_rnn(emb[i], (h, c))

            if not self.non_attention:
                # attention over source words
                att_result, att_prob = self.attention(h[-1], eg_emb, eg_mask, Wh_enc, topk=self.topk) # (k*b, 2dim)

                # attention over k vectors
                att_result_reshape = att_result.view(batch, self.topk, -1).permute(1, 0, 2) # (k, b, 2dim); Note that att_result.view(k, b, 2dim) is wrong
                att_result_topk, att_prob_topk = self.attention_topk(h[-1], att_result_reshape, cuda=cuda)

            # gated function
            if self.gated:
                h[-1] = self.gate(h[-1], torch.cat((features, att_result_topk), dim=1))
                outputs[i] = h[-1]

            else: # Ni's model
                h[-1] = self.liniear_comb_hiddens(torch.cat((h[-1], char_emb), dim=1))
                outputs[i] = torch.cat((h[-1], att_result_topk), dim=1) 

            # save attention weights (Jan. 22)
            if return_att_weights:
                att_weight = att_prob * att_prob_topk.view(1, -1).permute(1,0) # (k*b, srcLen)
                att_weights[i] = att_weight.view(batch, -1) # (b, srcLen*k)

        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)

        if return_att_weights:
            return decoded, (h, c), att_weights
        else:
            return decoded, (h, c)

class EG_GADETSKY(EDIT):
    """﻿Conditional Generators of Words Definitions Artyom [﻿Gadetsky+ ACL18]"""
    def __init__(self, args, corpus):
        self.feature_num = args.emsize  # v(word); bUc;
        if args.char:
            self.feature_num += 160
        super(EG_GADETSKY, self).__init__(args.model, len(corpus.id2word), args.emsize, args.dhid, self.feature_num, args.nlayers, corpus.max_len, dropout=args.dropout, tie_weights=args.tied, init_range=args.init_range, init_dist=args.init_dist, seed_feeding=args.seed_feeding,  CH=args.char, gated=args.gated, nchar=len(corpus.id2char), max_char_len=corpus.max_char_len, att_type=args.att_type)
        self.name = 'EG_GADETSKY'
        self.topk = 1 # TODO: Enable use multi examples
        self.logsoftmax = nn.LogSoftmax()
        self.get_mask = Attention_Gadetsky(args.dhid * 2, args.dhid, args.emsize)

        # Gadetsky's model.
        if args.model in ['LSTM', 'GRU']: # feed masked word embedding (and char embedding) to RNN at every time step
            rnn = []
            for i in range(args.nlayers):
                if i == 0:
                    rnn.append(getattr(nn, args.model + 'Cell')(self.dhid + self.feature_num, self.dhid))
                else:
                    rnn.append(getattr(nn, args.model + 'Cell')(self.dhid, self.dhid))
            self.rnn = nn.ModuleList(rnn)

    def forward(self, input, hidden, vec, eg_emb, eg_mask, Wh_enc, cuda, seed_feeding=True, char_emb=None, batch_ensemble=False, p=None, return_att_weights=False):
        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((vec.unsqueeze(0), self.encoder(input))))
        else:
            emb = self.drop(self.encoder(input))
        batch = emb.size(1)

        # decode words
        h, c = hidden
        outputs = cudaw(Variable(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_()))

        # change eg_mask into a binary mask
        eg_mask[(eg_mask == 0)] = 1
        eg_mask[(eg_mask != 1)] = 0
        mask = self.get_mask(eg_emb, eg_mask)

        features = mask * vec  # masked vector
        if self.CH:
            features = torch.cat([features, char_emb], dim=1)

        for i in range(emb.size(0)):
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(torch.cat((features, emb[i]), dim=1), (h, c))
            outputs[i] = h[-1]

        # decoded: (len, batch, vocab)
        decoded = self.readout(outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))).view(outputs.size(0), outputs.size(1), self.readout.out_features)
        return decoded, (h, c)

#!/usr/bin/env bash
# Script to reproduce the BLEU scores reported in [Noraset+ AAAI2017]

SENTENCE_BLEU=./sentence-bleu # https://github.com/moses-smt/mosesdecoder/blob/master/mert/sentence-bleu.cpp
REF=./data/wiki/test.txt # change this if you store the data in different directories
MODEL=./model/wiki_proposed

#################################################################################
cat $REF | cut -f1,4 > $REF.tmp
cat $MODEL/train.log | python3 log2out.py > $MODEL/test.out
python3 norbleu.py --ignore_sense_id --bleu_path $SENTENCE_BLEU --ref $REF.tmp --hyp $MODEL/test.out > $MODEL/test.res
cat $MODEL/test.res

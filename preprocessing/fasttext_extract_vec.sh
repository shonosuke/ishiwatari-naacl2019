#!/usr/bin/env bash

FASTTEXT=/disk/ishiwatari/downloads/fastText-0.1.0/fasttext
MODEL=/disk/ishiwatari/downloads/fasttext.wiki.en/wiki.en.bin
DATA=/disk/ishiwatari/defgen/onto_nora/

# extract all words
#cat $DATA/train.txt | cut -d"%" -f1 | sort | uniq > $DATA/hoge1.txt
#cat $DATA/valid.txt | cut -d"%" -f1 | sort | uniq >> $DATA/hoge1.txt
#cat $DATA/test.txt | cut -d"%" -f1 | sort  | uniq >> $DATA/hoge1.txt
#cat $DATA/train.txt | cut -f4 | sed -e 's/ /\n/g' | sort | uniq > $DATA/hoge2.txt
#cat $DATA/hoge1.txt $DATA/hoge2.txt | sort | uniq > $DATA/hoge.txt

# use exactly the same words as word2vec to make word vectors with fasttext
cat /disk/ishiwatari/defgen/onto_nora/filtered.vec | cut -d" " -f1 | sort | uniq | $FASTTEXT print-word-vectors $MODEL > $DATA/ft_filtered.vec

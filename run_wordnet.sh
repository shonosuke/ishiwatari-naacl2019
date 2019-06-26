#!/usr/bin/env bash
# Experiments on Wordnet

DATA_DIR=./data/wordnet # change this if you store the data in different directories
MODEL_DIR=./model
SRC_DIR=.
VEC=$DATA_DIR/filtered.vec
VEC_TYPE=w2v
BATCH_SIZE=1024 # When using Quadro P6000 with 24GB memory
mkdir $MODEL_DIR

#################################################################################
MODEL=$MODEL_DIR/wordnet_proposed; mkdir $MODEL
env CUDA_VISIBLE_DEVICES=0 python3 $SRC_DIR/main.py --data $DATA_DIR --vocab_size 10000 --tied --vec $VEC --batch_size $BATCH_SIZE --cuda --save $MODEL/model.pt --init_embedding --mode EG_HIERARCHICAL  --seed_feeding --gated --char --data_usage 100 > $MODEL/train.log &

MODEL=$MODEL_DIR/wordnet_ni; mkdir $MODEL
env CUDA_VISIBLE_DEVICES=1 python3 $SRC_DIR/main.py --data $DATA_DIR --vocab_size 10000 --tied --vec $VEC --batch_size $BATCH_SIZE --cuda --save $MODEL/model.pt --init_embedding --mode EG_HIERARCHICAL --char --data_usage 100 > $MODEL/train.log &

MODEL=$MODEL_DIR/wordnet_noraset; mkdir $MODEL
env CUDA_VISIBLE_DEVICES=2 python3 $SRC_DIR/main.py --ignore_sense_id --data $DATA_DIR --vocab_size 10000 --tied --vec $VEC --batch_size $BATCH_SIZE --cuda --save $MODEL/model.pt --init_embedding --mode NORASET --seed_feeding --gated --char --data_usage 100 > $MODEL/train.log &

MODEL=$MODEL_DIR/wordnet_gadetsky; mkdir $MODEL
env CUDA_VISIBLE_DEVICES=3 python3 $SRC_DIR/main.py --data $DATA_DIR --vocab_size 10000 --tied --vec $VEC --batch_size $BATCH_SIZE --cuda --save $MODEL/model.pt --init_embedding --mode EG_GADETSKY --seed_feeding --char --data_usage 100 > $MODEL/train.log &
wait

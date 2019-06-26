#!/usr/bin/env bash


##################################################################################
## Try cheat and max_vocab data settings
##################################################################################
#DISKROOT=/disk/ishiwatari
#MODEL_DIR=$DISKROOT/defgen/model
#DATA=$DISKROOT/defgen/onto_nora
#BATCH_SIZE=256
#W2V=$DATA/filtered.vec

## NOTE! DO NOT forget to add --ignore_sense_id option when running with word2vec!
#MODEL=$MODEL_DIR/onto_w2v_d50_v13679_copynet_k2
#mkdir $MODEL/att_vis
#gpu_select 0 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda --att_vis > ${MODEL}/test.out  2> ${MODEL}/test.res
#

#MODEL=$MODEL_DIR/maxvoc_w2v_d20_v13679_copynet_k5
#mkdir $MODEL/att_vis
#gpu_select 2 python3 test.py --data $DATA/max_vocab/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda --att_vis > ${MODEL}/test.out  2> ${MODEL}/test.res &
#
#MODEL=$MODEL_DIR/cheat_w2v_d20_v13679_copynet_k5
#mkdir $MODEL/att_vis
#gpu_select 5 python3 test.py --data $DATA/cheat_def/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda --att_vis > ${MODEL}/test.out  2> ${MODEL}/test.res &
#wait

#MODEL=$MODEL_DIR/onto_w2v_d2_v13679_sgch
#gpu_select 0 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &
#
#MODEL=$MODEL_DIR/onto_w2v_d10_v13679_sgch
#gpu_select 1 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &
#
#MODEL=$MODEL_DIR/onto_w2v_d20_v13679_sgch
#gpu_select 2 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &
#
#MODEL=$MODEL_DIR/onto_w2v_d50_v13679_sgch
#gpu_select 3 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &
#
#MODEL=$MODEL_DIR/onto_w2v_d70_v13679_sgch
#gpu_select 6 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &
#
#wait
#
#MODEL=$MODEL_DIR/onto_w2v_d5_v13679_sgch
#gpu_select 6 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &
#wait

#MODEL=$MODEL_DIR/cheat_w2v_d100_v13679_hierar_k5
#mkdir $MODEL_DIR/att_vis
#gpu_select 0 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda --att_vis > ${MODEL}/test.out  2> ${MODEL}/test.res &
#
#MODEL=$MODEL_DIR/cheat_w2v_d100_v13679_hierar_k2
#mkdir $MODEL_DIR/att_vis
#gpu_select 1 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda --att_vis > ${MODEL}/test.out  2> ${MODEL}/test.res &
#
#MODEL=$MODEL_DIR/cheat_w2v_d100_v13679_hierar_k1
#mkdir $MODEL_DIR/att_vis
#gpu_select 2 python3 test.py --data $DATA/w2v --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda --att_vis > ${MODEL}/test.out  2> ${MODEL}/test.res &
#wait

#MODEL=$MODEL_DIR/onto_w2v_d100_v13679_sgch
#gpu_select 4 python3 test.py --data $DATA/w2v --gen --show_logloss --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &
#
#MODEL=$MODEL_DIR/onto_w2v_d100_v13679_copynet_k5
#mkdir $MODEL/att_vis
#gpu_select 5 python3 test.py --data $DATA/w2v --gen --att_vis --show_logloss --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &

#MODEL=$MODEL_DIR/onto_w2v_d100_v13679_copynet_cao_k5
#mkdir $MODEL/att_vis
#gpu_select 2 python3 test.py --data $DATA/w2v --gen --att_vis --show_logloss --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &


#MODEL=$MODEL_DIR/strong10000_w2v_d100_v13679_cao_simple_withNora_do05
##DATA=$MODEL_DIR/rank_rnn1_mlp1_6feats_lr000001
#gpu_select 5 python3 test.py --data $DATA/strong_cheat/w2v_10000 --gen --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test.out  2> ${MODEL}/test.res &

#MODEL=$MODEL_DIR/strong10000_w2v_d100_v13679_cao_simple_withNora_do05
#DATA=$MODEL_DIR/br_mlp1_lr00001_b1024_filter26
#gpu_select 5 python3 test.py --data $DATA --gen --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test_br.out  2> ${MODEL}/test_br.res &

##################################################################################
# Vs. human annotation

DISKROOT=/disk/ishiwatari
MODEL_DIR=$DISKROOT/defgen/model
DATA=$DISKROOT/defgen/onto_nora
BATCH_SIZE=256
W2V=$DATA/filtered.vec

# vs. human annotation
MODEL=$MODEL_DIR/oxford_noraset_d100_v10000_sgch
DATA=$MODEL_DIR/
gpu_select 5 python3 test_lm.py --data $DATA --gen --vec $W2V --ignore_sense_id --model ${MODEL}/model.pt --batch_size $BATCH_SIZE --cuda > ${MODEL}/test_br.out  2> ${MODEL}/test_br.res &


wait
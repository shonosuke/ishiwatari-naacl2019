#!/usr/bin/env sh
##################################################################################
### Experiments with Noraset's data
#DATA_DIR=/data/ishiwatari/defgen/onto_nora
#W2V=/data/ishiwatari/defgen/vec/onto_nora/GoogleNews-vectors-negative300.txt
#
#python2 preprocessing/senseretrofit.py -v $W2V -q $DATA_DIR/onto.txt
#python3 preprocessing/filter_vec.py $W2V $W2V.sense $DATA_DIR $DATA_DIR/filtered.vec
#mkdir $DATA_DIR/retro $DATA_DIR/w2v
#cp $DATA_DIR/*.txt $DATA_DIR/retro
#cp $DATA_DIR/*.txt $DATA_DIR/w2v
#
## For each data, extract nearest neighbors from train data
#python3 preprocessing/search_nn.py --data $DATA_DIR/w2v --vec $W2V --ignore_sense_id --K 30 &
#python3 preprocessing/search_nn.py --data $DATA_DIR/retro --vec $W2V.sense --K 30 &
#wait

##################################################################################
## [temp!] Experiments with val_test ontological information
#DATA_DIR=/data/ishiwatari/defgen/onto_nora/with_val_test_onto
#W2V=/data/ishiwatari/defgen/vec/onto_nora/GoogleNews-vectors-negative300.txt
#
#python2 preprocessing/senseretrofit.py -v $W2V -q $DATA_DIR/onto.txt
#python3 preprocessing/filter_vec.py $W2V $W2V.sense $DATA_DIR $DATA_DIR/filtered.vec
#mkdir $DATA_DIR/retro $DATA_DIR/w2v
#cp $DATA_DIR/*.txt $DATA_DIR/retro
#cp $DATA_DIR/*.txt $DATA_DIR/w2v
#
## For each data, extract nearest neighbors from train data
#python3 preprocessing/search_nn.py --data $DATA_DIR/w2v --vec $W2V --ignore_sense_id --K 30 &
#python3 preprocessing/search_nn.py --data $DATA_DIR/retro --vec $W2V.sense --K 30 &
#wait

#################################################################################
### [temp!] Experiments with mini data
#DATA_DIR=/data/ishiwatari/defgen/onto_nora/mini
#W2V=/data/ishiwatari/defgen/onto_nora/mini/GoogleNews-vectors-negative300.txt
#
#python2 preprocessing/senseretrofit.py -v $W2V -q $DATA_DIR/onto.txt
#python3 preprocessing/filter_vec.py $W2V $W2V.sense $DATA_DIR $DATA_DIR/filtered.vec
#mkdir $DATA_DIR/retro $DATA_DIR/w2v
#cp $DATA_DIR/*.txt $DATA_DIR/retro
#cp $DATA_DIR/*.txt $DATA_DIR/w2v
#
## For each data, extract nearest neighbors from train data
#python3 preprocessing/search_nn.py --data $DATA_DIR/w2v --vec $W2V --ignore_sense_id --K 6 &
#python3 preprocessing/search_nn.py --data $DATA_DIR/retro --vec $W2V.sense --K 6 &
#wait

#################################################################################
## [temp!] Experiments with refined cheating definition selection models
#DATA_DIR=/data/ishiwatari/defgen/onto_nora
#W2V=/data/ishiwatari/defgen/vec/onto_nora/GoogleNews-vectors-negative300.txt

#python2 preprocessing/senseretrofit.py -v $W2V -q $DATA_DIR/onto.txt
#python3 preprocessing/filter_vec.py $W2V $W2V.sense $DATA_DIR $DATA_DIR/filtered.vec

#mkdir $DATA_DIR/cheat_def $DATA_DIR/max_vocab $DATA_DIR/strong_cheat
#mkdir $DATA_DIR/cheat_def/retro $DATA_DIR/cheat_def/w2v
#mkdir $DATA_DIR/max_vocab/retro $DATA_DIR/max_vocab/w2v
#mkdir $DATA_DIR/strong_cheat/retro $DATA_DIR/strong_cheat/w2v_all $DATA_DIR/strong_cheat/w2v_3000 $DATA_DIR/strong_cheat/w2v_10000

#cp $DATA_DIR/filtered.vec $DATA_DIR/cheat_def/ &
#cp $DATA_DIR/*.txt $DATA_DIR/cheat_def/ &
#cp $DATA_DIR/*.txt $DATA_DIR/cheat_def/retro &
#cp $DATA_DIR/*.txt $DATA_DIR/cheat_def/w2v &
#wait
#cp $DATA_DIR/filtered.vec $DATA_DIR/max_vocab/ &
#cp $DATA_DIR/*.txt $DATA_DIR/max_vocab/ &
#cp $DATA_DIR/*.txt $DATA_DIR/max_vocab/retro &
#cp $DATA_DIR/*.txt $DATA_DIR/max_vocab/w2v &
#wait
#cp $DATA_DIR/filtered.vec $DATA_DIR/strong_cheat/ &
#cp $DATA_DIR/*.txt $DATA_DIR/strong_cheat/ &
#cp $DATA_DIR/*.txt $DATA_DIR/strong_cheat/retro &
#cp $DATA_DIR/*.txt $DATA_DIR/strong_cheat/w2v_all &
#cp $DATA_DIR/*.txt $DATA_DIR/strong_cheat/w2v_3000 &
#cp $DATA_DIR/*.txt $DATA_DIR/strong_cheat/w2v_10000 &
#wait


## For each data, extract nearest neighbors from train data
#python3 preprocessing/search_nn.py --mode cheat --data $DATA_DIR/cheat_def/w2v --vec $W2V --ignore_sense_id --K 30 &
#python3 preprocessing/search_nn.py --mode cheat --data $DATA_DIR/cheat_def/retro --vec $W2V.sense --K 30 &
#wait
#
#python3 preprocessing/search_nn.py --mode max_vocab --data $DATA_DIR/max_vocab/w2v --vec $W2V --ignore_sense_id --K 30 &
#python3 preprocessing/search_nn.py --mode max_vocab --data $DATA_DIR/max_vocab/retro --vec $W2V.sense --K 30 &
#wait

#python3 preprocessing/search_nn.py --mode strong_cheat --data $DATA_DIR/strong_cheat/w2v_all   --L -1    --vec $W2V --ignore_sense_id --K 50 &
#python3 preprocessing/search_nn.py --mode strong_cheat --data $DATA_DIR/strong_cheat/w2v_3000  --L 3000  --vec $W2V --ignore_sense_id --K 50 &
#python3 preprocessing/search_nn.py --mode strong_cheat --data $DATA_DIR/strong_cheat/w2v_10000 --L 10000 --vec $W2V --ignore_sense_id --K 50 &
#python3 preprocessing/search_nn.py --mode strong_cheat --data $DATA_DIR/strong_cheat/retro --vec $W2V.sense --K 30
wait

###################################################################################
### [temp!] Experiments with Fasttext (on t102)
#DATA_DIR=/disk/ishiwatari/defgen/onto_nora
#FAST=/disk/ishiwatari/defgen/onto_nora/ft_filtered.vec
#
###python2 preprocessing/senseretrofit.py -v $FAST -q $DATA_DIR/onto.txt
##mkdir $DATA_DIR/ft
##cp $DATA_DIR/*.txt $DATA_DIR/ft
#preprocessing/fasttext_extract_vec.sh
#
### For each data, extract nearest neighbors from train data
#python3 preprocessing/search_nn.py --data $DATA_DIR/ft --vec $DATA_DIR/ft_filtered.vec --ignore_sense_id --K 20
#
## make strong cheat data
#mkdir $DATA_DIR/strong_cheat/ft_10000
##cp $DATA_DIR/*.txt $DATA_DIR/strong_cheat/ft_10000
#python3 preprocessing/search_nn.py --mode strong_cheat --data $DATA_DIR/strong_cheat/ft_10000 --L 10000 --vec $FAST --ignore_sense_id --K 20


##################################################################################
### [temp!] Experiments with mini data and fasttext vectors
#DATA_DIR=/home/ishiwatari/src/defgen_pt/data/onto_nora/mini
#VEC=/disk/ishiwatari/defgen/onto_nora/ft_filtered.vec
#
#mkdir $DATA_DIR/ft
#cp $DATA_DIR/*.txt $DATA_DIR/ft
#
## For each data, extract nearest neighbors from train data
#python3 preprocessing/search_nn.py --data $DATA_DIR/ft --vec $VEC --ignore_sense_id --K 6
#wait

##################################################################################
## [temp!] Experiments with example sentences (instead of nearest neighbors)
#DATA_DIR=/disk/ishiwatari/defgen/onto_nora
#VEC=/disk/ishiwatari/defgen/onto_nora/filtered.vec
#CORE_NLP=/disk/ishiwatari/downloads/stanford-corenlp-full-2018-02-27
#
#mkdir $DATA_DIR/eg $DATA_DIR/eg_tag
#for file in train valid test; do
#    python3 preprocessing/extract_example.py --data $DATA_DIR/w2v/${file}.txt --out $DATA_DIR/eg/ --corenlp_path $CORE_NLP &
#    python3 preprocessing/extract_example.py --data $DATA_DIR/w2v/${file}.txt --out $DATA_DIR/eg_tag/ --corenlp_path $CORE_NLP --replace_target_word &
#done; wait

##################################################################################
## extract data from wikidata
#DATA_DIR=/disk/ishiwatari/defgen/wikidata
#VEC=/disk/ishiwatari/defgen/vec/GoogleNews-vectors-negative300.txt
#
#python3 preprocessing/extract_wikidata.py --data $DATA_DIR/merged.jsonlines --vec $VEC --out $DATA_DIR
# extract data from oxford dictionary

DATA_DIR=/disk/ishiwatari/defgen/oxford_dic
VEC=/disk/ishiwatari/defgen/vec/GoogleNews-vectors-negative300.txt
python3 preprocessing/extract_oxford_dic.py --data $DATA_DIR/orig --vec $VEC --out $DATA_DIR
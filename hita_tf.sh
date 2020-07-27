#!/bin/bash

#mkdir -p model

model=model
corpus=corpus

SL=hi
TL=ta
pref=spinning-storage/rameshak


#echo 'starting model training'
#echo 'started preprocessing the data '
python3 /home/rameshak/$pref/software/OpenNMT-py/preprocess.py -train_src $corpus/train.$SL -train_tgt $corpus/train.$TL -valid_src $corpus/devset.$SL -valid_tgt corpus/devset.$TL \
-save_data corpus/model_data \
--src_vocab_size 2000 \
--tgt_vocab_size 2000

#echo 'preprocessed data saved in corpus/model_data'
echo 'starting to train the data'
python3 /home/rameshak/$pref/software/OpenNMT-py/train.py -data corpus/model_data -save_model $model/lstm  \
-train_from $model/lstm_step_160000.pt \
-layers 6 -rnn_size 512 -word_vec_size 500 -transformer_ff 2048 -heads 8  \
-encoder_type transformer -decoder_type transformer -position_encoding \
-train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
-batch_size 1000 -batch_type tokens -normalization tokens  -accum_count 2 \
-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.0005  \
-max_grad_norm 0 -param_init 0  -param_init_glorot \
-label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
-world_size 2 \
-gpu_ranks 0 1 

echo 'model saved'

#!/bin/bash

pref=spinning-storage/rameshak
opennmt=/home/rameshak/$pref/software/OpenNMT-py
model=model
SL=en
TL=ta
#we have 4 test sets: kde4, tz, ubuntu, software.

#nvidia-smi 

evaluation=evaluation

#mkdir $evaluation -p

tes=corpus/test_sw.$SL   
ref=corpus/test_sw.$TL

# Decoding (translation)
#python3 $opennmt/translate.py -model /home/rameshak/spinning-storage/rameshak/hita_tf/model/lstm_step_200000.pt -src $tes -output $evaluation/pred_kde.txt -replace_unk -verbose

#Evaluation
#/home/rameshak/spinning-storage/rameshak/software/scripts/multeval/multeval.sh eval --refs $ref --hyps-baseline $evaluation/pred_kde.txt --meteor.language en > $evaluation/results_kde
#Evaluation -- measuring bleu
/home/rameshak/$pref/software/scripts/generic/multi-bleu.perl $ref < $evaluation/test_google.ta > $evaluation/results_kde4

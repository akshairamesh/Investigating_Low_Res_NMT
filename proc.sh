#!/bin/bash

SL=hi
TL=ta

#we have 4 test sets: kde4, tz, ubuntu, wikimatrix.


pref=spinning-storage/rameshak
~/$pref/software/subword_nmt/learn_bpe.py --input corpus/train.$SL -s 32000 -o corpus/bpe.$SL.32k &
~/$pref/software/subword_nmt/learn_bpe.py --input corpus/train.$TL -s 32000 -o corpus/bpe.$TL.32k &
wait

~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$SL.32k < corpus/train.$SL > corpus/train.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$TL.32k < corpus/train.$TL > corpus/train.BPE.$TL &

~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$SL.32k < corpus/testset_kde4.$SL > corpus/testset_kde4.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$TL.32k < corpus/testset_kde4.$TL > corpus/testset_kde4.BPE.$TL &


~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$SL.32k < corpus/testset_tz.$SL > corpus/testset_tz.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$TL.32k < corpus/testset_tz.$TL > corpus/testset_tz.BPE.$TL &


~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$SL.32k < corpus/testset_ub.$SL > corpus/testset_ub.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$TL.32k < corpus/testset_ub.$TL > corpus/testset_ub.BPE.$TL &


~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$SL.32k < corpus/testset_wiki.$SL > corpus/testset_wiki.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$TL.32k < corpus/testset_wiki.$TL > corpus/testset_wiki.BPE.$TL &

~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$SL.32k < corpus/devset.$SL > corpus/devset.BPE.$SL &
~/$pref/software/subword_nmt/apply_bpe.py -c corpus/bpe.$TL.32k < corpus/devset.$TL > corpus/devset.BPE.$TL &

wait
echo 'Hi, Proc done...'



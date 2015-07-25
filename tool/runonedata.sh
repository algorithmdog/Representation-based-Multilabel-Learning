#!/bin/bash
mkdir -p result
mkdir -p experiment
echo "" > result/$1.result
for((i=0;i<1;i++));do
python train.py  -batch 1 -niter 2 -sample_ratio 5  -num_factor $2  ~/multi_label_data/$1/svm_cv/$i/train-$i.arff experiment/$1.model.$i > train.log
echo "" > experiment/$1.result.arff
python predict.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff experiment/$1.result.arff experiment/$1.model.$i
python eval.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff experiment/$1.result.arff >> result/$1.result

done

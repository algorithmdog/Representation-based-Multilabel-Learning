#!/bin/bash
mkdir -p result
mkdir -p experiment
echo "" > result/$1.result
for((i=0;i<1;i++));do
python train.py -m 1 -b 1 -n 2 -t instance_sample  -sample_ratio 5  -num_factor $2  -sparse_thr 1 ~/multi_label_data/$1/svm_cv/$i/train-$i.arff experiment/$1.model.$i > train.log
echo "" > experiment/$1.result.arff
python predict.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff experiment/$1.result.arff experiment/$1.model.$i
python eval.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff experiment/$1.result.arff >> result/$1.result

done

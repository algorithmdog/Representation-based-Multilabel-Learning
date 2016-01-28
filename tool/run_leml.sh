#!/bin/bash
mkdir -p result
mkdir -p result
name=leml_$1_l20.1_i10_h$2

echo "" > result/$name.result
echo "" > result/$name.log

for((i=0;i<1;i++));do
python train_leml.py -l2 0.1 -i 10 -h $2  ~/multi_label_data/$1/svm_cv/$i/train-$i.arff result/$1.model.$i 2>> result/$name.log
python predict.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff result/$1.result.arff result/$1.model.$i   2>> result/$name.log
echo "\n" >> result/$name.log
python eval.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff result/$1.result.arff >> result/$name.result

done

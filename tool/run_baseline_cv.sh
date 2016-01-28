#!/bin/bash
mkdir -p result
mkdir -p result
name=baseline_$1

echo "" > result/$name.result
echo "" > result/$name.log

for((i=0;i<5;i++));do
python train_baseline.py  ~/multi_label_data/$1/svm_cv/$i/train-$i.arff result/$1.model.$i 2>> result/$name.log
python predict.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff result/$1.result.arff result/$1.model.$i     2>> result/$name.log
echo "" >> result/$name.log
python eval.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff result/$1.result.arff >> result/$name.result

done

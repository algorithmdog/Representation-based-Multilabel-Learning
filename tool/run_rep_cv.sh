#!/bin/bash
mkdir -p result
mkdir -p result
name=rep_$1_b1_i2_sr5_ha0_l20.001_h$2

echo "" > result/$name.result
echo "" > result/$name.log

for((i=0;i<5;i++));do
python train_rep.py  -b 1 -i 2 -sr 5 -ha 0 -l2 0.001 -h $2  ~/multi_label_data/$1/svm_cv/$i/train-$i.arff result/$1.model.$i 2>> result/$name.log
python predict.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff result/$1.result.arff result/$1.model.$i            2>> result/$name.log
echo "\n" >> result/$name.log
python eval.py ~/multi_label_data/$1/svm_cv/$i/test-$i.arff result/$1.result.arff >> result/$name.result

done

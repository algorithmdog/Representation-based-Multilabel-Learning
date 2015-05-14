#!/bin/bash
mkdir -p result
echo "" > result/$1.result
for((i=0;i<5;i++));do
python train.py -m 1 -n 20 -t instance_sample  ~/multi_label_data/$1/arff_cv/$i/train-$i.arff experiment/$1.model.$i
python predict.py ~/multi_label_data/$1/arff_cv/$i/test-$i.arff experiment/$1.result.arff experiment/$1.model.$i
python eval.py ~/multi_label_data/$1/arff_cv/$i/test-$i.arff experiment/$1.result.arff >> result/$1.result

done

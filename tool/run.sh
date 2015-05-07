#!/bin/bash
mkdir -p result
echo "" > result/$1.result
for((i=0;i<5;i++));do
python train.py -m 1 -n 20 -t full  ~/multi_label_data/$1/arff_cv/$i/train-$i.arff experiment/model
python predict.py ~/multi_label_data/$1/arff_cv/$i/test-$i.arff experiment/result.arff experiment/model
python eval.py ~/multi_label_data/$1/arff_cv/$i/test-$i.arff experiment/result.arff >> result/$1.result

done

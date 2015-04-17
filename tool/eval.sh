#!/bin/bash
mkdir data

python train.py -n 20 ~/multi_label_data/$1/$1-train-trans.arff data/model
python predict.py ~/multi_label_data/$1/$1-test-trans.arff data/result.arff data/model
python eval.py ~/multi_label_data/$1/$1-test-trans.arff data/result.arff >> data/eval_result

======================
rustle1314/latent_factor_for_multi_label
======================
This project provides a latent factor model for multi-label classification, even the number of labels is extremely large. 

Data Format
------------
The first line of the training data file is "#num_feature={$1} num_label={$2}", where $1 denotes the number of features and $2 denotes the number of labels. For example, if we have 20 features and 6 labels, the first line shall be "#num_feature=20 num_label=6".
 
The following lines denote training data. Each line denotes an instance. '1,2 1:4.5 2:3' denotes an instance with the 2-th and 3-th label. The weight of the 1-th feature is 4.5 and the weight of the 2-th feature is 3.
The index of feature and label starts with zero.

Here is an example::

    #num_feature=20 num_label=6
    1,2 1:4.5 2:3
    2,4 19:1 2:5
    1,5 10:2 8:1


Train, Predict and Eval
-----------------------
1.1 Train

You can use train.py to train a model for a multi-label classification with many labels problem

Usage: python train.py [options] train_file model_file

Options::

- l2_lambda: the l2 regularization coefficient (default 0.001)
- struct: the architecture of instance represnation learner: [num_node_layer1,num_node_layer2,...] (default [])
- batch: batch, the number of instances in a batch (default 100)
- niter: num of iter, the number of iterations (default 20)
- num_factor: the number of inner factors (default 50)
- sample_ratio: the ratio of sampling (default 5)

For example, "python train.py -l2_lambda 0.1 train.txt model" will train a model with regularization coefficient 0.1. The training set file is ./train.txt and the trained model will be saved in ./model.


1.2 Predict

You can use predict.py to make predictions with a trained model.

Usage: python predict.py test_file result_file model_file

1.3 Eval

You can use eval.py to evaluate the predictions

Usage: python eval.py result_file true_file

Requirements
---------------
Now I only test the code with Python2.7 and I will test it with Python3 as soon as possible.

To run this code, you should installed the following modules::

- numpy
- scipy


Contributors
------------
- `lietal <https://github.com/rustle1314>`_i

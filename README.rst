======================
rustle1314/Representation-based-Multilabel-Learning
======================
This project provides representation-based learning methods for multi-label classification, even the number of labels is extremely large, include:

* Web Scale Annotation by Image Embedding (WSABIE)

  Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.

* Low rank Empirical risk minimization for Multi-Label Learning (LEML)
 
  Yu, Hsiang-Fu, et al. "Large-scale multi-label learning with missing labels." arXiv preprint arXiv:1307.5101 (2013).

* Representation-based Multi-label Learning with Sampling (RMLS)

  Li et al. "Towards Label Imbalance in Multi-label Classification with Many Labels"

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

You can use train_wsabie.py and train_leml.py to train a model for a multi-label classification with many labels problem. Usage: 

  Usage: python train_wsabie.py(or train_leml.py) [options] train_file model_file

If you don't know how to set up the options, you just need "python train_wsabie.py" and the program will give detailed information.


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

# KGE

The Knowledge Graph Embedding (KGE) repository is a implementation of the state of the art techniques related to Statistical Relational Learning (SRL) to solve Link Prediction problems. The techniques implemented in this code include TransE, DistMult, Canonical Polyadic and ComplEx.

## Installation

The code depends on Downhill and Theano packages. Install it, along with other dependencies with:

```
pip install downhill theano
```

## Example of use

The simplest way to generate and evaluate the models is calling the `run.py` script. The `model` parameter is the techniques available, `data` is the name of the dataset to be executed, `k` is the dimension of the embedding vectors, `epoch` is the number of epochs to be executed and `folds` is the number of folds used in the k-fold cross-validation technique. The simplest way to execute the KGE techniques is:

```
python run.py --model Complex --data bicycleparking --k 100 --epoch 1000 --folds 5
```

The code can be executed in the x86 or using GPUs. For execute the code using GPUs you need to add the flag `DEVICE=cuda0` before call the execution line. Addtional parameters can be fitted: `lmbda` is the lambda value, `lr` is the learning rate, `bsize` is the number of examples in the batch and `negative` is the number of negative samples used. 

## Add more data

To run the techniques using your own data, you need to add the file in the subfolder `datasets` using csv formation and the triples separated by tabs with extension `.txt`. After that, you only need to execute the techniques with the name of the dataset.

## Developer notes

To submit bugs and feature requests, report at [project issues](https://github.com/QROWD/KGE/issues).

## References


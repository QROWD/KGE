# KGE

The Knowledge Graph Embedding (KGE) repository is a implementation of the state of the art techniques related to Statistical Relational Learning to solve Link Prediction problems.

## Installation

The code depends on Downhill and Theano packages. Install it, along with other dependencies with:

```
pip install downhill theano

```

## Example of use

The simplest way to execute the KGE techniques is:

```
python run.py --model Complex --data wn18 --k 100 --epoch 1000 --folds 5

```

## Add your own data

To run the techniques using your own data, you need to add the dataset in the subfolder datasets using csv formation and the triples separated by tabs. After that, you only need to execute the techniques with the name of the dataset.

## Developer notes

To submit bugs and feature requests, report at [project issues](https://github.com/QROWD/KGE/issues).

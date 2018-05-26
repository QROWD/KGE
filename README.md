# KGE

The Knowledge Graph Embedding (KGE) repository is an implementation of the state of the art techniques related to Statistical Relational Learning (SRL) to solve Link Prediction problems. These techniques map the structure of large knowledge graphs on models able to predict missing relationships in new triples [1-2]. The techniques implemented in this code include TransE, DistMult, RESCAL and ComplEx.

## Technical requirements

The system was develop in python 2.7. The code depends on rdflib, downhill and theano [3] packages. Install it, along with other dependencies with:

```
pip install rdflib downhill theano
```

## Example of use

The simplest way to generate and evaluate the models is calling the `run.py` script. The `model` parameter is the techniques available, the `data` is the full path of the dataset to be executed, the `k` is the dimension of the embedding vectors, the `epoch` is the number of epochs to be executed and the `folds` is the number of folds used in the k-fold cross-validation technique. The simplest way to execute the KGE techniques is:

```
python run.py evaluation --model Complex --data /tmp/wn18k.txt --k 200 --epoch 1000 --folds 5
```

The code can be executed in the x86 or using GPUs. To execute the code using GPUs you need to add the flag `DEVICE=cuda0` before calling the execution line. Additional parameters can be fitted: `lmbda` is the lambda value, `lr` is the learning rate, `bsize` is the number of examples in the batch, `negative` is the number of negative samples used and `folds` the number of folds in the cross-validation technique. 

The output is the main information about the dataset, the stochastic gradient descent error and the average performance of the models for each fold. The performance is a vector where the values represent the evaluation metrics like the Mean Reciprocal Rank (MRR) and Hits at N, with N = {1, 3, 10}. The output is similar to that:

```python
  MRR	  H@1	  H@3	 H@10
0.714	0.618	0.805	0.825
0.713	0.620	0.803	0.821
0.715	0.622	0.805	0.826
0.710	0.615	0.802	0.823
0.713	0.619	0.803	0.823
```

The best model generated after the cross-validation execution will be exported in the main folder with the name `model.txt`. To evaluate a new data, you should call the prediction function with the full path of the test data:

```
python run.py prediction --data /tmp/wn18k.txt
```
The output is a table called `out.csv` with the associated probability of each triple.

## Add more data

To run the techniques using your own data, you need to add the file in the subfolder `datasets` using csv or rdf format. In the csv format, the triples need to separated by tabs. After that, you only need to execute the techniques with the name of the dataset.

## Developer notes

To submit bugs and feature requests, report at [project issues](https://github.com/QROWD/KGE/issues).

## References

[1] Maximilian Nickel, Kevin Murphy, Volker Tresp and Evgeniy Gabrilovich. (2016). A review of relational machine learning for knowledge graphs. Proceedings of the IEEE, 104(1):11-33.

[2] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Eric Gaussier and Guillaume Bouchard. (2016). Complex embeddings for simple link prediction. In International Conference on Machine Learning (ICML), 48:2071-2080.

[3] Theano Development Team. (2017). Theano: A Python framework for fast computation of mathematical expressions.


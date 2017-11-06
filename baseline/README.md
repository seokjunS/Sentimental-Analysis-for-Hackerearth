# Baseline

## Installation

### packages

```
$ pip install jupyter
$ pip install lightbgm
$ pip install cat boost
or
$ pip install -r requirements.txt # TODO
```

*TODO:* add other dependencies(or requirements.txt)
 
## Test

```
$ cp ../data/*.csv .
$ mkdir submissions
```

then Run all cells

## Experiments & Results

### Evaluation Metrics

- Accuracy(%)

### Models

- randomly select happy or unhappy **0.49845**
- Using `Naive Bayes`
	- Bag of Words **0.76297**
	- TF-IDF **0.80592**
- Using `lightgbm`
    - Basic **0.86562**
    - with TF-IDF **0.85931**
- Using `catboost`
    - basic **TODO**


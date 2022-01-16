# Jigsaw-Toxic-Comments-Severity-Using-ML
Jigsaw Toxic Comments Severity Using ML 

## Introduction
- A high performance model developed to rate the toxicity of comments.
- Extensively Fine tuned state of the art machine learning models. 
- Extensively done Data Processing, Ingestion, Feature Engineering.
- Text Embedding by using Fast-Text
- Ensemble models trained on text clusters.


## Requirements

- Python [3.7 or above] - https://www.python.org/downloads/
- Git - https://git-scm.com/download/
- Pipenv [Python package for virtual envs] - Install it using `pip install pipenv` 
- sklearn 
- pandas 
- numpy 
- gensim

## Project and code structure information
- src
  - data_ingestion :- data loading and preparing 
  - data_processing :- cleaning text by applying lemmetization on text, removing stop words and special characters
  - word_embeddings :- word embedding using Fast-Text
  - eda_src :- showing word cloud, check the skewness and kurtosis of data
  - clustering :- applied text clustering
  - model_dev :- different machine learning models with fine tuning 
  - evaluation :- used R2-score, mean-squared-error, root-mean-sqaure-error as metrics to evaluate the models.

## Download Data and trained models
[Data](https://onedrive.live.com/?authkey=%21AA6AFKx74NJJZpU&id=DAF5585B7D5A73FE%21261&cid=DAF5585B7D5A73FE)

## How to run pipeline
```
python main.py
```

## 🙌 Show your support

Be sure to leave a ⭐️ if you like the project!

## Contributors
- [Ayush Singh](https://github.com/ayush714) 
- [Kirti Goyal](https://github.com/Kirti1807)

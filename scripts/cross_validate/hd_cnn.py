# CNN CrossValidator

import numpy as np
from nltk import TweetTokenizer

from scripts.utils import scratch_path
from tklearn.datasets import load_fdcl18, load_dwmw17
from tklearn.model_selection import CrossValidator
from tklearn.neural_network import NeuralNetClassifier
from tklearn.neural_network.model import TextCNN
from tklearn.preprocessing.tweet import TweetPreprocessor
from tklearn.text.word_vec import load_word2vec
from tklearn.utils import pprint

DATASET = 'FDCL18'

if __name__ == '__main__':
    # Load Dataset and Extract Features
    if DATASET.lower().startswith('f'):
        df = load_fdcl18(num_classes=2)
        pprint({'dataset': 'FDCL18(num_classes=2)'})
    else:
        df = load_dwmw17(num_classes=2)
        pprint({'dataset': 'DWMW17(num_classes=2)'})
    df['clean_tweets'] = df.tweet.apply(TweetPreprocessor(normalize=['link', 'mention']).preprocess)
    df['tokens'] = df.clean_tweets.apply(TweetTokenizer().tokenize)
    # Load Resources
    word2vec = load_word2vec()
    # Hyperparameters
    kwargs = {
        'model': 'multichannel',
        'epoch': 100,
        'learning_rate': 0.01,
        'max_sent_len': 50,
        'batch_size': 50,
        # 'word_dim': 300,
        'filters': [3, 4, 5],
        'filter_num': [100, 100, 100],
        'dropout_prob': 0.5,
        'norm_limit': 3,
    }
    pprint(kwargs)
    # """ # Additional Parameters
    kwargs['module'] = TextCNN
    kwargs['corpus'] = df.tokens
    kwargs['word_vectors'] = word2vec
    # """ # Cross-validate and save predictions
    scorers = ['accuracy', 'precision', 'recall', 'f1']
    estimator = NeuralNetClassifier(**kwargs)
    cv = CrossValidator(NeuralNetClassifier, kwargs, n_splits=5, scoring=scorers)
    df['predictions'], cv_results = cv.cross_val_predict(df.tokens, df.label, return_scores=True)
    # """ Print Scores
    pprint(cv_results)
    scores = {}
    for scorer in scorers:
        scores[scorer] = ['%.2f' % (np.average(cv_results[scorer]) * 100) + ',']
    pprint(scores, type='table')
    # """ Save Predictions #
    df.to_excel(scratch_path('cnn_predictions.xlsx'))
    # """ #

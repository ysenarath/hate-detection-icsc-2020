# CNN Train-Validate-Test

from nltk import TweetTokenizer

from scripts.utils import scratch_path
from tklearn.datasets import load_dataset
from tklearn.matrices.scorer import get_score_func
from tklearn.model_selection import train_dev_test_split
from tklearn.neural_network import NeuralNetClassifier
from tklearn.neural_network.model import TextCNN
from tklearn.preprocessing.tweet import TweetPreprocessor
from tklearn.text.word_vec import load_word2vec
from tklearn.utils import pprint


def run(dataset, hyperparameters, metrics, fname=None):
    # # Load Resources
    word2vec = None
    if hyperparameters['model'] != 'rand':
        word2vec = load_word2vec()
    # # Load Dataset
    df = load_dataset(dataset[0], **dataset[1])
    # # Preprocess
    df['clean_tweets'] = df.tweet.apply(TweetPreprocessor(normalize=['link', 'mention']).preprocess)
    df['tokens'] = df.clean_tweets.apply(TweetTokenizer().tokenize)
    X_train, X_dev, X_test, y_train, y_dev, y_test = train_dev_test_split(df.tokens, df.label)
    # # Train
    clf = NeuralNetClassifier(
        module=TextCNN, corpus=df.tokens, word_vectors=word2vec, metrics=metrics, **hyperparameters
    )
    clf.fit(X_train, y_train, validation_data=(X_dev, y_dev))
    # # Predict
    y_pred = clf.predict(X_test)
    # # Evaluate
    pprint(dict(
        dataset=dataset,
        hyperparameters=hyperparameters,
        scores={scorer: get_score_func(scorer)(y_test, y_pred) for scorer in metrics}
    ))
    # # Save to file
    X_test['pred'] = y_pred
    X_test.to_excel(scratch_path('predictions_%s.xlsx' % fname))


if __name__ == '__main__':
    run(
        dataset=('FDCL18', dict(num_classes=2)),
        hyperparameters={
            'model': 'multichannel',
            'epoch': 35,
            'learning_rate': 0.01,
            'max_sent_len': 50,
            'batch_size': 512,
            # 'word_dim': 300,
            'filters': [3, 4, 5],
            'filter_num': [100, 100, 100],
            'dropout_prob': 0.5,
            'norm_limit': 3,
        },
        metrics=['accuracy', 'precision', 'recall', 'f1']
    )

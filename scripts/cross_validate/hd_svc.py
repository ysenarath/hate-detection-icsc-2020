import nltk
import numpy as np
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

from scripts.utils import scratch_path
from tklearn.datasets import load_dwmw17, load_fdcl18
from tklearn.feature_extraction.embedding import mean_embedding
from tklearn.feature_extraction.hatespeech import HatebaseVectorizer
from tklearn.feature_extraction.transfer_learning import TransferVectorizer
from tklearn.model_selection import CrossValidator
from tklearn.neural_network import NeuralNetClassifier
from tklearn.neural_network.model import TextCNN
from tklearn.preprocessing.tweet import TweetPreprocessor
from tklearn.text.word_vec import load_numberbatch
from tklearn.utils import get_logger, pprint

nltk.download('stopwords')

stopwords = set(stopwords.words('english'))

logger = get_logger(__name__)


def run(dataset, features, word_embedding, metrics, fname):
    if dataset.lower().startswith('f'):
        df = load_fdcl18()
    else:
        df = load_dwmw17()
    tqdm.pandas(desc='Preprocessing Progress: ')
    df['clean_tweet'] = df.tweet.progress_apply(TweetPreprocessor(normalize=['link', 'mention']).preprocess, )
    tqdm.pandas(desc='Tokenizing Progress: ')
    df['tokens'] = df.clean_tweet.progress_apply(TweetTokenizer().tokenize)
    # #
    # Feature Extraction
    # tfidf_pipeline
    ff = []
    if 'tfidf_vectorizer' in features:
        tfidf_kwargs = dict(
            tokenizer=TweetTokenizer().tokenize,
            stop_words=stopwords,
            min_df=.0025,
            max_df=0.25,
            ngram_range=(1, 3)
        )
        ff += [('tfidf_vectorizer', TfidfVectorizer(**tfidf_kwargs), 'clean_tweet')]
    # framenet_pipeline
    if 'framenet_pipeline' in features:
        count_vectorizer = ('count_vectorizer', CountVectorizer())
        truncated_svd = ('truncated_svd', TruncatedSVD(algorithm='randomized', n_components=10))
        ff += [('framenet_pipeline', Pipeline([count_vectorizer, truncated_svd]), 'framenet')]
    # mean_embedding
    if 'mean_embedding' in features:
        ff += [('mean_embedding', mean_embedding(word_embedding), 'tokens')]
    # hatebase_vectorizer
    if 'hatebase_vectorizer' in features:
        ff += [('hatebase_vectorizer', HatebaseVectorizer(features=features['hatebase_vectorizer']), 'clean_tweet')]
    # transfer_vectorizer
    if 'transfer_vectorizer' in features:
        hyper_params = features['transfer_vectorizer']
        hyper_params['module'] = TextCNN
        hyper_params['corpus'] = df.tokens
        hyper_params['word_vectors'] = word_embedding
        # """ # Cross-validate and save predictions
        args = [NeuralNetClassifier, hyper_params, ['conv_%i' % i for i in range(3)], False]
        ff += [('transfer_vectorizer', TransferVectorizer(*args), 'tokens')]
    # # Estimator
    pipeline = Pipeline([('column_transformer', ColumnTransformer(ff)), ('clf', LinearSVC())])
    # # Evaluation (Cross Validation)
    # """ # Cross-validate and save predictions
    cv = CrossValidator(pipeline, n_splits=5, scoring=metrics)
    df['predictions'], cv_results = cv.cross_val_predict(df, df.label, return_scores=True)
    # """ Print Scores
    pprint({'dataset': dataset, 'features': features})
    pprint(cv_results)
    scores = {}
    for scorer in metrics:
        scores[scorer] = ['%.2f' % (np.average(cv_results[scorer]) * 100) + ',']
    pprint(scores, type='table')
    # """ Save Predictions #
    df.to_excel(scratch_path('predictions_%s_%s.xlsx' % (dataset, fname)))


def run_all(datasets, features, metrics):
    # Load static objects
    # word_embedding = load_word2vec()
    weights = load_numberbatch()
    for dataset in datasets:
        feature_dict = {}
        for fid0, feature_map in enumerate(features):
            for fid1, args in enumerate(feature_map[1]):
                feature_dict.update({feature_map[0]: args})
                run(dataset=dataset, features=feature_dict, word_embedding=weights, metrics=metrics,
                    fname=str(fid0) + '_' + str(fid1))


if __name__ == '__main__':
    run_all(
        datasets=['fdcl18', 'dwmw17'],
        features=[
            ('tfidf_vectorizer', [None]),
            ('hatebase_vectorizer', [
                ['average_offensiveness'],
                ['average_offensiveness', 'is_unambiguous'],
                ['average_offensiveness', 'is_unambiguous', 'hateful_meaning'],
                ['average_offensiveness', 'is_unambiguous', 'hateful_meaning', 'nonhateful_meaning'],
            ]),
            ('framenet_pipeline', [None]),
            ('mean_embedding', [None]),
            # ('transfer_vectorizer', [
            #     {
            #         'model': 'static',
            #         'epoch': 10,
            #         'learning_rate': 0.01,
            #         'max_sent_len': 50,
            #         'batch_size': 50,
            #         # 'word_dim': 300,
            #         'filters': [3, 4, 5],
            #         'filter_num': [100, 100, 100],
            #         'dropout_prob': 0.5,
            #         'norm_limit': 3,
            #     }
            # ]),
        ],
        metrics=['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro',
                 'f1_macro'],
    )

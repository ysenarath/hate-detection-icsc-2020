import nltk
import numpy as np
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

from scripts.utils import scratch_path
from tklearn.datasets import load_dwmw17, load_fdcl18
from tklearn.feature_extraction.embedding import mean_embedding
from tklearn.feature_extraction.hatespeech import HatebaseVectorizer
from tklearn.feature_extraction.transfer_learning import TransferVectorizer
from tklearn.model_selection import CrossValidator
from tklearn.neural_network import NeuralNetClassifier, BaseEstimator
from tklearn.neural_network.model import TextCNN
from tklearn.preprocessing.tweet import TweetPreprocessor
from tklearn.utils import get_logger, pprint

nltk.download('stopwords')

stopwords = set(stopwords.words('english'))

logger = get_logger(__name__)


class CCAFusion(TransformerMixin, BaseEstimator):
    def __init__(self, c1, c2):
        self.pipes = [c1, c2]
        self.max_iter = 500
        self.cca = None

    def fit(self, X, y=None, **fit_params):
        C = []
        n_components = None
        for pipe in self.pipes:
            c = pipe.fit_transform(X, y)
            if hasattr(c, 'toarray'):
                c = c.toarray()
            if n_components is None:
                n_components = c.shape[1]
            else:
                n_components = min(c.shape[1], n_components)
            C += [c]
        self.cca = CCA(n_components=n_components, max_iter=self.max_iter)
        self.cca.fit(*C)
        return self

    def transform(self, X, y=None):
        C = []
        for pipe in self.pipes:
            c = pipe.transform(X, y)
            if hasattr(c, 'toarray'):
                c = c.toarray()
            C += [c]
        return self.cca.transform(*C)[0]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)


def run(dataset, features, word2vec, metrics, fname):
    if dataset.lower().startswith('f'):
        df = load_fdcl18(num_classes=2)
    else:
        df = load_dwmw17(num_classes=2)
    tqdm.pandas(desc='Preprocessing Progress: ')
    df['clean_tweet'] = df.tweet.progress_apply(TweetPreprocessor(normalize=['link', 'mention']).preprocess, )
    tqdm.pandas(desc='Tokenizing Progress: ')
    df['tokens'] = df.clean_tweet.progress_apply(TweetTokenizer().tokenize)
    # #
    # Feature Extraction
    # tfidf_pipeline
    ff1 = []
    ff2 = []
    if 'tfidf_vectorizer' in features:
        tfidf_kwargs = dict(
            tokenizer=TweetTokenizer().tokenize,
            stop_words=stopwords,
            min_df=.0025,
            max_df=0.25,
            ngram_range=(1, 3),
            max_features=5000,
        )
        ff1 += [('tfidf_vectorizer', TfidfVectorizer(**tfidf_kwargs), 'clean_tweet')]
    # framenet_pipeline
    if 'framenet_pipeline' in features:
        count_vectorizer = ('count_vectorizer', CountVectorizer())
        truncated_svd = ('truncated_svd', TruncatedSVD(algorithm='randomized', n_components=10))
        ff2 += [('framenet_pipeline', Pipeline([count_vectorizer, truncated_svd]), 'framenet')]
    # mean_embedding
    if 'mean_embedding' in features:
        ff1 += [('mean_embedding', mean_embedding(word2vec), 'tokens')]
    # hatebase_vectorizer
    if 'hatebase_vectorizer' in features:
        ff2 += [('hatebase_vectorizer', HatebaseVectorizer(features=features['hatebase_vectorizer']), 'clean_tweet')]
    # transfer_vectorizer
    if 'transfer_vectorizer' in features:
        hyper_params = features['transfer_vectorizer']
        hyper_params['module'] = TextCNN
        hyper_params['corpus'] = df.tokens
        hyper_params['word_vectors'] = word2vec
        # """ # Cross-validate and save predictions
        args = [NeuralNetClassifier, hyper_params, ['conv_%i' % i for i in range(3)], False]
        ff2 += [('transfer_vectorizer', TransferVectorizer(*args), 'tokens')]
    # # Estimator
    pipeline = Pipeline(
        [('column_transformer', CCAFusion(ColumnTransformer(ff1), ColumnTransformer(ff2))), ('clf', LinearSVC())])
    # # Grid Search``
    # param_grid = [
    #     {'clf__C': [0.1, 1, 10, 50], 'classifier': linear_svc},
    #     # {'classifier': sgd_classifier},
    # ]
    # gs = GridSearchCV(pipeline, param_grid, cv=5)
    # result = gs.fit(df, df.label).predict(df)
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


if __name__ == '__main__':
    run(
        dataset='dwmw17',
        features={
            'tfidf_vectorizer': [None],
            'hatebase_vectorizer': ['average_offensiveness', 'is_unambiguous', 'hateful_meaning', 'nonhateful_meaning']
        },
        # word2vec=load_word2vec(),
        word2vec=None,
        metrics=['accuracy', 'precision', 'recall', 'f1'],
        fname="cca_pred.exls"
    )

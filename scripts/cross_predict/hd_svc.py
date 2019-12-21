import nltk
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

from tklearn.datasets import load_dwmw17, load_fdcl18
from tklearn.feature_extraction.embedding import mean_embedding
from tklearn.feature_extraction.hatespeech import HatebaseVectorizer
from tklearn.feature_extraction.transfer_learning import TransferVectorizer
from tklearn.neural_network import NeuralNetClassifier, get_score_func
from tklearn.neural_network.model import TextCNN
from tklearn.preprocessing.tweet import TweetPreprocessor
from tklearn.text.word_vec import load_word2vec
from tklearn.utils import get_logger, pprint

nltk.download('stopwords')

stopwords = set(stopwords.words('english'))

logger = get_logger(__name__)


def run(dataset, features, word2vec, metrics, fname=None):
    if dataset == 'fdcl18':
        df1 = load_fdcl18(num_classes=2)
        df2 = load_dwmw17(num_classes=2)
        df2 = df2.drop(['count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
    else:
        df1 = load_dwmw17(num_classes=2)
        df2 = load_fdcl18(num_classes=2)
        df1 = df1.drop(['count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
    # # Preprocessing
    preprocess = TweetPreprocessor(normalize=['link', 'mention']).preprocess
    tokenize = TweetTokenizer().tokenize
    # # # DF 1 - Preprocessing
    tqdm.pandas(desc='Preprocessing Progress: ')
    df1['clean_tweet'] = df1.tweet.progress_apply(preprocess)
    tqdm.pandas(desc='Tokenizing Progress: ')
    df1['tokens'] = df1.clean_tweet.progress_apply(tokenize)
    # # # DF 2 - Preprocessing
    tqdm.pandas(desc='Preprocessing Progress: ')
    df2['clean_tweet'] = df2.tweet.progress_apply(preprocess)
    tqdm.pandas(desc='Tokenizing Progress: ')
    df2['tokens'] = df2.clean_tweet.progress_apply(tokenize)
    # #
    # # Feature Extraction
    # # # tfidf_pipeline
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
    # # # framenet_pipeline
    if 'framenet_pipeline' in features:
        count_vectorizer = ('count_vectorizer', CountVectorizer())
        truncated_svd = ('truncated_svd', TruncatedSVD(algorithm='randomized', n_components=10))
        ff += [('framenet_pipeline', Pipeline([count_vectorizer, truncated_svd]), 'framenet')]
    # # # mean_embedding
    if 'mean_embedding' in features:
        ff += [('mean_embedding', mean_embedding(word2vec), 'tokens')]
    # # # hatebase_vectorizer
    if 'hatebase_vectorizer' in features:
        ff += [('hatebase_vectorizer', HatebaseVectorizer(features=features['hatebase_vectorizer']), 'clean_tweet')]
    # # # transfer_vectorizer
    if 'transfer_vectorizer' in features:
        hyper_params = features['transfer_vectorizer']
        hyper_params['module'] = TextCNN
        hyper_params['corpus'] = df1.tokens
        hyper_params['word_vectors'] = word2vec
        # """ # Cross-validate and save predictions
        args = [NeuralNetClassifier, hyper_params, ['conv_%i' % i for i in range(3)], False]
        ff += [('transfer_vectorizer', TransferVectorizer(*args), 'tokens')]
    # # # estimator
    pipeline = Pipeline([('column_transformer', ColumnTransformer(ff)), ('clf', LinearSVC())])
    # # Grid Search``
    # param_grid = [
    #     {'clf__C': [0.1, 1, 10, 50], 'classifier': linear_svc},
    #     # {'classifier': sgd_classifier},
    # ]
    # gs = GridSearchCV(pipeline, param_grid, cv=5)
    # result = gs.fit(df, df.label).predict(df)
    # # Evaluation
    pipeline.fit(df1, df1.label)
    y_true, y_pred = df2.label, pipeline.predict(df2)
    # df2['predictions'] = y_pred
    # """ Print Scores
    pprint({'dataset': dataset, 'features': features})
    scores = {}
    for scorer in metrics:
        scores[scorer] = [get_score_func(scorer)(y_true, y_pred)]
    pprint(scores, type='table')
    # """ Save Predictions #
    # fname
    # df2.to_excel(scratch_path('predictions_%s_%s.xlsx' % (dataset, fname)))


def run_all(**kwargs):
    word2vec = load_word2vec()
    print("Individual Feature Comparison")
    for dataset in kwargs['DATASETS']:
        for features in kwargs['FEATURES']:
            run(
                dataset=dataset,
                features=features,
                metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                word2vec=word2vec,
            )


if __name__ == '__main__':
    run_all(
        DATASETS=['dwmw17', 'fdcl18'],
        FEATURES=[
            {
                'tfidf_vectorizer': None,
            },
            {
                'tfidf_vectorizer': None,
                'hatebase_vectorizer': ['average_offensiveness', 'is_unambiguous', 'hateful_meaning',
                                        'nonhateful_meaning'],
                'framenet_pipeline': None,
                'mean_embedding': None,
            }
        ]
    )

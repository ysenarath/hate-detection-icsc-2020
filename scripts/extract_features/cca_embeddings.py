import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

from tklearn.feature_extraction.hatespeech import load_hatebase, HatebaseVectorizer
from tklearn.text.word_vec import load_word2vec

hatebase = load_hatebase()

word2vec = load_word2vec()

embeddings = {}
for w in hatebase.keys():
    if w in word2vec.vocab:
        embeddings[w] = [word2vec.word_vec(w)]

hv = HatebaseVectorizer(features=['average_offensiveness'])

hv.fit(None)

C1 = np.array([v[0] for w, v in embeddings.items()])
C1.shape

C2 = hv.feature_vectors.loc[[hv.index[w] for w in embeddings.keys()]]
C2.shape

cca = CCA(n_components=25)

cca.fit(C2, C1)

embeddings = {}
for w in word2vec.vocab:
    embeddings[w] = [word2vec.word_vec(w)]

rev_index = {v: w for (w, v) in hv.index.items()}

words = list(rev_index.keys())

C2 = hv.feature_vectors.loc[words]
C2.shape

X = cca.transform(C2)

df = pd.DataFrame(X)

df['word'] = [rev_index[i] for i in words]

df.to_csv("/scratch/ywijesu/words2.csv")

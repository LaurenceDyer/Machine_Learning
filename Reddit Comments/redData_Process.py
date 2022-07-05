import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

from nltk import word_tokenize          
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from nltk.stem import WordNetLemmatizer

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans

wordnet_lem = WordNetLemmatizer()

pd.set_option('display.max_rows', 1000)
pd.set_option('max_colwidth',80)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

stop_words = set(stopwords.words('english')) 
allDat = pd.read_csv("Comments_Sub.csv")

stop_words = set(stopwords.words('english')) 

allDat['body'] = allDat['body'].str.strip()
allDat['body'] = allDat['body'].str.replace(r"\r;","",regex=True)
allDat['body'] = allDat['body'].str.replace(r"\n","",regex=True)
allDat['body'] = allDat['body'].str.replace(r"\t;","",regex=True)
allDat['body'] = allDat['body'].str.replace(r"&gt;","",regex=True)
allDat['body'] = allDat['body'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ',regex=True)

labels = allDat["Tag"]

cv = TfidfVectorizer(lowercase=True,
                         stop_words='english',
                         ngram_range=(1, 1),
                         tokenizer=word_tokenize, max_df = 0.975, min_df = 0.01)


X = cv.fit_transform(allDat["body"])

print("Performing dimensionality reduction using LSA")
t0 = time.time()
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD()
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

print("done in %fs" % (time.time() - t0))


explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))

km = KMeans(max_iter=100, n_init=10)

km.fit(X)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print(km.cluster_centers_)

print(km.labels_)
print(set(km.labels_))

u_labels = np.unique(km.labels_)
centroids = km.cluster_centers_

print(X.shape)

allDat["labels_new"] = km.labels_
allDat["labels_test"] = labels

allDat.to_csv("allDat_Sub_test.csv")

X_out = pd.DataFrame(X)

X_out["label_old"] = labels
X_out["labels_new"] = km.labels_

X_out.to_csv("KMeans.csv")

for i in u_labels:
    plt.scatter(X[labels == i , 0] , X[labels == i , 1] , label = i)

plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = "k")
plt.legend()
plt.show()

with open("model.pkl", "wb") as f:
    pickle.dump(km, f)


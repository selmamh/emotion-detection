
from time import time
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import warnings
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('stopwords')
# nltk.download('wordnet')
from preprocessing import Word2vec, convert_to_vec, load_data, clean_text, drop_value_from_target, encoding_target, tfidf_vectorizer,TrunSVD,split_data
from models import *


#load the data
df = load_data("output.csv")

#create a new column for clean data
df = df.assign(essay_clean = clean_text(df["essay"]))

#drop surprise emotion
df = drop_value_from_target(df, "surprise")

#encoding the target label
df["target"] = encoding_target(df["emotion"])

#tfidf vectorizer
tfidf, tfv = tfidf_vectorizer(df,"essay_clean")

#truncated SVD
truncSVD, svd = TrunSVD(tfidf)

#train the word2vec model
model = Word2vec(df["essay_clean"])


#split the data original text
X_train, X_test, y_train, y_test = split_data(df["essay_clean"],df["target"],test_size = 0.2)

#convert train and test data to vectors
X_train_vect_avg,X_test_vect_avg = convert_to_vec(model, X_train, X_test)

#params for random forest
# param_grid_rf = {'n_estimators': [100, 300, 500, 800, 1200], 'max_depth':[5, 8, 15, 25],'min_samples_split': 
#     [ 10, 15, 100],'min_samples_leaf': [ 5, 10] }

# params for LinearSVC
# param_grid_svc = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}


svc= svm.SVC()
rf = RandomForestClassifier()
lr = LogisticRegression()


pipeline = Pipeline(
    [
        ("tfidf", tfv),
        ("svd", svd),
        ("svc", svc),
    ]
)
parameters = {
    "tfidf__max_df": (0.5, 0.75, 1.0),
    "tfidf__ngram_range": ((1, 1), (1, 2),(1,3),(1,4)),  
    "svd__n_components": [400,500,600], 

    "svc__C" : [0.1,1, 10, 100],
    "svc__gamma":[1,0.1,0.01,0.001],
    "svc__kernel": ['rbf', 'poly', 'sigmoid',"linear"],

    # "rf__n_estimators": [100, 300, 500, 800, 1200], 
    # "rf__max_depth":[5, 8, 15, 25],
    # "rf__min_samples_split": [ 10, 15, 100],
    # "rf__min_samples_leaf": [ 5, 10] ,

    # 'lr__penalty' : ['l1', 'l2'],
    # 'lr__C' : np.logspace(-4, 4, 20),
    # 'lr__solver' : ['liblinear'],
 
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
print(parameters)
t0 = time()
grid_search.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
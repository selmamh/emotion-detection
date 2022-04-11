
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import warnings
from sklearn import svm

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('stopwords')
# nltk.download('wordnet')
from preprocessing import load_data, clean_text, drop_value_from_target, encoding_target, tfidf_vectorizer,TrunSVD,split_data
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
tfidf = tfidf_vectorizer(df,"essay_clean")

#truncated SVD
truncSVD = TrunSVD(tfidf)

#split the data
X_train, X_test, y_train, y_test = split_data(truncSVD,df["target"],test_size = 0.2)


#params for random forest
param_grid_rf = {'n_estimators': [100, 300, 500, 800, 1200], 'max_depth':[5, 8, 15, 25],'min_samples_split': 
    [ 10, 15, 100],'min_samples_leaf': [ 5, 10] }

# #applying logistic regression
# y_pre_lr =  logisticRegression_model(X_train,y_train,X_test)
# print("Reports from Linear regession model: ")
# print_report(y_pre_lr, y_test)

# #applying random forest
# print("****************************************************************************")
# y_pre_lr =  randomForest_model(X_train,y_train,X_test)
# print("Reports from random forest model: ")
# print_report(y_pre_lr, y_test)

# #applying Linear SVC
# print("****************************************************************************")
# y_pre_lr =  linearSvc_model(X_train,y_train,X_test)
# print("Reports from Linear SVC model: ")
# print_report(y_pre_lr, y_test)

#params for LinearSVC
param_grid_svc = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

#Grid search for linear SVC
grid_SVC = gridSearch(svm.SVC(), param_grid_svc, X_train,y_train)
y_grid_pred = grid_SVC.predict(X_test)
print_report(y_grid_pred, y_test)
print("best params for linear svc = ",grid_SVC.best_estimator_)


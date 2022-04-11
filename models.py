from sklearn import svm
import pandas as pd

import sklearn.metrics as mt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

def linearSvc_model(X_train,y_train,X_test):
    svc = svm.LinearSVC()
    svc.fit(X_train, y_train)
    y_pre_svcL = svc.predict(X_test)
    return y_pre_svcL

def randomForest_model(X_train,y_train,X_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pre_rf = rf.predict(X_test)
    return y_pre_rf

def linearRegression_model(X_train,y_train,X_test):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pre_lr = clf.predict(X_test)
    return y_pre_lr
    
def gridSearch(model, params_dict, X_train,y_train):
    grid = GridSearchCV(model,params_dict,refit=True,verbose=2)
    grid.fit(X_train,y_train)
    return grid



def print_report(y_pred, y_test):
    print(mt.classification_report(y_test, y_pred))

def plot_cm(y_test, y_pre_svcL):
    cm = confusion_matrix(y_test, y_pre_svcL)


    cm_df = pd.DataFrame(cm,
                     index = ['0','1','2','3', '4'], 
                     columns = ['0','1','2','3', '4'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()
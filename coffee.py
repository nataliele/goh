import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from datetime import date, datetime, timedelta
from fuzzywuzzy import process

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from mlflow import log_metric, log_param, log_artifacts
from mlflow.tracking import MlflowClient
import mlflow
from sklearn.metrics import PrecisionRecallDisplay


class ColumnSelection(BaseEstimator, TransformerMixin):
    """extract feature column(s) for further transformation"""
    def __init__(self, col_name):
        self.col_name = col_name
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.col_name]
    
class ColumnSelection2D(BaseEstimator, TransformerMixin):
    """extract feature column for OneHotEncoder.
    Because OHE expects a 2D array, we have to reshape the column serires at the end"""
    def __init__(self, col_name):
        self.col_name = col_name
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.col_name].values.reshape(-1, 1)


class Calculate_age(BaseEstimator, TransformerMixin):
    """create age variable (in years) from date_contacted and dob"""
    def __init__(self, date_contacted, dob):
        self.date_contacted = date_contacted
        self.dob = dob
        self.age = 'age'
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.date_contacted] = pd.to_datetime(X[self.date_contacted])
        X[self.dob] = pd.to_datetime(X[self.dob])
        X[self.age] = (X[self.date_contacted] - X[self.dob]).dt.days/365
        return X[self.age].values.reshape(-1, 1)


class Group_employment(BaseEstimator, TransformerMixin):
    """create age variable (in years) from date_contacted and dob"""
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        status_lst = []
        for status in ['full-time', 'retired', 'part-time', 'unemployed', 'student']:
            str2Match = status
            strOptions = X[self.col].unique()
            Ratios = process.extract(str2Match,strOptions, limit=70)
            status_lst.append([value_set[0] for value_set in Ratios if value_set[1]>85])
        
        cond_lst = [X[self.col].isin(status_lst[0]), X[self.col].isin(status_lst[1]), X[self.col].isin(status_lst[2]), X[self.col].isin(status_lst[3]), X[self.col].isin(status_lst[4])]
        choice_lst = ['full-time', 'retired', 'part-time', 'unemployed', 'student']
        X[f'{self.col}_status'] = np.select(cond_lst, choice_lst, default='')
        
        return X[f'{self.col}_status'].values.reshape(-1, 1)


class Group_owns_coffee_machine(BaseEstimator, TransformerMixin):
    """create age variable (in years) from date_contacted and dob"""
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        yes_lst = ['YES', 'yes', 'y', 'yup', 'Yes', 'ya']
        no_lst = ['nope', 'no', "I don't know", 'No', 'dunno', 'nah', 'n', 'NO']
        
        cond_lst = [X[self.col].isin(yes_lst), X[self.col].isin(no_lst)]
        choice_lst = ['yes', 'no']
        X[f'{self.col}_status'] = np.select(cond_lst, choice_lst, default='')
        
        return X[f'{self.col}_status'].values.reshape(-1, 1)


class Group_preference(BaseEstimator, TransformerMixin):
    """create age variable (in years) from date_contacted and dob"""
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        status_lst = []
        for status in ['beer', 'wine', 'coffee', 'tea', 'soda', 'water']:
            str2Match = status
            strOptions = X[self.col].unique()
            Ratios = process.extract(str2Match,strOptions, limit=66)
            status_lst.append([value_set[0] for value_set in Ratios if value_set[1]>85])
        
        cond_lst = [X[self.col].isin(status_lst[0]), X[self.col].isin(status_lst[1]), X[self.col].isin(status_lst[2]), X[self.col].isin(status_lst[3]), X[self.col].isin(status_lst[4]), X[self.col].isin(status_lst[5])]
        choice_lst = ['beer', 'wine', 'coffee', 'tea', 'soda', 'water']
        X[f'{self.col}_status'] = np.select(cond_lst, choice_lst, default='')
        
        return X[f'{self.col}_status'].values.reshape(-1, 1)


def create_pipeline():

    # union features
    union = FeatureUnion([
                        ('age', Pipeline([
                            ('create_age', Calculate_age('date_contacted', 'dob'))
                        ])),
                        ('employment', Pipeline([
                            ('create_employment', Group_employment('employment')),
                            ('ohe', OneHotEncoder(sparse=False))
                        ])),
                        ('owns_cm', Pipeline([
                            ('create_owns_cm', Group_owns_coffee_machine('owns_coffee_machine')),
                            ('ohe', OneHotEncoder(sparse=False))
                        ])),
                        ('bev_pref', Pipeline([
                            ('create_bev_pref', Group_preference('beverage_preference')),
                            ('ohe', OneHotEncoder(sparse=False))
                        ])),
                        ('bool', Pipeline([
                            ('selector', ColumnSelection(['gender', 'owns_car', 'owns_home'])),
                            ('ohe', OneHotEncoder(sparse=False))
                        ])),
                        ('cont', Pipeline([
                            ('selector', ColumnSelection(['number_of_bags_purchased_competitor', 'competitor_satisfaction']))
                        ]))
                    ])

    # create pipline
    pipeline = Pipeline([
                        ('union', union),
                    #     ('svd',  TruncatedSVD(n_components=50, random_state=42)),
                    #     ('undersampling', RandomUnderSampler(random_state=42)),
                    #     ('smote', SMOTE(random_state=42)),
                    #     ('logreg', LogisticRegression())
                    #     ('rf', RandomForestClassifier(random_state=42))
                        ('hgbm', HistGradientBoostingClassifier(random_state=42, learning_rate=0.01))
                    ])

    return pipeline


def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
    print("Number of training samples:", len(X_train))
    model = clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    y_pred = np.where(y_prob>=0.5, 1, 0)
    print("Micro-averaged F1 score on test set: "
          "%0.3f" % f1_score(y_test, y_pred, average='micro'))
    print(classification_report(y_test, y_pred))
    print("-" * 10)
    print('Percentage of positive predictions', (y_pred==1).sum()/len(y_pred)*100, '%')
    print()
    return y_pred, y_prob, model


if __name__ == "__main__":

    experiment_id = mlflow.set_experiment('GoHealth')
    with mlflow.start_run(run_name='base_model') as run:
        mlflow_client = MlflowClient()

        # Log an artifact (output file)
        # if not os.path.exists("outputs"):
        #     os.makedirs("outputs")
        # with open("outputs/test.txt", "w") as f:
        #     f.write("hello world!")
        # log_artifacts("outputs")

        # import datasets
        train_raw = pd.read_csv('train.csv').iloc[:, 1:]
        prediction_raw = pd.read_csv('predictions.csv').iloc[:, 1:]

        # create train and test set from original train dataset
        X_train, X_test, y_train, y_test = train_test_split(train_raw.drop(columns='bought_coffee'), train_raw['bought_coffee'], random_state=42)

        # create pipeline:
        clf = create_pipeline()

        # print metrics
        y_hgbm, y_prob, model = eval_and_print_metrics(clf, X_train, y_train, X_test, y_test)

        # log_param()
        # log_metric('Average precision', ap)

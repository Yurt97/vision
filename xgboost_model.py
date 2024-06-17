import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import argparse
from pathlib import Path
class XGB:
    def Train():
        #csv_dir: str
        df = pd.read_csv(Path("C:/Users/dmitr/neural/test.csv"))
        #df_test = pd.read_csv(Path(test_csv))
        df.head()

        X = df.drop(["looking", "x", "y"], axis=1).copy()
        y = df["looking"].copy()
        #X_testset = df_test.drop("looking", axis=1).copy()
        #y_testset = df_test["looking"].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)

        clf_xgb = xgb.XGBClassifier(objective = "binary:logistic", missing = np.nan, seed = 42)
        clf_xgb.fit(X_train, y_train, verbose = True, early_stopping_rounds = 10, eval_metric = "aucpr", eval_set = [(X_test, y_test)])
        y_test_predicted = clf_xgb.predict(X_test)
        print(clf_xgb.classes_)
        cm = confusion_matrix(y_test, y_test_predicted, labels = clf_xgb.classes_)
        cm_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["not looking","looking"])
        cm_disp.plot()
        plt.show()

XGB.Train()
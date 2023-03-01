import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV


class ModelData:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.tuned_models = []
        self.results_dfs = []

    def model_evaluation(self, models: list, model_type: str):
        for model in models:
            model.fit(self.X_train, self.y_train)
            print(f'{"="*10}{type(model).__name__}{"="*10}')
            self.confusion_matrix_sklearn(model)
            self.model_performance_classification_sklearn(model, model_type)
        return pd.concat(self.results_dfs, ignore_index=True)

    def tune_model_hyperparameter(self, model, parameters: dict):
        # Type of scoring used to compare parameter combinations
        scorer = metrics.make_scorer(metrics.f1_score)

        # Run the grid search
        grid_obj = GridSearchCV(model, parameters, scoring=scorer, n_jobs=-1)

        grid_obj = grid_obj.fit(self.X_train, self.y_train)

        # Set the clf to the best combination of parameters
        model_estimator = grid_obj.best_estimator_
        self.tuned_models.append(model_estimator)

        return model_estimator

    # defining a function to compute different metrics to check performance of a classification model built using sklearn
    def model_performance_classification_sklearn(self, model, model_type: str):
        """
        Function to compute different metrics to check classification model performance

        model: classifier
        X_test: independent variables
        y_test: dependent variable
        """

        # predicting using the independent variables
        pred = model.predict(self.X_test)

        acc = accuracy_score(self.y_test, pred)  # to compute Accuracy
        recall = recall_score(self.y_test, pred)  # to compute Recall
        precision = precision_score(self.y_test, pred)  # to compute Precision
        f1 = f1_score(self.y_test, pred)  # to compute F1-score

        # creating a dataframe of metrics
        df_perf = pd.DataFrame(
            {
                "Model": type(model).__name__,
                "Type": model_type,
                "Accuracy": acc,
                "Recall": recall,
                "Precision": precision,
                "F1": f1,
            },
            index=[0],
        )

        self.results_dfs.append(df_perf)

        return df_perf

    def confusion_matrix_sklearn(self, model):
        """
        To plot the confusion_matrix with percentages

        model: classifier
        X_test: independent variables
        y_test: dependent variable
        """
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        labels = np.asarray(
            [
                [f"{item:,.0f}" + f"\n{item / cm.flatten().sum():.2%}"]
                for item in cm.flatten()
            ]
        ).reshape(2, 2)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=labels, fmt="")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        plt.show()

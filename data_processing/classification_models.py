import numpy as np
import matplotlib.pyplot as plt


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# MLP Classifier
def mlp_classifier(train_x, train_y):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 1000), random_state=1)
    model.fit(train_x, train_y)
    return model


def evaluate(test_positive, test_negative, model):
    number_of_correct = 0
    number_of_true = 0
    positive_predict = model.predict(test_positive)
    for predict in positive_predict:
        if predict == 1:
            number_of_correct += 1
            number_of_true += 1
    real_true = number_of_correct
    negative_predict = model.predict(test_negative)
    for predict in negative_predict:
        if predict == 0:
            number_of_correct += 1
        else:
            number_of_true += 1
    precision = real_true / number_of_true
    recall = real_true / len(test_positive)
    accuracy = number_of_correct / (len(test_positive) + len(test_negative))
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, accuracy, f1_score

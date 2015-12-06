import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
import cPickle
import sys
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from learners import LearnerSuite
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Instantiates a class that runs GridSearchCV on a variety of models
def get_best_clf(X_train, y_train, category):

    models = [
        RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1, max_depth=30),
        # DecisionTreeClassifier(),
        # GradientBoostingClassifier(n_estimators=300, learning_rate=1.0, max_depth=30, random_state=1),
        # KNeighborsClassifier(),
        # AdaBoostClassifier(n_estimators=300),
        # svm.SVC(),
        # SGDClassifier(loss="hinge", penalty="l2")
    ]
    tuning_ranges = {
        'RandomForestClassifier': {'max_features': list(np.unique(np.logspace(np.log10(2), np.log10(X_train.shape[1] - 1), 5).astype(np.int)))},
        'DecisionTreeClassifier': {'max_depth': [5, 10, 20, 50, None]},
        'GradientBoostingClassifier': {},
        'SVC': {},
        'SGDClassifier': {},
        'KNeighborsClassifier': {},
        'AdaBoostClassifier': {}
    }

    suite = LearnerSuite(tuning_ranges=tuning_ranges, njobs=1, cv=5, verbose=True, models=models)

    suite.fit(X_train, y_train)

    for modelName in suite.best_models:
        joblib.dump(suite.best_models[modelName], "trainedModels/" + category + "_" + modelName + '.pkl')

    print 'Accuracy scores when classifying on {}:'.format(category)

    for model_name in suite.best_scores:
        print model_name, suite.best_scores[model_name]

    model_scores = {model_name: suite.best_scores[model_name] for model_name in suite.best_scores}

    best_model = max(model_scores.iteritems(), key=operator.itemgetter(1))[0]

    print 'Best model for Category {}: {}'.format(category, best_model)

    clf = suite.best_models[best_model]

    return clf


def get_x_y_df(df, category, categories):
    input_df = train_df.copy(deep=True)
    y = input_df[category].values
    input_df = input_df.drop(categories, axis=1)
    x = input_df.values
    return x, y


# Creates a dictionary of the best model for each Product category
def get_category_models(train_df, categories, load_pre_built):
    clf_dict = dict()

    for category in categories:
        if load_pre_built:
            clf_dict[category] = joblib.load('full-set-trained-models/' + category + '_str.pkl')
        else:
            x_train, y_train = get_x_y_df(train_df, category, categories)
            clf_dict[category] = get_best_clf(x_train, y_train, category)

    return clf_dict


def measure_accuracy(y_test, y_pred):
    print 'Confusion Matrix: '
    print confusion_matrix(y_true=y_test, y_pred=y_pred)


def predict_all(clf_dict, test_df, categories):
    predictions = []

    for key, value in clf_dict.iteritems():
        y_test, x_test = get_x_y_df(test_df, key, categories)
        y_pred = value.predict(x_test)
        measure_accuracy(y_test, y_pred)
        predictions.append(y_pred)

    return predictions


def measure_total_accuracy(models, test_df, categories):
    category_dfs = dict()
    for category in categories:
        x_df, y_df = get_x_y_df(test_df, category, categories)
        category_dfs[category] = (x_df, y_df)

    correct_count = 0

    for i in range(0, len(test_df)):
        all_correct = True
        for category in categories:
            (x_df, y_df) = category_dfs[category]
            prediction = models[category].predict(x_df[i])
            if prediction != y_df[i]:
                all_correct = False
        if all_correct:
            correct_count += 1

    return float(correct_count) / float(len(test_df))


def main(train_df, test_df):
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    clf_dict = get_category_models(train_df, categories, len(sys.argv) > 1 and sys.argv[1] == "-p")

    total_accuracy = measure_total_accuracy(clf_dict, test_df, categories)

    print 'Total Accuracy on Test Set is: {}'.format(total_accuracy)


if __name__ == '__main__':
    df = pd.read_csv('./data/complete-full-set.csv')

    df.set_index(['customer_ID'], inplace=True)

    customer_ids = df.index.get_level_values(0).unique()

    msk = np.random.rand(len(customer_ids)) < 0.8

    train_cust_ids = customer_ids[msk]
    test_cust_ids = customer_ids[~msk]

    train_df = df[df.index.get_level_values(0).isin(train_cust_ids)]
    test_df = df[df.index.get_level_values(0).isin(test_cust_ids)]

    main(train_df, test_df)

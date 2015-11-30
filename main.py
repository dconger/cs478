import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
import cPickle

from sklearn.preprocessing import LabelEncoder
from learners import LearnerSuite
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

def transform_data(df):
    df.rename(columns={'homeowner': 'is_homeowner', 'married_couple': 'is_married_couple'}, inplace=True)

    # convert car_value from letters to numbers
    uvals = np.sort(df['car_value'].unique())
    carv_mapping = {cv: v for cv, v in zip(uvals, range(1, len(uvals)+1))}
    df['car_value'] = df['car_value'].map(carv_mapping)

    for state in df['state'].unique():
        df['is_' + state] = df['state'] == state

    df['is_C_previous_nan'] = df['C_previous'].isnull()
    df['is_C_previous_1'] = df['C_previous'] == 1
    df['is_C_previous_2'] = df['C_previous'] == 2
    df['is_C_previous_3'] = df['C_previous'] == 3
    df['is_C_previous_4'] = df['C_previous'] == 4

    for gs in np.sort(df['group_size'].unique()):
        df['is_group_size_' + str(gs)] = df['group_size'] == gs

    df['is_Mon'] = df['day'] == 0
    df['is_Tue'] = df['day'] == 1
    df['is_Wed'] = df['day'] == 2
    df['is_Thu'] = df['day'] == 3
    df['is_Fri'] = df['day'] == 4
    df['is_Sat'] = df['day'] == 5
    df['is_Sun'] = df['day'] == 6

    fill_values = {'risk_factor': df['risk_factor'].mean(), 'duration_previous': df['duration_previous'].mean()}
    df['is_risk_factor_missing'] = pd.isnull(df['risk_factor'])
    df['is_duration_previous_missing'] = pd.isnull(df['duration_previous'])
    df = df.fillna(value=fill_values)

    not_predictors = ['time', 'record_type', 'day', 'state', 'location', 'group_size', 'C_previous']

    df = df.drop(not_predictors, axis=1)

    return df

# Instantiates a class that runs GridSearchCV on a variety of models
def get_best_clf(X_train, y_train, category):
	models = [DecisionTreeClassifier(),
			  RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=2, max_depth=30)]

	tuning_ranges = {'DecisionTreeClassifier': {'max_depth': [5, 10, 20, 50, None]},
					 'RandomForestClassifier': {'max_features': list(np.unique(np.logspace(np.log10(2), np.log10(X_train.shape[1] - 1), 5).astype(np.int)))}}

	suite = LearnerSuite(tuning_ranges=tuning_ranges, njobs=1,
                                    cv=5, verbose=True, models=models)

	suite.fit(X_train, y_train)

	print 'Accuracy scores when classifying on {}:'.format(category)

	for model_name in suite.best_scores:
		print model_name, suite.best_scores[model_name]

	model_scores = {model_name: suite.best_scores[model_name] for model_name in suite.best_scores}

	best_model = max(model_scores.iteritems(), key=operator.itemgetter(1))[0]

	print 'Best model for Category {}: {}'.format(category, best_model)

	clf = suite.best_models[best_model]

	return clf

# Creates a dictionary of the best model for each Product category
def get_category_models(train_df, categories):
	
	clf_dict = dict()

	for category in categories:
		input_df = train_df.copy(deep=True)
		y_train = input_df[category].values
		input_df = input_df.drop(categories, axis=1)
		X_train = input_df.values
		clf_dict[category] = get_best_clf(X_train, y_train, category)

	return clf_dict

def measure_accuracy(y_test, y_pred):
	confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
	print 'Confusion Matrix: '
	print confmat

def predict_all(clf_dict, test_df, categories):

	predictions = []

	for key, value in clf_dict.iteritems():
		test_df_copy = test_df.copy(deep=True)
		y_test = test_df_copy[key]
		test_df_copy = test_df_copy.drop(categories, axis=1)
		X_test = test_df_copy.values
		clf = value
		y_pred = clf.predict(X_test)
		measure_accuracy(y_test, y_pred)
		predictions.append(y_pred)

	return predictions


def measure_total_accuracy(predictions_df, test_df):
	categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
	test_df_categories = test_df[categories]
	comparison = predictions_df.values == test_df_categories.values

	correct = 0
	
	for item in comparison:
		if False in item:
			pass
		else:
			correct += 1

	return float(correct) / float(len(test_df))

def main(train_df, test_df):
	categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

	clf_dict = get_category_models(train_df, categories)

	predictions = predict_all(clf_dict, test_df, categories)
	predictions_df = pd.DataFrame(predictions).T
	predictions_df.columns = categories
	
	total_accuracy = measure_total_accuracy(predictions_df, test_df)

	'Total Accuracy on Test Set is:', total_accuracy

if __name__ == '__main__':

	df = pd.read_csv('./data/train.csv')
	
	df.set_index(['customer_ID', 'shopping_pt'], inplace=True)

	customer_ids = df.index.get_level_values(0).unique()

	df = df.select(lambda x: x[0] < customer_ids[100], axis=0)

	df = transform_data(df)

	msk = np.random.rand(len(customer_ids)) < 0.8

	train_cust_ids = customer_ids[msk]
	test_cust_ids = customer_ids[~msk]

	train_df = df[df.index.get_level_values(0).isin(train_cust_ids)]
	test_df = df[df.index.get_level_values(0).isin(test_cust_ids)]

	main(train_df, test_df)

import argparse

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def relative_absolute_error(y_test, y_predict):
	y_test = np.array(y_test)
	y_predict = np.array(y_predict)
	error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
		y_test) - y_test))
	return error


from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas
import os

task_dict = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary':
	'cls', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg'
             }


def downstream_task_new(data, task_type):
	X = data.iloc[:, :-1]
	y = data.iloc[:, -1].astype(float)
	if task_type == 'cls':
		clf = RandomForestClassifier(random_state=0, n_jobs=128)
		f1_list = []
		skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
		for train, test in skf.split(X, y):
			X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
			], X.iloc[test, :], y.iloc[test]
			clf.fit(X_train, y_train)
			y_predict = clf.predict(X_test)
			f1_list.append(f1_score(y_test, y_predict, average='weighted'))
		return np.mean(f1_list)
	elif task_type == 'reg':
		kf = KFold(n_splits=5, random_state=0, shuffle=True)
		reg = RandomForestRegressor(random_state=0, n_jobs=128)
		rae_list = []
		for train, test in kf.split(X):
			X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
			], X.iloc[test, :], y.iloc[test]
			reg.fit(X_train, y_train)
			y_predict = reg.predict(X_test)
			rae_list.append(1 - relative_absolute_error(y_test, y_predict))
		return np.mean(rae_list)
	elif task_type == 'det':
		knn = KNeighborsClassifier(n_neighbors=5, n_jobs=128)
		skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
		ras_list = []
		for train, test in skf.split(X, y):
			X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
			], X.iloc[test, :], y.iloc[test]
			knn.fit(X_train, y_train)
			y_predict = knn.predict(X_test)
			ras_list.append(roc_auc_score(y_test, y_predict))
		return np.mean(ras_list)
	elif task_type == 'mcls':
		clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_jobs=128))
		pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
		skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
		for train, test in skf.split(X, y):
			X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
			clf.fit(X_train, y_train)
			y_predict = clf.predict(X_test)
			f1_list.append(f1_score(y_test, y_predict, average='micro'))
		return np.mean(f1_list)
	elif task_type == 'rank':
		pass
	else:
		return -1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, choices=['airfoil', 'amazon_employee',
	                                                    'ap_omentum_ovary', 'german_credit',
	                                                    'higgs', 'housing_boston', 'ionosphere',
	                                                    'lymphography', 'messidor_features', 'openml_620',
	                                                    'pima_indian', 'spam_base', 'spectf', 'svmguide3',
	                                                    'uci_credit_card', 'wine_red', 'wine_white', 'openml_586',
	                                                    'openml_589', 'openml_607', 'openml_616', 'openml_618',
	                                                    'openml_637'], default='wine_white')
	args = parser.parse_args()
	dataset_name = args.dataset
	data = pandas.read_hdf(dataset_name + '.hdf', 'data')
	type = task_dict[dataset_name]
	performance = downstream_task_new(data, type)
	print(f'the generated feature on dataset {dataset_name} performance is : ')
	print("%.3f" % performance)

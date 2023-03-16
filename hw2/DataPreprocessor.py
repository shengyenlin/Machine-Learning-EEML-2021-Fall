import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import RobustScaler

class DataPreprocessor:
    def __init__(self):
        self.train_mean = None
        self.train_std = None
        self.num_cols = ["age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]
        self.cat_cols = ["workclass", "education_num", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
        self.kept_cat_cols = ["workclass", "relationship", "sex"]
        self.kept_order_cat_cols = "education_num"
        self.kept_num_cols = ["age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]
        self.all_native_countries = None
        #used for remove less frequent categorical columns
        self.keep_cols = None
        self.drop_category = ["workclass_Never-worked"]
        self.robust_scaler = None

    def _transform_label(self, data_train):
        label_dict = {'<=50K': 0, '>50K': 1}
        data_train_ = data_train.copy()
        data_train_["income"] = data_train_["income"].apply(lambda x: label_dict[x])
        return data_train_ 
        
    def _do_one_hot_encoding(self, data_cat: pd.DataFrame, isTraining = False):
        if isTraining:
            self.all_native_countries = data_cat["native_country"].value_counts().index.sort_values().to_list()
        else:
            #fix missing columns in testing dataset
            data_cat["native_country"] = data_cat["native_country"].astype(pd.CategoricalDtype(categories=self.all_native_countries))
        data_one_hot = pd.get_dummies(data_cat)
        # if not isTraining:
        #     data_one_hot = data_one_hot.drop(self.drop_category, axis = 1)
        return data_one_hot
        
    def _normalize_data(self, X_data: pd.DataFrame, isTraining = False):
        if isTraining:
            self.train_mean = X_data.mean(axis = 0)
            self.train_std = X_data.std(axis = 0)
        normalized_data = (X_data - self.train_mean) / self.train_std
        return normalized_data

    def _remove_less_frequent_cat_features(self, X_data_cat: pd.DataFrame, lower_bound_ratio = 0.5, isTraining = False):
        if isTraining:
            cols = X_data_cat.columns
            keep_cols_bool = (X_data_cat.sum(axis = 0) / X_data_cat.shape[0]) > lower_bound_ratio
            self.keep_cols = cols[keep_cols_bool]
        return X_data_cat[self.keep_cols]

    def select_special_column(self, data: pd.DataFrame, isTraining = False):
        """
        Special column
        - Education_num: 1
        - workclass: never-worked
        """
        special_index = []
        cond1 = (data["education_num"] == 1)
        cond2 = (data["workclass"] == "Never-worked")
        #data_kept = data[~cond1 & ~cond2]
        if not isTraining:
            #Keep these rows in testing dataset
            #Manually label them after
            data_kept = data
            special_index = data[cond1 | cond2].index.tolist()
        return data_kept, special_index

    def keep_num_columns(self, data_num: pd.DataFrame):
        return data_num[self.kept_num_cols]

    def keep_cat_columns(self, data_cat: pd.DataFrame):
        return data_cat[self.kept_cat_cols]

    def keep_order_cat_columns(self, data: pd.DataFrame):
        return data[self.kept_order_cat_cols]

    def robust_scaling(self, data_num: pd.DataFrame, isTraining = False):
        if isTraining:
            self.robust_scaler = RobustScaler()
            self.robust_scaler.fit(data_num)
        data_num_scaled = self.robust_scaler.transform(data_num)
        return data_num_scaled

    def preprocess_train_data(self, data_train: pd.DataFrame):
        #avoid changes in original dataset
        data_train_ = self._transform_label(data_train)
        #data_train_, _ = self.select_special_column(data_train_, isTraining=True)

        #preprocessing - numerical 
        #data_train_num = self._normalize_data(data_train_num, isTraining=True)
        #data_train_num = self.keep_num_columns(data_train_)
        data_train_num = data_train_[self.num_cols]
        #data_train_num = self.robust_scaling(data_train_num, isTraining=True) #return np.array

        #preprocessing - categorical (16 features) / order categorical (1 features)
        #data_train_order_cat = self.keep_order_cat_columns(data_train_)
        #data_train_cat = self.keep_cat_columns(data_train_)
        data_train_cat = data_train_[self.cat_cols]
        data_train_cat = self._do_one_hot_encoding(data_train_cat, isTraining=True)
        print(data_train_cat.shape)
        #data_train_cat = self._remove_less_frequent_cat_features(data_train_cat, isTraining=True)

        #combine
        # data_train_cat = np.array(data_train_cat)
        # data_train_order_cat = np.array(data_train_order_cat)
        # print(data_train_cat.shape)
        # print(data_train_order_cat.shape)
        #data_train_preprocessed = np.concatenate([data_train_num, data_train_order_cat.reshape(-1, 1), data_train_cat], axis = 1)
        data_train_preprocessed = pd.concat([data_train_num, data_train_cat], axis = 1)
        X_train = np.array(data_train_preprocessed)

        #process y
        y_train = np.array(data_train_["income"])

        return X_train, y_train

    def preprocess_test_data(self, data_test: pd.DataFrame):
        #avoid changes in original dataset
        data_test_ = data_test.copy()
        #data_test_, special_index = self.select_special_column(data_test, isTraining=False)

        #preprocessing - numerical
        data_test_num = data_test_[self.num_cols]
        #data_test_num = self._normalize_data(data_test_num, isTraining=False)
        #data_test_num = self.keep_num_columns(data_test_)
        #data_test_num = self.robust_scaling(data_test_num, isTraining=False) #return np.array

        #preprocessing - categorical
        data_test_cat = data_test_[self.cat_cols]
        # data_test_cat = self.keep_cat_columns(data_test_)
        data_test_cat = self._do_one_hot_encoding(data_test_cat, isTraining=False)
        print(data_test_cat.shape)
        # data_test_order_cat = self.keep_order_cat_columns(data_test_)
        #data_test_cat = self._remove_less_frequent_cat_features(data_test_cat, isTraining=False)

        #combine        
        # data_test_cat = np.array(data_test_cat)
        # data_test_order_cat = np.array(data_test_order_cat)
        # print(data_test_cat.shape)
        # print(data_test_order_cat.shape)
        #data_test_preprocessed = np.concatenate([data_test_num, data_test_order_cat.reshape(-1, 1), data_test_cat], axis = 1)
        data_test_preprocessed = pd.concat([data_test_num, data_test_cat], axis = 1)
        X_test = np.array(data_test_preprocessed)
        return X_test

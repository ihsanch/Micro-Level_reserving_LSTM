#----------Datasets pre-processing--------#

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def prep_data():
    # import datasets
    data_train = pd.read_csv('./Datasets/Simulated_Cashflow_train_LSTM.csv', sep=';')
    data_valid = pd.read_csv('./Datasets/Simulated_Cashflow_valid_LSTM.csv', sep=';')
    data_test = pd.read_csv('./Datasets/Simulated_Cashflow_test_LSTM.csv', sep=';')

    # testing if the same category values are in both datasets
    test1 = (np.sort(data_train["cc"].unique()) == np.sort(data_valid["cc"].unique())).all()
    test2 = (np.sort(data_train["inj_part"].unique()) == np.sort(data_valid["inj_part"].unique())).all()
    assert test1
    assert test2

    # drop unused feature for the LSTM
    List_feature_drop = ['ClNr']

    # Create dictionaries for injured part and claim code
    labels_inj_part = data_train['inj_part'].astype('category').cat.categories.tolist()
    dict_inj_part = {k: v for k, v in zip(labels_inj_part, list(range(1, len(labels_inj_part) + 1)))}
    labels_cc = data_train['cc'].astype('category').cat.categories.tolist()
    dict_cc = {k: v for k, v in zip(labels_cc, list(range(1, len(labels_cc) + 1)))}

    def get_data(data_name):
        data_name = data_name.drop(List_feature_drop, axis=1)
        data_name['AY'] = data_name['AY'] - 1994 #Accident year expressed in number of years from 1994
        data_name['inj_part'].replace(dict_inj_part, inplace=True) #maping categories with the dictionary
        data_name['cc'].replace(dict_cc, inplace=True) #maping categories with the dictionary
        return data_name

    data_train = get_data(data_train)
    data_valid = get_data(data_valid)
    data_test = get_data(data_test)

    # standardize payments
    pay_cols = ['Pay{:02d}'.format(i) for i in range(0, 12)] #payment column names
    amount_mean = np.nanmean(data_train.loc[:, pay_cols].values) #evaluate payment mean based on the training datatset
    amount_std = np.nanstd(data_train.loc[:, pay_cols].values) #evaluate payment standard deviation (std) based on the training datatset
    scale_param = pd.DataFrame({"amount_mean": [amount_mean], "amount_std": [amount_std]})
    scale_param.to_csv(path_or_buf='scale_param.csv', sep=';') # save mean and std as a csv file for later use

    # save datasets before standardization for later comparison
    data_train.to_csv(path_or_buf='data_train_copy.csv', sep=';')
    data_valid.to_csv(path_or_buf='data_valid_copy.csv', sep=';')
    data_test.to_csv(path_or_buf='data_test_copy.csv', sep=';')

    # standardize payments
    data_train.loc[:, pay_cols] = (data_train.loc[:, pay_cols].values - amount_mean) / amount_std
    data_valid.loc[:, pay_cols] = (data_valid.loc[:, pay_cols].values - amount_mean) / amount_std
    data_test.loc[:, pay_cols] = (data_test.loc[:, pay_cols].values - amount_mean) / amount_std

    # age
    index_age = data_train.columns.get_loc("age") #age column index
    # scale feature "age" for the training dataset
    # evaluate min and max age based on the training datatset
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_train.iloc[:, [index_age]])
    data_train.iloc[:, [index_age]] = scaler.transform(data_train.iloc[:, [index_age]])
    # scale feature "age" for the validation dataset
    data_valid.iloc[:, [index_age]] = scaler.transform(data_valid.iloc[:, [index_age]])
    # scale feature "age" for the testing dataset
    data_test.iloc[:, [index_age]] = scaler.transform(data_test.iloc[:, [index_age]])

    # save datasets
    data_train.to_csv(path_or_buf='data_train.csv', sep=';')
    data_valid.to_csv(path_or_buf='data_valid.csv', sep=';')
    data_test.to_csv(path_or_buf='data_test.csv', sep=';')

    return data_train, data_valid

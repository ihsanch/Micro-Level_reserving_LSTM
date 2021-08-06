import torch
from torch.utils.data import DataLoader, TensorDataset
import configparser
import numpy as np
config = configparser.ConfigParser()

def format_and_label(data_name): # group and label feature columns
    pay_cols = ['Pay{:02d}'.format(i) for i in range(0, 12)] #list of payment column names
    ind_pay_cols = ['Ind_Pay_{:0d}'.format(i) for i in range(0, 12)] #list of payment indicator I_{k,j} column names
    obs_pay_cols = ['Obs_Pay_{:0d}'.format(i) for i in range(0, 12)] #list of indicator r_{k,j} column names (r_{k,j}=1
    time_ind = torch.FloatTensor(np.arange(1, 12)/11).repeat(1, len(data_name)).view(len(data_name), 11, 1) #development period j/11

    sequences_cols = [item for sublist in zip(ind_pay_cols, pay_cols, obs_pay_cols) for item in sublist][:-3] # order dynamic inputs by period
    Y_seq = [item for sublist in zip(ind_pay_cols, pay_cols) for item in sublist][2:] # order target pairs (I_{k,j},Y_{k,j}) by period

    X_info = data_name.loc[:, ['AY', 'RepDel', 'age']].values #group static quantitative features
    X_feature_cat_lob = data_name.loc[:, ['LoB']].values #label line of business column
    X_feature_cat_cc = data_name.loc[:, ['cc']].values #label claim code column
    X_feature_cat_inj_part = data_name.loc[:, ['inj_part']].values #label injured part column
    X_sequences = data_name.loc[:, sequences_cols].values.reshape(len(data_name), 11, 3) #reshape dynamic inputs
    X_sequences = torch.tensor(X_sequences).float()
    X_sequences = torch.cat((X_sequences, time_ind), dim=2) #concatenate the 3 dynamic features (I_{k,j},Y_{k,j},r_{k,j}) with scaled development period
    Y_sequences = data_name.loc[:, Y_seq].values.reshape(len(data_name), 11, 2)
    data = TensorDataset(torch.tensor(X_info).float(), torch.tensor(X_feature_cat_lob), torch.tensor(X_feature_cat_cc),
                         torch.tensor(X_feature_cat_inj_part), torch.tensor(X_sequences).float(),
                         torch.tensor(Y_sequences).float())
    return(data)


def rearrange_columns(samples): # identify inputs and targets
    X_info, X_feature_lob, X_feature_cc, X_feature_inj_part, X_sequences, Y_sequences = list(zip(*samples))

    X_info = torch.stack(X_info, dim=0)
    X_feature_lob = torch.tensor(X_feature_lob)
    X_feature_cc = torch.tensor(X_feature_cc)
    X_feature_inj_part = torch.tensor(X_feature_inj_part)
    X_sequences = torch.stack(X_sequences, dim=0)
    X = (X_info, X_feature_lob, X_feature_cc, X_feature_inj_part, X_sequences) #identify inputs
    Y_sequences = torch.stack(Y_sequences, dim=0)
    y = (Y_sequences) #identify targets
    return X, y


def get_dataloaders(train_dataset, valid_dataset, shuffle=True): #create batches for both training and validation datasets
    config.read('config_parameters.ini')  # read it to get hyper-parameters
    batch_size = config.getint('train_hyp','batch_size')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=rearrange_columns, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size,
        collate_fn=rearrange_columns, drop_last=True
    )
    return train_loader, valid_loader


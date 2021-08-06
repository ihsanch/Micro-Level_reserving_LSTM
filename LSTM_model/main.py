# import packages and functions
import numpy as np
import configparser
import torch.distributions
from data_preparation import prep_data
from loader_LSTM import format_and_label, get_dataloaders
from LSTM_model import LSTM_Model, train, save_model
config = configparser.ConfigParser()
torch.manual_seed(2019)
np.random.seed(2019)

# read hyper-parameters
config.read('config_parameters.ini')
# get the training and validation datasets
data_train, data_valid = prep_data()

# group and label feature columns
train_dataset = format_and_label(data_train)
valid_dataset = format_and_label(data_valid)

# create batches
train_loader, valid_loader = get_dataloaders(train_dataset, valid_dataset)

# read batch size
batch_size = config.getint('train_hyp', 'batch_size')

#scale hyper-parameter
alpha = 0.2
torch.manual_seed(2019)
np.random.seed(2019)

# define the network
net = LSTM_Model()
# training process
model = train(net, train_loader, valid_loader, alpha=alpha)
#save the network optimal weights
save_model("weights/Weight_LSTM_Model_"+str(batch_size)+"_"+str(alpha), model)
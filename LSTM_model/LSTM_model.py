# import functions
import numpy as np
import torch
import torch.distributions
from torch import optim
import torch.nn as nn
from poutyne.framework import Model, ModelCheckpoint, Callback, CSVLogger, EarlyStopping, ReduceLROnPlateau
from loss_functions import regression_loss, class_loss
import configparser
config = configparser.ConfigParser()

#---define LSTM network----
class LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()

        config.read('config_parameters.ini') #hyper-parameters file

        #define Embedding layer for each of the 3 categorical features
        self.embed_lob = nn.Embedding(config.getint('size', 'size_lob'), config.getint('embed_size', 'embed_size_lob'))
        self.embed_cc = nn.Embedding(config.getint('size', 'size_cc'), config.getint('embed_size', 'embed_size_cc'))
        self.embed_inj_part = nn.Embedding(config.getint('size', 'size_inj_part'), config.getint('embed_size', 'embed_size_inj_part'))

        sum_embed_size = sum([int(i) for i in [x[1] for x in config.items('embed_size')]][:-1]) #get the sum of embedding layers outputs
        self.hidden_linear = nn.Linear(3 + sum_embed_size, config.getint('embed_size', 'embed_size_context')) #define linear layer to reduce the contexte vector size

        #define LSTM module
        self.pay_lstm = nn.LSTM(input_size= 4 + config.getint('embed_size', 'embed_size_context'),
                                num_layers= 1,
                                hidden_size= config.getint('hidden_size', 'hidden_state_size'),
                                batch_first= True)
        self.Ind_pay_linear = nn.Linear(config.getint('hidden_size', 'hidden_state_size'), 1) #define linear layer to obtain preticted probability from hidden state h_{k,j
        self.pay_linear = nn.Linear(config.getint('hidden_size', 'hidden_state_size'), 1) #define linear layer to obtain preticted payment from hidden state
        self.prob = torch.tensor([1]) #initialize probability for teacher forcing process

    def step(self, input, hidden=None): #define how to obtain prediction from hidden stat h_{k,j}
        # LSTM module initialized with the claim history (hidden) and fed with input X
        output, hidden_out = self.pay_lstm(input.view(len(input), 1, len(input[0])), hidden) #output = hidden state h_{k,j}
        pay = self.pay_linear(output) #to obtain preticted payment
        Ind_pay = torch.sigmoid(self.Ind_pay_linear(output)) #apply a sigmoid function to obtain a probability value
        return Ind_pay, pay, hidden_out #return predictions and hidden history

    def forward(self, input, hidden=None):
        X_info, X_feature_cat_lob, X_feature_cat_cc, X_feature_cat_inj_part, X_sequences = input
        #apply embedding layer for each categorical features
        embed_cat_lob = self.embed_lob(X_feature_cat_lob)
        embed_cat_cc = self.embed_cc(X_feature_cat_cc)
        embed_cat_inj_part = self.embed_inj_part(X_feature_cat_inj_part)
        #concatenate the embedded features with the static quantitative informations
        embed_context = torch.cat((X_info, embed_cat_lob, embed_cat_cc, embed_cat_inj_part), dim=1)
        embed_context = self.hidden_linear(embed_context) #obtain the static context C_{0}

        size_embed_context = config.getint('embed_size', 'embed_size_context') #size of C_{0}
        #X_sequences is a tensor of dimension 3 : [batch size, number of period, (I_{k,j},Y_{k,j},r_{k,j},j/11)]
        #See loader_LSTM for more details about X_sequences construction
        size_seq = len(X_sequences) #batch size
        length_seq = len(X_sequences[0]) #number of period
        rep_embed_context = embed_context.repeat(1, length_seq).view(size_seq, length_seq , size_embed_context ) #make copies of context
        X_sequences = torch.cat((X_sequences, rep_embed_context), dim=2) #at each step i the input is: X_sequences[:,i,:] is concatenated with the context C_{0}

        Ind_pay_pred = torch.zeros(size_seq, length_seq) #initialize a tensor for the predicted probabilities
        pay_pred = torch.zeros(size_seq, length_seq) #initialize a tensor for the predicted payment
        sample_ber = torch.distributions.bernoulli.Bernoulli(self.prob) #call a bernoulli sampler with probability self.prob

        for i in range(length_seq): #for each step, the following treatments concern an entire bacth
            # for each step i we select the appropriate input tensor
            input_t = X_sequences[:, i]
            if i == 0:
                # at the first step, no history is known, we fed an LSTM module with the input
                Ind_pay, pay, hidden = self.step(input_t)
            else:
                if self.training:
                    #throughout the training process, we apply the teacher forcing
                    #We need to sample according to Bernoulli(prob) to select input index who are going to have a later replacement process
                    sample_prob = sample_ber.sample(torch.Size([size_seq])) #sample 0,1
                    index_input_TF = (sample_prob == 0).view(size_seq).nonzero().view(-1) #select index of sampled 0
                    #identify input index with non-observed features for a later replacement process
                    index_input_na = torch.isnan(input_t[:, 0]).nonzero().view(-1)
                    #keep unique selected input indices
                    index_input_na = torch.unique(torch.cat((index_input_na, index_input_TF), dim=-1))
                else:
                    #we don't use teacher forcing throughout the evaluation process
                    #identify only input index with non-observed features for a later replacement process
                    index_input_na = torch.isnan(input_t[:, 0]).nonzero().view(-1)

                if not torch.equal(index_input_na, torch.Tensor([]).long()):
                    input_t[index_input_na, 0] = Ind_pay_pred[index_input_na, i - 1].detach() #replace I_{k,j} with the predicted probability in X_{k,j}
                    # replace Y_{k,j} with the predicted expected payment within X_{k,j}
                    input_t[index_input_na, 1] = Ind_pay_pred[index_input_na, i - 1].detach() * pay_pred[index_input_na, i - 1].detach()

                Ind_pay, pay, hidden = self.step(input_t, hidden)  #fed the prepared input to an LSTM module

            # save the predicted values in the appropriate tensor
            Ind_pay_pred[:, i] = Ind_pay.view(size_seq)
            pay_pred[:, i] = pay.view(size_seq)

        # check the dimensions of the output tensors
        pay_pred = pay_pred.view(size_seq, length_seq)
        Ind_pay_pred = Ind_pay_pred.view(size_seq, length_seq)
        pred = (Ind_pay_pred, pay_pred)
        return pred

    def predict(self, input): #for new observations, call predict with trained LSTM
        Ind_pay_pred, pay_pred = self.forward(input)
        return Ind_pay_pred, pay_pred


#---define training functions----

#the probability per epoch for the bernoulli sampler (teacher forcing)
class prob_epoch(Callback):
    def __init__(self, model):
        super().__init__()
        config.read('config_parameters.ini')  # get hyper-parameters
        self.model = model
        self.p_epoch = config.getint('train_hyp', 'p_epoch')
        self.speed = config.getfloat('train_hyp', 'speed')

    def on_epoch_begin(self, epoch, logs): #probability function in terms of epoch
        self.model.model.prob = torch.Tensor([np.exp(-self.p_epoch * int(epoch / self.p_epoch) * self.speed)])

#the network loss function
class LossWeights(nn.Module):
    def __init__(self, alpha):
        super(LossWeights, self).__init__()
        self.log_vars = torch.nn.Parameter(torch.ones(2).float()) #initialize tasks variances
        self.alpha = alpha #scale hyper-parameter

    def forward(self, y_pred, y_true): #y_pred and y_true are tuples
        Ind_pay_pred, pay_pred, = y_pred #get predicted probability and payment from the tuple y_pred
        # weights for classification loss, value 1 if target I_{k,j}!=NA
        weights_ind = 1 - torch.isnan(y_true[:, :, 0]).float()
        # weights for regression loss, value 1 if Y_{k,j}!=0 and Y_{k,j}!= NA, i.e., I_{k,j}=1
        weights_pay = 1 - (torch.isnan(y_true[:, :, 0]).float() + (y_true[:, :, 0] == 0).float())

        l1 = class_loss(Ind_pay_pred.squeeze(), y_true[:, :, 0], weights_ind) #compute classification loss
        l2 = regression_loss(pay_pred, y_true[:, :, 1], weights_pay) #compute regression loss

        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1 * l1*self.alpha + self.log_vars[0] #add task uncertainty

        precision2 = torch.exp(-self.log_vars[1])
        loss2 = precision2 * l2 + self.log_vars[1] #add task uncertainty

        return loss1 + loss2


def train(pytorch_module, train_loader, valid_loader, alpha): #define the training function
    config.read('config_parameters.ini')  #get hyper-parameters
    optimizer = optim.Adam(pytorch_module.parameters(),
                           lr=config.getfloat('train_hyp', 'learning_rate'))   #optimizer
    loss_function = LossWeights(alpha=alpha) #call the network loss function
    model = Model(pytorch_module, optimizer, loss_function) #allows to train the network without hand-coding the epoch/step logic

    #define callbacks list
    callbacks_list = [
        ReduceLROnPlateau(patience=10),
        EarlyStopping(patience=15),
        # save the latest weights to be able to continue the optimization at the end for more epochs.
        ModelCheckpoint('ckpt/last_tf_epoch.ckpt', temporary_filename='ckpt/last_tf_epoch.ckpt.tmp'),
        # save the weights in a new file when the current model is better than all previous models.
        ModelCheckpoint('ckpt/best_tf_epoch_{epoch}.ckpt', monitor='val_loss', mode='min', save_best_only=True,
                        restore_best=True, verbose=True, temporary_filename='ckpt/best_tf_epoch.ckpt.tmp'),
        # save the losses for each epoch in a TSV.
        CSVLogger('ckpt/log_tf_MT_' + str(config.getint('train_hyp', 'batch_size')) + '.tsv', separator='\t'),
        prob_epoch(model),
    ]

    model.fit_generator(train_loader, valid_loader, epochs=config.getint('train_hyp', 'epochs'),
                        callbacks=callbacks_list) #trains the network

    return model

def save_model(name, model): #to save the optimal trained network weights
    model_w = model.model.state_dict()
    torch.save(model_w, name + str('.pt'))
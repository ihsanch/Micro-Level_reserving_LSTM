## Micro-level Reserving for General Insurance Claims using a Long Short-Term Memory Network

This directory contains the material to reproduce the case study on simulated data presented in the article: Micro-level Reserving for General Insurance Claims using a Long Short-Term Memory Network.

## Abstract of the article
Detailed information about individual claims are completely ignored
when insurance claims data are aggregated and structured in development triangles for loss reserving. In the hope of extracting predictive power from the individual claims characteristics, researchers have recently proposed to move away from these macro-level methods in favor of micro-level loss reserving approaches. We introduce a discrete-time individual reserving framework incorporating granular information in a deep learning approach named Long Short-Term Memory (LSTM) neural network. The network has two tasks at each time period: classifying whether there will be a payment and predicting the amount, if any. We illustrate the estimation procedure on a simulated and a real general insurance dataset. We compare our approach with the chain-ladder aggregate method using the predictive outstanding loss estimates and their actual values. Based on a generalized Pareto model for excess payments over a threshold, we adjust the LSTM reserve prediction to account for large claims.

## Starting point:  
- R version: 3.6 or later.
- Python work environment: the 'requirement.yml' file contains the Python packages used to train the network. 

## How to proceed?
1 - Based on the simulated data,  compile LSTM_model/Datasets/data_preparation.R to obtain the training, validation and testing datasets.
2 - To train the LSTM, use python file LSTM_model/Datasets/main.py.
3 - The notebook LSTM_model/Datasets/Results.ipynb provides reserve and ultimate ratios and some illustrations.

# References
* Chaoubi, I., Besse, C. Cossette, H. & Côté, M-P. (2021). Micro-level Reserving for General Insurance Claims using a Long Short-Term Memory Network.
* Gabrielli, A., & V Wüthrich, M. (2018). An individual claims history simulation machine. Risks, 6(2), 29.

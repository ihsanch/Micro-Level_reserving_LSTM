## Code by Ihsan Chaoubi
## Article: Micro-level Reserving for General Insurance Claims using a Long Short-Term Memory Network
## Simulated data pre-processing 
## This code do not contain the data preliminary analysis

# Packages ----------------------------------------------------------------
library(data.table)
library(readr)
library(splitstackshape)
library(plyr)
library(ChainLadder)

# Import data -------------------------------------------------------------------
# set the appropriate directory
#setwd("/")
data_claims <- read_delim("Simulated_Cashflow_brute.csv", ";", escape_double = FALSE, trim_ws = TRUE)
data_claims <- as.data.table(data_claims)

# Read the functions-------------------------------------------------------------------
source("preparation_functions.R")

# Data pre-processing -----------------------------------------------------
EY <- 2005 #evaluation date
data_claims <- data_claims[AY+RepDel <= EY,] #keep only claims reported before EY
# column names
col_names <- colnames(data_claims)

# Change payment columns type
changeCols <- col_names[c(9:20)]
data_claims[,(changeCols):= lapply(.SD, as.numeric), .SDcols = changeCols]

# ultimate payment (Additional column)
data_claims[,Pay_ult:=sum(Pay00+Pay01+Pay02+Pay03+Pay04+Pay05+Pay06+Pay07+Pay08+Pay09+Pay10+Pay11),by=ClNr]
data_claims[,summary(Pay_ult)] # ultimate payment summary
data_claims[,as.list(summary(Pay_ult)),by="LoB"] # ultimate payment summary by line of business
nrow(data_claims[Pay_ult==0,])/nrow(data_claims) # claims without payment

# Initialize indicator variables r_{k,j} and I_{k,j} (see article) ----------------------
for(i in 0:11) data_claims[,paste0("Obs_Pay_",i):=-1] # initialize the indicators r_{k,j} with value -1 meaning that the value is observed
for(i in 0:11) data_claims[,paste0("Ind_Pay_",i):=1] # initialize the indicators I_{k,j} with value 1 meaning that the payment Y_{k,j} is non-zero

Ind_pay <- sapply(0:11, function(i) paste0('Ind_Pay_',i))
pay <- c(sapply(0:9, function(i) paste0('Pay0',i)),'Pay10','Pay11')
for (i in 1:12) data_claims[,Ind_pay[i]:= as.numeric(get(pay[i])!=0)] # I_{k,j}=1 if Y_{k,j}!=0 and I_{k,j}= 0 if Y_{k,j}=0 

# Data split --------------------------------------------------------------
RNGversion("3.6.3")
set.seed(20190317) #Set Seed so that same sample can be reproduced in future experiment
sample.1 <- stratified(data_claims,c("AY","LoB","RepDel"),0.6)[,ClNr] #select 60% of claimants
data_train <- data_claims[ClNr %in% sample.1, ] # get the training dataset with the claimant number from sample.1
data_train[100,]
data_train[2,]

data.2<- data_claims[!(ClNr %in% sample.1), ] # 40% of remaining claimants
sample.2 <- stratified(data.2,"AY",0.5)[,ClNr] # select 50% of claimants in data.2 equivalent to 20% of data_claims
data_valid <- data.2[ClNr %in% sample.2, ] # validation dataset
data_test <- data.2[!(ClNr %in% sample.2), ] #testing dataset

# Export datasets with all observations for later comparison----------------------------

# keep only static covariates, incremental payments Y_{k,j} and indicators I_{k,j}
data_train_brute <- data_train[,c(1:20,46:57)]
pay_true <- sapply(1:11, function(i) paste0("Pay_", i,"_true") )
Ind_pay_true <- sapply(1:11, function(i) paste0("Ind_Pay_", i,"_true") )
colnames(data_train_brute)[c(10:20,22:32)] <- c(pay_true,Ind_pay_true) #rename columns adding label "true"

data_valid_brute <- data_valid[,c(1:20,46:57)]
colnames(data_valid_brute)[c(10:20,22:32)] <- c(pay_true,Ind_pay_true)

data_test_brute <- data_test[,c(1:20,46:57)]
colnames(data_test_brute)[c(10:20,22:32)] <- c(pay_true,Ind_pay_true)

write.table(data_train_brute, "./Simulated_Cashflow_train_brute_LSTM.csv", sep=";", row.names=FALSE)
write.table(data_valid_brute, "./Simulated_Cashflow_valid_brute_LSTM.csv", sep=";", row.names=FALSE)
write.table(data_test_brute, "./Simulated_Cashflow_test_brute_LSTM.csv", sep=";", row.names=FALSE)


# Coefficients CL (based on data_train) -------------------------------------------
# Cumulative cash flows
cum_CF <- ddply(data_train, .(AY), summarise, CF00=sum(Pay00),CF01=sum(Pay01),CF02=sum(Pay02),CF03=sum(Pay03),CF04=sum(Pay04),CF05=sum(Pay05),CF06=sum(Pay06),CF07=sum(Pay07),CF08=sum(Pay08),CF09=sum(Pay09),CF10=sum(Pay10),CF11=sum(Pay11))[,2:13]
for (j in 2:12){cum_CF[,j] <- cum_CF[,j-1] + cum_CF[,j]}
# Mack chain-ladder analysis
tri_dat <- array(NA, dim(cum_CF))
for (i in 0:11){
  for (j in 0:(11-i)) tri_dat[i+1,j+1] <- cum_CF[i+1,j+1]
}
tri_dat <- as.triangle(as.matrix(tri_dat))
dimnames(tri_dat)=list(origin=0:11, dev=0:11)
n <- 12
coef_CL <- sapply(1:(n-1),function(i) sum(tri_dat[c(1:(n-i)),i+1])/sum(tri_dat[c(1:(n-i)),i]))

### Chain-ladder ratios for validation data 
RR(summary_CL(data_valid,coef_CL))

### Chain-ladder ratios for testion data 
RR(summary_CL(data_test,coef_CL))

# censored training dataset-----------------------------------------
#Censored stochastic features at the evaluation date EY=2005
T1 <-Sys.time() 
CLM_cens_train <- cens_id(data_train) #identify claims with development beyond 2005
length(CLM_cens_train)/nrow(data_train) #percentage of claims with patterns to censored (masked)
data_train <- Mask_pattern(data_train,CLM_cens_train)
difftime(Sys.time(), T1) 

# censored validation dataset-----------------------------------------
#Censored stochastic features at the evaluation date EY=2005
T1 <-Sys.time() 
CLM_cens_valid <- cens_id(data_valid) #identify claims with development beyond 2005
length(CLM_cens_valid)/nrow(data_valid) #percentage of claims with patterns to censored (masked)
data_valid <- Mask_pattern(data_valid,CLM_cens_valid)
difftime(Sys.time(), T1) 

# censored testing dataset-----------------------------------------
#Censored stochastic features at the evaluation date EY=2005
T1 <-Sys.time() 
CLM_cens_test <- cens_id(data_test) #identify claims with development beyond 2005
length(CLM_cens_test)/nrow(data_test) #percentage of claims with patterns to censored (masked)
data_test <- Mask_pattern(data_test,CLM_cens_test)
difftime(Sys.time(), T1) 

# Export datasets for LSTM network use ----------------------------
write.table(data_train, "./Simulated_Cashflow_train_LSTM.csv", sep=";", row.names=FALSE)
write.table(data_valid, "./Simulated_Cashflow_valid_LSTM.csv", sep=";", row.names=FALSE)
write.table(data_test, "./Simulated_Cashflow_test_LSTM.csv", sep=";", row.names=FALSE)

##-------------- functions script -----------##
# Evaluation year T
EY <- 2005


# Compute reserve and ultimate with Chain-ladder

summary_CL <- function(data,coef_CL){
  cum_CF <- ddply(data, .(AY), summarise, CF00=sum(Pay00),CF01=sum(Pay01),CF02=sum(Pay02),CF03=sum(Pay03),CF04=sum(Pay04),CF05=sum(Pay05),CF06=sum(Pay06),CF07=sum(Pay07),CF08=sum(Pay08),CF09=sum(Pay09),CF10=sum(Pay10),CF11=sum(Pay11))[,2:13]
  for (j in 2:12){cum_CF[,j] <- cum_CF[,j-1] + cum_CF[,j]}
  tri_dat <- array(NA, dim(cum_CF))
  reserves <- data.frame(array(0, dim=c(12+1,3)))
  reserves <- setNames(reserves, c("true Res.","CL Res.","MSEP^(1/2)"))
  for (i in 0:11){
    for (j in 0:(11-i)){tri_dat[i+1,j+1] <- cum_CF[i+1,j+1]}
    reserves[i+1,1] <- cum_CF[i+1,12]-cum_CF[i+1,12-i]
    }
  reserves[13,1] <- sum(reserves[1:12,1])
  tri_dat <- as.triangle(as.matrix(tri_dat))
  dimnames(tri_dat)=list(origin=1:12, dev=1:12)
  Mack <- MackChainLadder(tri_dat, est.sigma="Mack")
  Tab <- matrix(numeric(12*7),ncol=7)
  Tab[,1] <- 0:11
  Tab[,2] <- summary(Mack)$ByOrigin[,1]
  Tab[,3] <- cum_CF[,12]
  Tab[,4] <- Tab[,3]-Tab[,2]
  Tab[,5] <- summary(Mack)$ByOrigin[,3]-Tab[,2]
  Tab[,6] <- Tab[,5]/Tab[,4]
  Tab[,7] <- summary(Mack)$ByOrigin[,3]
  colnames(Tab) <- c("AY", "Paid until 2005","True ultime","True reserve", "Pred reserve","r1","r2")
  return(Tab)
}
# Compute CL reserve and utimate ratios

RR <- function(Tab_CL){
  print(paste("reserve ratio",eval(sum(Tab_CL[,5])/sum(Tab_CL[,4]))))
  print(paste("ultimate ratio",sum(Tab_CL[2:12,7])/sum(Tab_CL[2:12,3])))
}

# identification of claims with accident year AY > 1994 -------------------
# otherwise all the 12 years of development are observed before 2005 
# and no need to censor
cens_id <- function(data_name) unlist(data_name[AY>1994,ClNr]) 

# censored (masked) patterns at the evaluation year 2005 ------------------------------------------------------
Mask_pattern <- function(data_name,CLM_id){ # CLM_id is a vector of claimant numbers with pattern to be masked
  col_names <- colnames(data_name)
  for(j in 1:length(CLM_id)){ 
    index_row <- which(data_name[,ClNr]==CLM_id[j]) # based on the claimant number, we get the associated index row in data_name 
    CLM_AY <- data_name[index_row,AY] # get the accident year 
    for(k in (EY-CLM_AY+1):11){ 
      # for all years beyond EY=2005 until the 12th development year
      # the first year is always observed for all claims
      # we need to check the remaining 11 years.
      set(data_name, i=index_row, j=col_names[9+k], val=  NA) # incremental payment Y_{k,j} = NA
      set(data_name, i=index_row, j=col_names[34+k], val= 1)  # indicator r_{k,j} = 1 meaning not observed period
      set(data_name, i=index_row, j=col_names[21+k], val= NA) # open indicator NA (not used in the LSTM)
      set(data_name, i=index_row, j=col_names[46+k], val= NA) # payment indicator I_{k,j} = NA 
    }
  } 
  data_name <- data_name[, Nb_Dev := EY-AY+1] # number of observed development years at the evaluation date
  return(data_name)
}

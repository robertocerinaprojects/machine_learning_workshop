# clean workspace
rm(list=ls())
# set decimals to digits instead of scientific
options(scipen=999)
# set work directory
setwd(dir = "~/Desktop/Machine Learning Intro Workshop/")

#libraries
library(data.table)
library(dplyr)
library(MASS)
library(madness)
library(zoo)

source('monitorplot.R') # https://github.com/avehtari/rhat_ess/blob/master/code/monitorplot.R
source('monitornew.R') # https://github.com/avehtari/rhat_ess/blob/master/code/monitornew.R
library(ROCit)
library(caret)
library(missRanger)
library(rpart)
library(rpart.plot)

library(parallel)
library(rstan) # need stan to fit model 
# Rstan Options
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

library(ranger)

# utils
inv_logit = function(x){exp(x)/(1+exp(x))}

# sample function is useful but buggy - if you specify a single integer it returns a sequence up to that integer
sample = function(x, size, replace = F, prob = NULL) {
  if (length(x) == 1) return(x)
  base::sample(x, size = size, replace = replace, prob = prob)
}
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## GOAL: Predict Probablity of Voting for `Leave` in Brexit referendum  ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

survey = fread('anon_survey.csv')

names(survey) = c("p_T_nextGE",#"Certainty.to.vote.in.general.election.10.is.certain" 
                  "nextGE_V",#"GEVote"                                             
                  "T_pastGE",#"DidyouvoteGE2015"                                    
                  "V_pastGE",#"Past.vote"                                          
                  "brexit_V",#"Eurefvote"                                           
                  "p_T_brexit",#"Certainty.to.vote.in.EU.referendum.10.is.certain"   
                  "gender",#"Gender"                                              
                  "age",#"Age"                                                
                  "region",#"Region"                                              
                  "ethnicity",#"Ethnicity"                                          
                  "edu",#"Education"                                           
                  "tenure",#"Housing"                                            
                  "mosaic2014",#"mosaic_2014_groups"                                  
                  "pcon",#"parliamentary_constituency"                          
                  "la"#"local_authority" 
)

# don't need mosaic, LA or post-code - we are estimating at the constituency level 
survey= survey[,!c("mosaic2014",'la')]

# clean variable levels 
# # # # # 
survey$p_T_nextGE = as.numeric(as.character(unlist(survey$p_T_nextGE)))/10
# # # # # 
survey$nextGE_V = as.factor(survey$nextGE_V)
levels(survey$nextGE_V) = c("",
                            "conservative",#"Conservative"                         
                            NA,#"Dont.know"                            
                            'green',#"Green.Party"                         
                            "labour",#"Labour"                               
                            "libdem",#"Liberal.Democrat"                     
                            "novote",#"None.-.wont.vote"                    
                            "other",#"Other.party"                          
                            "pc",#"Plaid.Cymru.-.the.Welsh.Nationalists" 
                            "snp",#"Scottish.National.Party.SNP"         
                            "ukip",#"UKIP"                                 
                            "novote")#"Would.not.vote")
# # # # # 
survey$T_pastGE = as.factor(survey$T_pastGE)
levels(survey$T_pastGE) = c(NA,NA,0,0,0,1)
survey$T_pastGE = as.numeric(as.character(unlist(survey$T_pastGE)))
# # # # # 
survey$V_pastGE = as.factor(survey$V_pastGE)
levels(survey$V_pastGE) =c(NA,
                           NA,#0
                           NA,#blank
                           "conservative",#"Conservative"                         
                           NA,#"Dont.know"   
                           NA,# "Dont.remember"
                           'green',#"Green.Party"                         
                           "labour",#"Labour"                               
                           "libdem",#"Liberal.Democrat"                     
                           "other",#"Other.party"                          
                           "pc",#"Plaid.Cymru.-.the.Welsh.Nationalists" 
                           "snp",#"Scottish.National.Party.SNP"         
                           "ukip")#"UKIP")
# # # # # 
survey$brexit_V = as.factor(survey$brexit_V)
levels(survey$brexit_V) = c(NA,
                            NA,#"Dont.know"      
                            "leave",#"Leave"          
                            "remain",#"remain"         
                            "novote")#"Would.not.vote")
# # # # # 
survey$p_T_brexit = as.numeric(survey$p_T_brexit)/10
# # # # # 
survey$gender = as.factor(survey$gender)
levels(survey$gender) = c("2. Female",#"Female"             
                          "1. Male",#"Male"
                          NA)
# # # # # 
survey$age.numeric = as.numeric(as.character(unlist(survey$age)))
survey$age = as.numeric(as.character(unlist(survey$age)))
survey$age = ifelse(survey$age>100,NA,survey$age)
survey$age = cut(survey$age,
                 breaks = c(-1,17,24,34,49,64,max(survey$age,na.rm=TRUE)),
                 labels = c("0. 0-17","1. 18-24","2. 25-34","3. 35-49","4. 50-64","5. 65+"))
# # # # # 
survey$region = as.factor(survey$region)
levels(survey$region) = c(
  #NA,#"0"                        
  "4. East Midlands",#"East.Midlands"            
  "6. East of England",#"East.of.England"          
  "7. London",#"London"                   
  "1. North East",#"North.East"              
  "2. North West",#"North.West"               
  NA,#"Prefer.not.to.say"        
  "11. Scotland",#"Scotland"                 
  "8. South East",#"South.East"               
  "9. South West",#"South.West"              
  NA,#"Unknown"
  "10. Wales",#"Wales"                    
  "5. West Midlands",#"West.Midlands"            
  "3. Yorkshire and the Humber"#"Yorkshire.and.The.Humber"
)
# # # # # 
survey$ethnicity = as.factor(survey$ethnicity)
levels(survey$ethnicity) = c(NA,
                             "3. Asian",#"Asian./.Asian.British:.Bangladeshi"          
                             #"3. Asian",#"Asian./.Asian.British:.Chinese"              
                             "3. Asian",#"Asian./.Asian.British:.Indian"              
                             "3. Asian",#"Asian./.Asian.British:.Other.Asian"          
                             "3. Asian",#"Asian./.Asian.British:.Pakistani"            
                             "3. Asian",#"Asian/Asian.British"                        
                             "2. Black",#"Black./.African./.Caribbean./.Black.British" 
                             "2. Black",#"Black/Black.British"                         
                             #"1. White",#"Gypsy./.Traveller./.Irish.Traveller"        
                             "4. Other",#"Mixed"                                      
                             "4. Other",#"Mixed./.Multiple.Ethnic.Groups"              
                             "4. Other",#"Other.ethnic.group"                         
                             "4. Other",#"Other.Ethnic.Group"                          
                             NA,#"Prefer.not.to.say"                           
                             "1. White")#"White")
# # # # #
survey$edu = as.factor(survey$edu)
levels(survey$edu) = c(NA,#"0",                        
                       "6. Other",#"Apprenticeship",# not quite accurate but will do     
                       NA,#"Dont.know",                
                       "2. Level 1",#"Level.1",                  
                       "3. Level 2",#"Level.1.or.2",            
                       "3. Level 2",#"Level.2",                  
                       "4. Level 3",#"Level.3",                  
                       "5. Level 4",#"Level.4",                  
                       "1. No Qualifications",#"No.formal.qualifications", 
                       "1. No Qualifications",#"No.qualifications",       
                       "6. Other")#"Other")
# # # # # 
survey$tenure = as.factor(survey$tenure)
levels(survey$tenure) = c(NA,#""                                                                             
                          "2. Owns with a mortgage or loan",#"Being.bought.on.a.mortgage"                                                   
                          "5. Lives here rent-free",#"Living.rent.free"                                                             
                          NA,#"Other/dont.know"                                                              
                          NA,#"Other/Dont.know"                                                              
                          "1. Owns outright",#"Owned.outright.by.household"                                                  
                          "2. Owns with a mortgage or loan",#"Owned.with.a.mortgage.or.loan"                                                
                          NA,#"Prefer.not.to.say"                                                            
                          "4. Rents (with or without housing benefit)",#"Privately.rented.from.someone.other.than.a.private.landlord.or.letting.agency"
                          "4. Rents (with or without housing benefit)",#"Rented.from.a.private.landlord.or.letting.agency"                             
                          "4. Rents (with or without housing benefit)",#"Rented.from.council/local.authority"                                          
                          "4. Rents (with or without housing benefit)",#"Rented.from.Housing.Association"                                              
                          "4. Rents (with or without housing benefit)",#"Rented.from.Local.Authority"                                                  
                          "4. Rents (with or without housing benefit)",#"Rented.from.private.landlord"                                                 
                          "3. Part-owns and part-rents (shared ownership)",#"Shared.ownership.part.owned.and.part.rented"                                  
                          "4. Rents (with or without housing benefit)"#"Social.housing.rented.other.than.from.council/local.authority"
)


# # # Load data at the constituency level 
pcon_vars = read.csv("BES-2015-General-Election-results-file-v2.21.csv")
# selwct a useful subset
pcon_vars = pcon_vars[,c("ONSConstID",
                         "UKIP15",
                         "c11PopulationDensity",
                         "c11PassportEU",
                         "c11Christian",
                         "c11Muslim",
                         "c11IndustryAgriculture",
                         "c11Degree",
                         "c11Retired",
                         "c11EthnicityAsian",
                         "c11EthnicityBlack",
                         "c11Employed",
                         "c11NSSECLongtermUnemployed",
                         "c11Deprived4"
)]
# complete dataset 
library(missRanger)
pcon_vars_complete = missRanger::missRanger(data = pcon_vars,verbose = T,returnOOB = T,maxiter = 30)

## ## ## ## ## Full Model Comparison
library(mltools)
## get complete cases
survey.complete = survey[complete.cases(survey),]
## only focus on voters
survey.complete.voters = survey.complete[which(survey.complete $p_T_brexit>=0.8),]
## brexit vote - outcome
y = survey.complete.voters$brexit_V
## area-level covariates
X.area = as.data.table(pcon_vars_complete)[match(survey.complete.voters$pcon,pcon_vars_complete$ONSConstID),!"ONSConstID"]
## individual-level covariates
## one-hot it to feed it to STAN
X.ind = one_hot(dt = survey.complete.voters[,!c("age.numeric","pcon","p_T_nextGE","T_pastGE","nextGE_V","brexit_V","p_T_brexit")])
## drop covariates which have no variance (all)
X.ind = X.ind[,!apply(X.ind,2,function(x){sd(x)<0.1}),with=F]

## drop reference categories
X.ind = X.ind[,!c("V_pastGE_conservative",
                  "gender_1. Male",
                  "age_1. 18-24",
                  "region_7. London",
                  "ethnicity_1. White",
                  "edu_3. Level 2",
                  "tenure_1. Owns outright")]
## put these together 
X = cbind(X.area,X.ind)
## scale the parameters
X = as.matrix(cbind(intercept=1,scale(X)))



## leave 15% of the dataset for test 
train_id = sample(1:dim(X)[1],size = 0.85*dim(X)[1])

##put data together a list to give 
data_list = list(y = ifelse(y=="leave",1,0)[train_id],
                 n = dim(X[train_id,])[1],
                 X = X[train_id,],
                 p = dim(X[train_id,])[2])

## define model 
model_linear = "data{
   int<lower = 1> n; // total number of observations
   int<lower = 1> p; // number of covariates in design matrix
vector<lower = 0>[n] y; // vector of outcomes
     matrix[n, p] X; // design matrix
}

parameters{
       vector[p] beta;
  real<lower=0> sigma;
}

transformed parameters{
vector[n] mu;
mu =  X * beta; 
}

model{
// likelihood
y ~ normal(mu,sigma);
}

generated quantities{
vector[n] log_lik;
vector[n] y_gen;
  
for (i in 1:n) {
log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  y_gen[i] = normal_rng(mu[i], sigma);
} }
"

# # # parameters to monitor
pars_linear = c("log_lik","y_gen",'beta','sigma')

# # # fit the model 
fit_linear <-stan(model_code = model_linear, 
                  data = data_list, 
                  iter = 1500,
                  warmup = 1000,
                  thin = 4,
                  cores = 4,
                  chains = 4,
                  control = list(max_treedepth = 10, adapt_delta = 0.8),
                  verbose = T)


## define model 
model_logit = "data{
   int<lower = 1> n; // total number of observations
   int<lower = 1> p; // number of covariates in design matrix
int<lower = 0> y[n]; // vector of outcomes
     matrix[n, p] X; // design matrix
}

parameters{
       vector[p] beta;
}

transformed parameters{
vector[n] mu;
mu =  X * beta; 
}

model{
// likelihood
y ~ bernoulli_logit(mu);
}

generated quantities{
vector[n] log_lik;
vector[n] y_gen;
  
for (i in 1:n) {
log_lik[i] = bernoulli_logit_lpmf(y[i] | mu[i]);
  y_gen[i] = bernoulli_logit_rng(mu[i]);
} }
"

# # # parameters to monitor
pars_logit = c("log_lik","y_gen",'beta')


# # # fit the model 
fit_logit <-stan(model_code = model_logit, 
                  data = data_list, 
                  iter = 1500,
                  warmup = 1000,
                  thin = 4,
                  cores = 4,
                  chains = 4,
                  control = list(max_treedepth = 10, adapt_delta = 0.8),
                  verbose = T)

## define model 
model_logit_ridge = "data{
   int<lower = 1> n; // total number of observations
   int<lower = 1> p; // number of covariates in design matrix
int<lower = 0> y[n]; // vector of outcomes
     matrix[n, p] X; // design matrix
}

parameters{
      vector[p] beta_center;
     real<lower = 0> lambda;
}

transformed parameters{
                           vector[n] mu;
  real<lower = 0> sigma_beta = 1/lambda;
vector[p] beta = beta_center*sigma_beta;

                    mu =  X * beta; 
}

model{
  beta_center ~ normal(0,1);
       lambda ~ cauchy(0,1);

// likelihood
y ~ bernoulli_logit(mu);
}

generated quantities{
vector[n] log_lik;
vector[n] y_gen;
  
for (i in 1:n) {
log_lik[i] = bernoulli_logit_lpmf(y[i] | mu[i]);
  y_gen[i] = bernoulli_logit_rng(mu[i]);
} }
"

# # # parameters to monitor
pars_logit_ridge = c("log_lik","y_gen",'beta','lambda')

# # # fit the model 
 fit_logit_ridge <- stan(model_code = model_logit_ridge, 
                         data = data_list, 
                         iter = 1500,
                         warmup = 1000,
                         thin = 4,
                         cores = 4,
                         chains = 4,
                         control = list(max_treedepth = 10, adapt_delta = 0.8),
                         verbose = T)
 

# # # define model 
fit_cart = 
rpart(formula = y~.,
      data = data.table(cbind(y = data_list$y,data_list$X[,-1])),
      control = rpart.control(minsplit = 20, 
                              minbucket = round(20/3), 
                              cp = 0.01,
                              maxcompete = 4,
                              maxsurrogate = 5, 
                              usesurrogate = 2, 
                              xval = 10,
                              surrogatestyle = 0, 
                              maxdepth = 30))
fit_cart_class = 
  rpart(formula = y~.,
        data = data.table(cbind(y = data_list$y,data_list$X[,-1])),
        control = rpart.control(minsplit = 20, 
                                minbucket = round(20/3), 
                                cp = 0.01,
                                maxcompete = 4,
                                maxsurrogate = 5, 
                                usesurrogate = 2, 
                                xval = 10,
                                surrogatestyle = 0, 
                                maxdepth = 30),
        method = 'class')

pdf(file = 'model_comparison/cart_explain.pdf',width =10,height =10)
rpart.plot::rpart.plot(fit_cart)
dev.off()

# # # define model 
dt = cbind(y = data_list$y,data_list$X)
colnames(dt) = make.names(colnames(dt),unique = T)
fit_rf = ranger(formula = y~.,data = dt,num.trees = 500,probability = T)

fit_rf_class = ranger(formula = y~.,data = dt,num.trees = 500,classification = T)

# # # make out of sample predictions
extracted_linear =extract(fit_linear , 
                          pars = c('beta','sigma'), 
                          permuted = TRUE, 
                          inc_warmup = FALSE,
                          include = TRUE)
extracted_logit =extract(fit_logit, 
                         pars = c('beta'), 
                         permuted = TRUE, 
                         inc_warmup = FALSE,
                         include = TRUE)

extracted_logit_ridge =extract(fit_logit_ridge , 
                         pars = c('beta'), 
                         permuted = TRUE, 
                         inc_warmup = FALSE,
                         include = TRUE)

prop_pred_linear = X[-train_id,] %*% colMeans(extracted_linear $beta)
prop_pred_logit = inv_logit(X[-train_id,] %*% colMeans(extracted_logit $beta))
prop_pred_logit_ridge = inv_logit(X[-train_id,] %*% colMeans(extracted_logit_ridge $beta))
prop_pred_cart = predict(fit_cart,newdata = as.data.table(X[-train_id,]))
prop_pred_cart_class = predict(fit_cart_class,newdata = as.data.table(X[-train_id,]))

X_test.rf = X[-train_id,]
colnames(X_test.rf) = make.names(colnames(X_test.rf),unique = T)
prop_pred_rf = predict(fit_rf,data = as.data.table(X_test.rf))
prop_pred_rf_class = predict(fit_rf_class,data = as.data.table(X_test.rf))

## ## regression metrics
rmse <- function(y,yhat){sqrt(mean((y-yhat)^2))}
mae <- function(y,yhat){mean(abs(y-yhat))}
prop_pred_metrics = data.table(Metric = c("mae","rmse","cor"),
                               Linear = c(mae(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_linear),
                                          rmse(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_linear),
                                          cor(y = ifelse(y=="leave",1,0)[-train_id],x = prop_pred_linear)), 
                               Logistic = c(mae(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_logit),
                                            rmse(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_logit),
                                            cor(y = ifelse(y=="leave",1,0)[-train_id],x = prop_pred_logit)), 
                               CART =c(mae(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_cart),
                                       rmse(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_cart),
                                       cor(y = ifelse(y=="leave",1,0)[-train_id],x = prop_pred_cart)))
library(xtable)
print(xtable(prop_pred_metrics), include.rownames=FALSE)

## ## classification metrics
library('ROCit')
# ROCit objects
roc_linear <- rocit(score = as.numeric(unlist(prop_pred_linear)), class = ifelse(y=="leave",1,0)[-train_id]) 
roc_logit <- rocit(score = as.numeric(unlist(prop_pred_logit)), class = ifelse(y=="leave",1,0)[-train_id]) 
roc_cart <- rocit(score = as.numeric(unlist(prop_pred_cart_class[,"1"])), class = ifelse(y=="leave",1,0)[-train_id]) 

# plot the first, then the second, then add the legend
kplot_logit <- ksplot(roc_logit)
kplot_cart <- ksplot(roc_cart)
pdf(file = 'model_comparison/ROC_curve_comparison.pdf',width =10,height =5)
par(mfrow = c(1,2))

plot(roc_linear, col = c(1,"gray50"), 
     legend = FALSE, YIndex = FALSE)
lines(roc_logit$TPR ~ roc_logit$FPR, 
      col = 2, lwd = 2)
lines(roc_cart$TPR ~ roc_cart$FPR, 
      col = 3, lwd = 2)
legend("bottomright", col = c(1,2,3,1),
       c(paste("Linear AUC:",round(roc_linear$AUC,2)), paste("Logit AUC:",round(roc_logit$AUC,2)),paste("CART AUC:",round(roc_cart$AUC,2)),'Coin-toss'), lwd = 2,lty = c(1,1,1,2))

kplot_linear <- ksplot(roc_linear,legend = F)
polygon(x = c(kplot_linear$Cutoff,rev(kplot_linear$Cutoff)),y=c(kplot_linear$`F(c)`,rev(kplot_linear$`G(c)`)),col = adjustcolor('grey',0.3))
lines(kplot_logit$`F(c)` ~ kplot_logit$Cutoff, 
      col = c('red'), lwd = 2)
lines(kplot_logit$`G(c)` ~ kplot_logit$Cutoff, 
      col = c('lightcoral'), lwd = 2)
segments(x0 = kplot_logit$`KS Cutoff`,
         y0 = kplot_logit$`G(c)`[which(kplot_logit$Cutoff==kplot_logit$`KS Cutoff`)],
         y1 = kplot_logit$`F(c)`[which(kplot_logit$Cutoff==kplot_logit$`KS Cutoff`)],
         x1 = kplot_logit$`KS Cutoff`,col = 'orange',lwd = 3)
polygon(x = c(kplot_logit$Cutoff,rev(kplot_logit$Cutoff)),y=c(kplot_logit$`F(c)`,rev(kplot_logit$`G(c)`)),col = adjustcolor('lightcoral',0.3))

lines(kplot_cart$`F(c)` ~ kplot_cart$Cutoff, 
      col = c('darkgreen'), lwd = 2)
lines(kplot_cart$`G(c)` ~ kplot_cart$Cutoff, 
      col = c('green'), lwd = 2)
segments(x0 = kplot_cart$`KS Cutoff`,
         y0 = kplot_cart$`G(c)`[which(kplot_cart$Cutoff==kplot_cart$`KS Cutoff`)],
         y1 = kplot_cart$`F(c)`[which(kplot_cart$Cutoff==kplot_cart$`KS Cutoff`)],
         x1 = kplot_cart$`KS Cutoff`,col = 'orange',lwd = 3)
polygon(x = c(kplot_cart$Cutoff,rev(kplot_cart$Cutoff)),y=c(kplot_cart$`F(c)`,rev(kplot_cart$`G(c)`)),col = adjustcolor('green',0.3))
legend("bottomright", col = c(1,2,3,1),
       c("Linear", "Logit","CART"), lwd = 2,lty = c(1,1,1))
dev.off()


## ## Fixed v. Random Effects
## generate p separate colours
library(RColorBrewer)
n <- dim(extracted_logit_ridge$beta)[2]
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
## 
pdf(file = 'model_comparison/fixed_v_random.pdf',width =10,height =10)
par(mfrow = c(2,1))
plot(density(extracted_logit$beta[,1]),xlim = c(-2,2),ylim = c(0,5),col = 'black',
     main = 'logistic regression - fixed effects',xlab = 'logit-scale leave suppoort')
for(j in 2:dim(extracted_logit$beta)[2]){
  par(new = T)
plot(density(extracted_logit$beta[,j]),xlim = c(-2,2),ylim = c(0,5),col = 'black',
       xlab = "",ylab = "",xaxt = "n",yaxt = "n",main = "")
}
abline(v = 0,lty = 2)

plot(density(extracted_logit_ridge$beta[,1]),
     xlim = c(-2,2),ylim = c(0,5),
     col = 'black',
     main = 'logistic regression - ridge effects',xlab = 'logit-scale leave suppoort')
for(j in 2:dim(extracted_logit_ridge$beta)[2]){
  par(new = T)
  plot(density(extracted_logit_ridge$beta[,j]),xlim = c(-2,2),ylim = c(0,5),col = 'black',
       xlab = "",ylab = "",xaxt = "n",yaxt = "n",main = "")
}
abline(v = 0,lty = 2)
dev.off()

pdf(file = 'model_comparison/fixed_v_random_prediction.pdf',width =5,height =5)
plot(y = prop_pred_logit,x = prop_pred_logit_ridge,xlim = c(0,1),ylim = c(0,1),col = 'black',
     main = 'effect of regularization on logit predictions',
     xlab = 'regularized predictions',ylab = 'raw predictions')
abline(h = 0.5,lty = 2)
abline(v = 0.5,lty = 2)
abline(0,1)
fit <- loess(formula = prop_pred_logit ~  prop_pred_logit_ridge)
j = order(prop_pred_logit_ridge)
lines(x = prop_pred_logit_ridge[j],y = fit$fitted[j],col = 'red',lwd=2)
dev.off()


pdf(file = 'model_comparison/cart_v_rf_prediction.pdf',width =5,height =5)
plot(x = prop_pred_rf$predictions,y = prop_pred_cart,xlim = c(0,1),ylim = c(0,1),col = 'black',
     main = 'random forests v. CART predictions',
     xlab = 'RF predictions',ylab = 'CART predictions')
abline(h = 0.5,lty = 2)
abline(v = 0.5,lty = 2)
abline(0,1)
fit <- loess(formula =  prop_pred_cart ~  prop_pred_rf$predictions)
j = order(prop_pred_rf$predictions)
lines(x = prop_pred_rf$predictions[j],y = fit$fitted[j],col = 'red',lwd=2)
dev.off()






## ## regression metrics
rmse <- function(y,yhat){sqrt(mean((y-yhat)^2))}
mae <- function(y,yhat){mean(abs(y-yhat))}
prop_pred_metrics = data.table(Metric = c("mae","rmse","cor"),
                               Linear = c(mae(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_linear),
                                          rmse(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_linear),
                                          cor(y = ifelse(y=="leave",1,0)[-train_id],x = prop_pred_linear)), 
                               Logistic = c(mae(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_logit),
                                            rmse(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_logit),
                                            cor(y = ifelse(y=="leave",1,0)[-train_id],x = prop_pred_logit)), 
                               CART =c(mae(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_cart),
                                       rmse(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_cart),
                                       cor(y = ifelse(y=="leave",1,0)[-train_id],x = prop_pred_cart)),
                               Ridge_Logit =c(mae(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_logit_ridge),
                                       rmse(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_logit_ridge),
                                       cor(y = ifelse(y=="leave",1,0)[-train_id],x = prop_pred_logit_ridge)),
                               Random_Forest =c(mae(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_rf$predictions),
                                       rmse(y = ifelse(y=="leave",1,0)[-train_id],yhat = prop_pred_rf$predictions),
                                       cor(y = ifelse(y=="leave",1,0)[-train_id],x = prop_pred_rf$predictions))
                               )
library(xtable)
print(xtable(prop_pred_metrics), include.rownames=FALSE)

## ## classification metrics
library('ROCit')
# ROCit objects
roc_linear <- rocit(score = as.numeric(unlist(prop_pred_linear)), class = ifelse(y=="leave",1,0)[-train_id]) 
roc_logit <- rocit(score = as.numeric(unlist(prop_pred_logit)), class = ifelse(y=="leave",1,0)[-train_id]) 
roc_cart <- rocit(score = as.numeric(unlist(prop_pred_cart)), class = ifelse(y=="leave",1,0)[-train_id]) 
roc_rf <- rocit(score = as.numeric(unlist(prop_pred_rf$predictions)), class = ifelse(y=="leave",1,0)[-train_id]) 
roc_ridge <- rocit(score = as.numeric(unlist(prop_pred_logit_ridge)), class = ifelse(y=="leave",1,0)[-train_id]) 

# plot the first, then the second, then add the legend
kplot_logit <- ksplot(roc_logit)
kplot_cart <- ksplot(roc_cart)
kplot_rf <- ksplot(roc_rf)
kplot_ridge <- ksplot(roc_ridge)
pdf(file = 'model_comparison/ROC_curve_comparison_reg.pdf',width =10,height =5)
par(mfrow = c(1,2))

plot(roc_linear, col = adjustcolor(c(1,"gray50"),0.3), 
     legend = FALSE, YIndex = FALSE)
lines(roc_logit$TPR ~ roc_logit$FPR, 
      col = adjustcolor(2,0.3), lwd = 2)
lines(roc_cart$TPR ~ roc_cart$FPR, 
      col = adjustcolor(3,0.3), lwd = 2)
lines(roc_ridge$TPR ~ roc_ridge$FPR, 
      col = 4, lwd = 2)
lines(roc_rf$TPR ~ roc_rf$FPR, 
      col = 6, lwd = 2)
legend("bottomright", col = c(1,2,3,4,6,1),
       c(paste("Linear AUC:",round(roc_linear$AUC,2)), 
         paste("Logit AUC:",round(roc_logit$AUC,2)),
         paste("CART AUC:",round(roc_cart$AUC,2)),
         paste("Ridge AUC:",round(roc_cart$AUC,2)),
         paste("RF AUC:",round(roc_cart$AUC,2)),
         'Coin-toss'), lwd = 2,lty = c(1,1,1,1,1,2))

kplot_linear <- ksplot(roc_linear,legend = F)
polygon(x = c(kplot_linear$Cutoff,rev(kplot_linear$Cutoff)),y=c(kplot_linear$`F(c)`,rev(kplot_linear$`G(c)`)),col = adjustcolor(1,0.15))
lines(kplot_logit$`F(c)` ~ kplot_logit$Cutoff, 
      col = adjustcolor('red',0.1), lwd = 2)
lines(kplot_logit$`G(c)` ~ kplot_logit$Cutoff, 
      col = adjustcolor('lightcoral',0.1), lwd = 2)
segments(x0 = kplot_logit$`KS Cutoff`,
         y0 = kplot_logit$`G(c)`[which(kplot_logit$Cutoff==kplot_logit$`KS Cutoff`)],
         y1 = kplot_logit$`F(c)`[which(kplot_logit$Cutoff==kplot_logit$`KS Cutoff`)],
         x1 = kplot_logit$`KS Cutoff`,col = 'orange',lwd = 3)
polygon(x = c(kplot_logit$Cutoff,rev(kplot_logit$Cutoff)),y=c(kplot_logit$`F(c)`,rev(kplot_logit$`G(c)`)),col = adjustcolor(2,0.1))

lines(kplot_cart$`F(c)` ~ kplot_cart$Cutoff, 
      col = adjustcolor('darkgreen',0.1), lwd = 2)
lines(kplot_cart$`G(c)` ~ kplot_cart$Cutoff, 
      col = adjustcolor('green',0.1), lwd = 2)
segments(x0 = kplot_cart$`KS Cutoff`,
         y0 = kplot_cart$`G(c)`[which(kplot_cart$Cutoff==kplot_cart$`KS Cutoff`)],
         y1 = kplot_cart$`F(c)`[which(kplot_cart$Cutoff==kplot_cart$`KS Cutoff`)],
         x1 = kplot_cart$`KS Cutoff`,col = 'orange',lwd = 3)
polygon(x = c(kplot_cart$Cutoff,rev(kplot_cart$Cutoff)),y=c(kplot_cart$`F(c)`,rev(kplot_cart$`G(c)`)),col = adjustcolor(3,0.15))

lines(kplot_ridge$`F(c)` ~ kplot_ridge$Cutoff, 
      col = adjustcolor('blue',0.1), lwd = 2)
lines(kplot_ridge$`G(c)` ~ kplot_ridge$Cutoff, 
      col = adjustcolor('skyblue',0.1), lwd = 2)
segments(x0 = kplot_ridge$`KS Cutoff`,
         y0 = kplot_ridge$`G(c)`[which(kplot_ridge$Cutoff==kplot_ridge$`KS Cutoff`)],
         y1 = kplot_ridge$`F(c)`[which(kplot_ridge$Cutoff==kplot_ridge$`KS Cutoff`)],
         x1 = kplot_ridge$`KS Cutoff`,col = 'orange',lwd = 3)
polygon(x = c(kplot_ridge$Cutoff,rev(kplot_ridge$Cutoff)),y=c(kplot_ridge$`F(c)`,rev(kplot_ridge$`G(c)`)),col = adjustcolor(4,0.15))

lines(kplot_rf$`F(c)` ~ kplot_rf$Cutoff, 
      col = adjustcolor('blue'), lwd = 2)
lines(kplot_rf$`G(c)` ~ kplot_rf$Cutoff, 
      col =adjustcolor('skyblue'), lwd = 2)
segments(x0 = kplot_rf$`KS Cutoff`,
         y0 = kplot_rf$`G(c)`[which(kplot_rf$Cutoff==kplot_rf$`KS Cutoff`)],
         y1 = kplot_rf$`F(c)`[which(kplot_rf$Cutoff==kplot_rf$`KS Cutoff`)],
         x1 = kplot_rf$`KS Cutoff`,col = 'orange',lwd = 3)
polygon(x = c(kplot_rf$Cutoff,rev(kplot_rf$Cutoff)),y=c(kplot_rf$`F(c)`,rev(kplot_rf$`G(c)`)),col = adjustcolor(6,0.3))

legend("bottomright", col = c(1,2,3,4,5,1),
       c("Linear", "Logit","CART","Ridge","RF"), lwd = 2,lty = c(1,1,1,1,1))
dev.off()




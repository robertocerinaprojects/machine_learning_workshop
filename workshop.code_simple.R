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

## inspiration for SGD function: 
## https://www.r-bloggers.com/2017/02/implementing-the-gradient-descent-algorithm-in-r/


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## STOCHASTIC GRADIENT DESCENT EXAMPLE ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

# draw covariates and covariate coefficients
p = 9
X_corr = 0.1
n = 10000
Sigma = array(1,c(p,p))*X_corr
diag(Sigma) = 1
X = cbind(1,mvrnorm(n = n,mu = rep(0,p) ,Sigma = Sigma))
beta = sapply(X = 1:(p+1),function(x){rnorm(n = 1,mean = 0,sd = 1)})
y = X %*% beta + rnorm(n = n,mean = 0,sd = 1)

gradientDesc <- function(y,X, learn_rate, conv_threshold, max_iter,plot.which=NA,batch_size = 0.01) {

  # clean data 
  X = as.matrix(X)
  n =  dim(X)[1]
  
  # define formula for calculating score 
  MSE_fun <- function(beta,X,n,y){sum((y - ( X %*% beta  ))^2)/n}

  # initialize betas
  beta = sapply(1:dim(X)[2],function(x){rnorm(1,0,1)})
  # calculate current score:
  MSE = MSE_fun(beta = beta, X = X,n =n,y = y)
  
  # initialize SGD
  MSE_list = MSE
  beta_list = beta
  converged = F
  iter = 0
  while(converged == F) {
    
    # Implement the gradient descent algorithm
    sample_id = sample(1:dim(X)[1],size = batch_size*dim(X)[1])
    X_star = as.matrix(X[sample_id ,])
    y_star = y[sample_id]
    # calculate gradient function 
    gd = function(MSE_fun,beta){
      beta=madness(val=beta)
      z=MSE_fun(beta = beta, X = X_star, n = dim(X_star)[1],y = y_star)
      attr(z,"dvdx")
    } 
    delta <- gd(MSE_fun,beta = matrix(beta))
    beta <-as.numeric(unlist( beta - learn_rate* delta) ) 
    # get new score
    MSE_new <- MSE_fun(beta,X_star,n,y_star)
    
    # prepare output object 
    MSE_list = c(MSE_list,MSE_new)
    names(MSE_list) = paste("MSE_iter_",1:length(MSE_list),sep="")
    beta_list = cbind(beta_list,beta)
    colnames(beta_list) = paste("beta_iter_",1:length(MSE_list),sep="")
    

    output = list(    converged = converged,
                      MSE_list = MSE_list,
                      beta_list = beta_list)
    
    
    if(all(abs(0-delta)<=conv_threshold)) {
      converged = T
      print('convergence achieved')
      if(sum(is.na(plot.which)==0)){
        plot(x = beta_list[plot.which[1],],
             y = beta_list[plot.which[2],],
             pch = 1,cex = 1.2,col = adjustcolor('black',0.25),
             xlab = paste('beta',plot.which[1]),
             ylab = paste('beta',plot.which[2]),
             main = 'learning over iterations',
             xlim = c(min(beta_list[plot.which[1],]),max(beta_list[plot.which[1],])),
             ylim = c(min(beta_list[plot.which[2],]),max(beta_list[plot.which[2],])))
        lines(beta_list[plot.which[1],], beta_list[plot.which[2],],col = adjustcolor('black',0.25))
        text(x = beta_list[plot.which[1],], 
             y = beta_list[plot.which[2],],
             cex = 0.75,
             label = 1:dim( beta_list)[2])
      }
      return(output)
    }
    
    
    iter = iter + 1
    if(iter > max_iter) { 
      converged = T
      print('reached max.iter')
      if(sum(is.na(plot.which)==0)){
        plot(x = beta_list[plot.which[1],],
             y = beta_list[plot.which[2],],
             pch = 1,cex = 1.2,col = adjustcolor('black',0.25),
             xlab = paste('beta',plot.which[1]),
             ylab = paste('beta',plot.which[2]),
             main = 'learning over iterations',
             xlim = c(min(beta_list[plot.which[1],]),max(beta_list[plot.which[1],])),
             ylim = c(min(beta_list[plot.which[2],]),max(beta_list[plot.which[2],])))
        lines(beta_list[plot.which[1],], beta_list[plot.which[2],],col = adjustcolor('black',0.25))
        text(x = beta_list[plot.which[1],], 
             y = beta_list[plot.which[2],],
             cex = 0.75,
             label = 1:dim( beta_list)[2])
      }
      
      return(output)
    }
    
    # calculate current score:
    MSE = MSE_new
  }
}
# run simple lm for comparison
# Run the function 
lm_fit = lm(y~X-1)
SGD_fit = gradientDesc(y = y,X = X, learn_rate = 0.01,conv_threshold = 0.15,max_iter = 10000,
                       plot =NA,batch_size = 0.01)

pdf(file = paste('SGD_learning.pdf',sep=""),height = 5,width = 10)
par(mfrow = c(1,2))
plot(x = 1:dim(SGD_fit$beta_list)[2],
     SGD_fit$beta_list[1,],
     xlim = c(0,dim(SGD_fit$beta_list)[2]*1.2),
     pch = NA,ylim = c(min(SGD_fit$beta_list)-0.05*max(SGD_fit$beta_list),
                       max(SGD_fit$beta_list)+0.05*max(SGD_fit$beta_list)),
     ylab = 'beta',xlab = 'iteration',
     main = 'learning reg. coefficients')
for(k in 1:dim(SGD_fit$beta_list)[1]){ 
  lines(x = 1:dim(SGD_fit$beta_list)[2],SGD_fit$beta_list[k,]) 
  }
  points(x = rep(dim(SGD_fit$beta_list)[2],length(lm_fit$coefficients)),y = lm_fit$coefficients,pch = 16,col = 'purple')
  text(x = rep(dim(SGD_fit$beta_list)[2],length(lm_fit$coefficients))*1.1,y = lm_fit$coefficients,label = paste("beta",0:(k-1),sep="_"))

plot(x = 1:dim(SGD_fit$beta_list)[2],
     log(SGD_fit$MSE_list),
     xlim = c(0,dim(SGD_fit$beta_list)[2]),
     pch = NA,ylim = c(min(log(SGD_fit$MSE_list)),
                       max(log(SGD_fit$MSE_list))),
     ylab = 'log(MSE)',xlab = 'iteration',
     main = 'cost at iteration')  
lines(1:dim(SGD_fit$beta_list)[2],
      log(SGD_fit$MSE_list) )
dev.off()


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## BAYESIAN EXAMPLES ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
library(parallel)
library(rstan) # need stan to fit model 
# Rstan Options
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

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
## ## ## ## ## LINEAR REGRESSION  (LINEAR PROBABILITY MODEL)
## ## ## ## ## Brexit vote by UKIP vote in past election
survey.complete = survey[complete.cases(survey),]
survey.complete.voters = survey.complete[which(survey.complete $p_T_brexit>=0.8),]
y = survey.complete.voters$brexit_V
X = cbind(intercept = 1,pcon_vars_complete$UKIP15[match(survey.complete.voters$pcon,pcon_vars_complete$ONSConstID)])
p = dim(X)[2]
n = dim(X)[1]


data_list = list(y = ifelse(y=="leave",1,0),
                 n = n,
                 X = X,
                 p = p)

model = "data{
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
pars = c("log_lik","y_gen",'beta','sigma')

# # # fit the model 
fit_object <- stan(model_code = model, 
            data = data_list, 
            iter = 1000,
            warmup = 500,
            thin = 4,
            cores = 4,
            chains = 4,
            control = list(max_treedepth = 10, adapt_delta = 0.8),
            verbose = T)


# # # CONVERGENCE DIAGNOSTICS # # #
source('monitorplot.R') # https://github.com/avehtari/rhat_ess/blob/master/code/monitorplot.R
source('monitornew.R') # https://github.com/avehtari/rhat_ess/blob/master/code/monitornew.R

# calculate elevant quantities to monitor convergence (this may take a while for a large number of parameters)
mon <- monitor(fit_object)
samp <- as.array(fit_object) 
res <- monitor_extra(samp)

# # # Rhat
pdf(file = 'LinearRegression_1var/convergence_Rhat.pdf',width =10,height =7.5)
par(mfrow = c(2,2))
plot(res$Rhat,ylim = c(.99 ,1.1),ylab = "Rhat",main = "Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
plot(res$sRhat,ylim = c(.99 ,1.1),ylab = "sRhat",main = "classic split-Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
plot(res$zsRhat,ylim = c(.99 ,1.1),ylab = "zsRhat",main = "rank normalised split-Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
plot(res$zfsRhat,ylim = c(.99 ,1.1),ylab = "zfsRhat",main = "folded rank normalised split-Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
dev.off()

# # # Neff
pdf(file = 'LinearRegression_1var/convergence_neff.pdf',width =10,height =7.5)
par(mfrow = c(3,2))
plot(res$seff,ylim = c(0,max(res$seff,na.rm=T)),ylab = "seff",main = "classic ESS", xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$zseff,ylim = c(0,max(res$zseff,na.rm=T)),ylab = "zseff",main = "bulk ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$tailseff,ylim = c(0,max(res$tailseff,na.rm=T)),ylab = "tailseff",main = "tail ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$medsseff,ylim = c(0,max(res$medsseff,na.rm=T)),ylab = "medsseff",main = "median ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$madsseff,ylim = c(0,max(res$madsseff,na.rm=T)),ylab = "madsseff",main = "mad ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
dev.off()

# # # Mixing
pdf(file = 'LinearRegression_1var/traceplot_beta.pdf',width =10,height =5)
rstan::traceplot(object = fit_object,pars = 'beta')
dev.off()
pdf(file = 'LinearRegression_1var/traceplot_sigma.pdf',width =5,height =5)
rstan::traceplot(object = fit_object,pars = 'sigma')
dev.off()

# # # ppc
extracted = extract(fit_object , 
                    pars = c('beta','sigma','y_gen'), 
                    permuted = TRUE, 
                    inc_warmup = FALSE,
                    include = TRUE)


pdf(file = 'LinearRegression_1var/posterior_predictive_check.pdf',width =10,height =7.5)
gen.mean =  colMeans(extracted $y_gen)
plot(y = jitter(data_list$y[order(gen.mean)],factor = 1/10),
     x = 1:length(gen.mean),
     pch = 16,
     col = adjustcolor('black',0.5),
     main = 'posterior predictive check',
     ylab = 'leave intentions',
     xlab = 'order by pred. prob.',
     ylim = c(-0.5,1.5))
lines(y = zoo::rollmean(x = data_list$y[order(gen.mean)],k = 30,fill = NA),
      x = 1:length(gen.mean))
lines(y = gen.mean[order(gen.mean)],
      x = 1:length(gen.mean),col = 'blue',lwd = 1.5)
lines(y = apply(extracted $y_gen,2,quantile,0.1)[order(gen.mean)],
      x = 1:length(gen.mean),col = 'blue',lwd = 1.5,lty = 2)
lines(y = apply(extracted $y_gen,2,quantile,0.9)[order(gen.mean)],
      x = 1:length(gen.mean),col = 'blue',lwd = 1.5,lty = 2)
legend('topleft',legend = c('mean','80% P.I.','moving average'),lty = c(1,2,1),col = c('blue','blue','black'))
dev.off()

# # # linear regression - simulated fit
pdf(file = 'LinearRegression_1var/param_hist.pdf',width =12.5,height =3.5)
par(mfrow = c(1,3))
hist(extracted$beta[,1],main = 'intercept',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$beta[,1]),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$beta[,1]),2)),bty = "n")
hist(extracted$beta[,2],main = 'slope',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$beta[,2]),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$beta[,2]),2)),bty = "n")
hist(extracted$sigma,main = 'sampling standard deviation',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$sigma),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$sigma),2)),bty = "n")
dev.off()


# # # linear regresson - simpulated fit
pdf(file = 'LinearRegression_1var/alpha_beta_plot.pdf',width =7.5,height =7.5)

plot(y = jitter(data_list$y,factor = 1/10),x = data_list$X[,2],
     cex = 0.25,col = adjustcolor('black',0.15),ylim = c(-0.1,1.1),
     ylab = 'leave vote choice',xlab = 'UKIP 2015 vote',
     xaxt = "n")
points(x = mean(data_list$X[which(data_list$y==1),2]),y = 1,col = 'orange',lwd = 2,pch = 21)
abline(v = mean(data_list$X[which(data_list$y==1),2]),lty = 3)
points(x = mean(data_list$X[which(data_list$y==0),2]),y = 0,col = 'orange',lwd = 2,pch = 21)
abline(v = mean(data_list$X[which(data_list$y==0),2]),lty = 3)

axis(side = 1, at = seq(0,100,by = 5), label = seq(0,100,by = 5))

for(i in 1:dim(extracted$beta)[1]){
abline(a = extracted$beta[i,1],b = extracted$beta[i,2],col = adjustcolor('orange',0.1))
}
abline(a = mean(extracted$beta[,1]),b = mean(extracted$beta[,2]),col = 'purple')
arrows(x0 = 0,y0 =  mean(extracted$beta[,1]),x1 = 5,y1 =  mean(extracted$beta[,1]),length = 0.1,lwd = 1.5,col = 'purple')
intercept = mean(extracted$beta[,1])
intercept.round = round(intercept,4)
text(x = 9,y =  mean(extracted$beta[,1]),label = bquote(alpha==.(intercept.round)))

slope = mean(extracted$beta[,2])
arrows(x0 = 17.5,y0 =  17.5*slope+intercept,x1 =30,y1 = 17.5*slope+intercept,length = 0.1,lwd =  1.5,col = 'purple')
arrows(x0 = 30,y0 =  17.5*slope+intercept,x1 =30,y1 = 30*slope+intercept,length = 0.1,lwd =  1.5,col = 'purple')
slope.round = round(slope,4)
text(x = 35,y =  30*slope+intercept -0.075,label = bquote(beta==.(slope.round)))
dev.off()


## ## ## ## ## LINEAR REGRESSION  (LINEAR PROBABILITY MODEL)
## ## ## ## ## Brexit vote by UKIP vote in past election & age (numeric) 
y = survey.complete.voters$brexit_V
X = cbind(intercept = 1,
          age = survey.complete.voters$age.numeric,
          pcon_vars_complete$UKIP15[match(survey.complete.voters$pcon,pcon_vars_complete$ONSConstID)])
p = dim(X)[2]
n = dim(X)[1]


data_list = list(y = ifelse(y=="leave",1,0),
                 n = n,
                 X = X,
                 p = p)

model = "data{
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
pars = c("log_lik","y_gen",'beta','sigma')

# # # fit the model 
fit_object <- stan(model_code = model, 
                   data = data_list, 
                   iter = 1000,
                   warmup = 500,
                   thin = 4,
                   cores = 4,
                   chains = 4,
                   control = list(max_treedepth = 10, adapt_delta = 0.8),
                   verbose = T)


# # # CONVERGENCE DIAGNOSTICS # # #
source('monitorplot.R') # https://github.com/avehtari/rhat_ess/blob/master/code/monitorplot.R
source('monitornew.R') # https://github.com/avehtari/rhat_ess/blob/master/code/monitornew.R

# calculate elevant quantities to monitor convergence (this may take a while for a large number of parameters)
mon <- monitor(fit_object)
samp <- as.array(fit_object) 
res <- monitor_extra(samp)

# # # Rhat
pdf(file = 'LinearRegression_2var/convergence_Rhat.pdf',width =10,height =7.5)
par(mfrow = c(2,2))
plot(res$Rhat,ylim = c(.99 ,1.1),ylab = "Rhat",main = "Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
plot(res$sRhat,ylim = c(.99 ,1.1),ylab = "sRhat",main = "classic split-Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
plot(res$zsRhat,ylim = c(.99 ,1.1),ylab = "zsRhat",main = "rank normalised split-Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
plot(res$zfsRhat,ylim = c(.99 ,1.1),ylab = "zfsRhat",main = "folded rank normalised split-Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
dev.off()

# # # Neff
pdf(file = 'LinearRegression_2var/convergence_neff.pdf',width =10,height =7.5)
par(mfrow = c(3,2))
plot(res$seff,ylim = c(0,max(res$seff,na.rm=T)),ylab = "seff",main = "classic ESS", xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$zseff,ylim = c(0,max(res$zseff,na.rm=T)),ylab = "zseff",main = "bulk ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$tailseff,ylim = c(0,max(res$tailseff,na.rm=T)),ylab = "tailseff",main = "tail ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$medsseff,ylim = c(0,max(res$medsseff,na.rm=T)),ylab = "medsseff",main = "median ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$madsseff,ylim = c(0,max(res$madsseff,na.rm=T)),ylab = "madsseff",main = "mad ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
dev.off()

# # # Mixing
pdf(file = 'LinearRegression_2var/traceplot_beta.pdf',width =10,height =5)
rstan::traceplot(object = fit_object,pars = 'beta')
dev.off()
pdf(file = 'LinearRegression_2var/traceplot_sigma.pdf',width =5,height =5)
rstan::traceplot(object = fit_object,pars = 'sigma')
dev.off()

# # # ppc
extracted = extract(fit_object , 
                    pars = c('beta','sigma','y_gen'), 
                    permuted = TRUE, 
                    inc_warmup = FALSE,
                    include = TRUE)

pdf(file = 'LinearRegression_2var/posterior_predictive_check.pdf',width =10,height =7.5)
gen.mean =  colMeans(extracted $y_gen)
plot(y = jitter(data_list$y[order(gen.mean)],factor = 1/10),
     x = 1:length(gen.mean),
     pch = 16,
     col = adjustcolor('black',0.5),
     main = 'posterior predictive check',
     ylab = 'leave intentions',
     xlab = 'order by pred. prob.',
     ylim = c(-0.5,1.5))
lines(y = zoo::rollmean(x = data_list$y[order(gen.mean)],k = 30,fill = NA),
      x = 1:length(gen.mean))
lines(y = gen.mean[order(gen.mean)],
      x = 1:length(gen.mean),col = 'blue',lwd = 2)
lines(y = apply(extracted $y_gen,2,quantile,0.1)[order(gen.mean)],
      x = 1:length(gen.mean),col = 'blue',lwd = 2,lty = 2)
lines(y = apply(extracted $y_gen,2,quantile,0.9)[order(gen.mean)],
      x = 1:length(gen.mean),col = 'blue',lwd = 2,lty = 2)
legend('topleft',legend = c('mean','80% P.I.','moving average'),lty = c(1,2,1),col = c('blue','blue','black'))
dev.off()


# # # linear regression - simulated fit
pdf(file = 'LinearRegression_2var/param_hist.pdf',width =10,height =10)
par(mfrow = c(2,2))
hist(extracted$beta[,1],main = 'intercept',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$beta[,1]),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$beta[,1]),2)),bty = "n")
hist(extracted$beta[,2],main = 'slope (age)',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$beta[,2]),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$beta[,2]),2)),bty = "n")
hist(extracted$beta[,2],main = 'slope (UKIP vote)',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$beta[,2]),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$beta[,2]),2)),bty = "n")
hist(extracted$sigma,main = 'sampling standard deviation',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$sigma),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$sigma),2)),bty = "n")
dev.off()


# # # linear regression - fit plane 
prediction <- function(x, y){ as.matrix(cbind(1,x,y)) %*% colMeans(extracted$beta) }
# prepare variables.
x <- seq(18,100,length.out = 30)
y <- seq(0,45,length.out = 30)
z <- outer(x, y, prediction )

# plot the 3D surface
pdf(file = 'LinearRegression_2var/predicted_probabilties_plot.pdf',width =7.5,height =7.5)
persp(x, y, z,theta = -60,phi =0,
      xlab = 'age',ylab = 'UKIP 2015',zlab = "Pr(Leave)",
      cex.lab = 1,col = adjustcolor('black',0.25),
      main ='regression plane',ticktype = 'detailed',zlim = c(0,1))
dev.off()


## ## ## ## ## LOGISTIC REGRESSION 
## ## ## ## ## Brexit vote by UKIP vote in past election & age (numeric) 
y = survey.complete.voters$brexit_V
X = cbind(intercept = 1,
          age = survey.complete.voters$age.numeric,
          pcon_vars_complete$UKIP15[match(survey.complete.voters$pcon,pcon_vars_complete$ONSConstID)])
p = dim(X)[2]
n = dim(X)[1]


data_list = list(y = ifelse(y=="leave",1,0),
                 n = n,
                 X = X,
                 p = p)

model = "data{
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
pars = c("log_lik","y_gen",'beta')

# # # fit the model 
fit_object <- stan(model_code = model, 
                   data = data_list, 
                   iter = 1000,
                   warmup = 500,
                   thin = 4,
                   cores = 4,
                   chains = 4,
                   control = list(max_treedepth = 10, adapt_delta = 0.8),
                   verbose = T)

# # # CONVERGENCE DIAGNOSTICS # # #
source('monitorplot.R') # https://github.com/avehtari/rhat_ess/blob/master/code/monitorplot.R
source('monitornew.R') # https://github.com/avehtari/rhat_ess/blob/master/code/monitornew.R

# calculate elevant quantities to monitor convergence (this may take a while for a large number of parameters)
mon <- monitor(fit_object)
samp <- as.array(fit_object) 
res <- monitor_extra(samp)

# # # Rhat
pdf(file = 'LogitRegression_2var/convergence_Rhat_logit.pdf',width =10,height =7.5)
par(mfrow = c(2,2))
plot(res$Rhat,ylim = c(.99 ,1.1),ylab = "Rhat",main = "Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
plot(res$sRhat,ylim = c(.99 ,1.1),ylab = "sRhat",main = "classic split-Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
plot(res$zsRhat,ylim = c(.99 ,1.1),ylab = "zsRhat",main = "rank normalised split-Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
plot(res$zfsRhat,ylim = c(.99 ,1.1),ylab = "zfsRhat",main = "folded rank normalised split-Rhat",xlab = 'estimated parameters')
abline(h = 1.01,col= 'red',lty = 2)
dev.off()

# # # Neff
pdf(file = 'LogitRegression_2var/convergence_neff_logit.pdf',width =10,height =7.5)
par(mfrow = c(3,2))
plot(res$seff,ylim = c(0,max(res$seff,na.rm=T)),ylab = "seff",main = "classic ESS", xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$zseff,ylim = c(0,max(res$zseff,na.rm=T)),ylab = "zseff",main = "bulk ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$tailseff,ylim = c(0,max(res$tailseff,na.rm=T)),ylab = "tailseff",main = "tail ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$medsseff,ylim = c(0,max(res$medsseff,na.rm=T)),ylab = "medsseff",main = "median ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
plot(res$madsseff,ylim = c(0,max(res$madsseff,na.rm=T)),ylab = "madsseff",main = "mad ESS",xlab = 'estimated parameters')
abline(h = 400,col= 'red',lty = 2)
dev.off()

# # # Mixing
pdf(file = 'LogitRegression_2var/traceplot_beta_logit.pdf',width =10,height =5)
rstan::traceplot(object = fit_object,pars = 'beta')
dev.off()


# # # ppc
extracted = extract(fit_object , 
                    pars = c('beta','y_gen'), 
                    permuted = TRUE, 
                    inc_warmup = FALSE,
                    include = TRUE)
pdf(file = 'LogitRegression_2var/posterior_predictive_check_logit.pdf',width =10,height =7.5)
gen.mean =  colMeans(extracted $y_gen)
plot(y = jitter(data_list$y[order(gen.mean)],factor = 1/10),
     x = 1:length(gen.mean),
     pch = 16,
     col = adjustcolor('black',0.5),
     main = 'posterior predictive check',
     ylab = 'leave intentions',
     xlab = 'order by pred. prob.',
     ylim = c(-0.5,1.5))
lines(y = zoo::rollmean(x = data_list$y[order(gen.mean)],k = 30,fill = NA),
      x = 1:length(gen.mean))
lines(y = gen.mean[order(gen.mean)],
      x = 1:length(gen.mean),col = 'blue',lwd = 2)
lines(y = apply(extracted $y_gen,2,quantile,0.1)[order(gen.mean)],
      x = 1:length(gen.mean),col = 'blue',lwd = 2,lty = 2)
lines(y = apply(extracted $y_gen,2,quantile,0.9)[order(gen.mean)],
      x = 1:length(gen.mean),col = 'blue',lwd = 2,lty = 2)
legend('topleft',legend = c('mean','80% P.I.','moving average'),lty = c(1,2,1),col = c('blue','blue','black'))
dev.off()

# # # linear regression - simulated fit
pdf(file = 'LogitRegression_2var/param_hist_logit.pdf',width =10,height =10)
par(mfrow = c(2,2))
hist(extracted$beta[,1],main = 'intercept',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$beta[,1]),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$beta[,1]),2)),bty = "n")
hist(extracted$beta[,2],main = 'slope (age)',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$beta[,2]),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$beta[,2]),2)),bty = "n")
hist(extracted$beta[,2],main = 'slope (UKIP vote)',border = NA,col = 'orange',breaks = 25)
abline(v = mean(extracted$beta[,2]),lwd = 1.5,col = 'purple')
legend("topright",legend = paste("posterior mean:",round(mean(extracted$beta[,2]),2)),bty = "n")
dev.off()


# # # linear regression - fit plane 
prediction <- function(x, y){ inv_logit(as.matrix(cbind(1,x,y)) %*% colMeans(extracted$beta) )}
# prepare variables.
x <- seq(18,100,length.out = 30)
y <- seq(0,45,length.out = 30)
z <- outer(x, y, prediction )

# plot the 3D surface
pdf(file = 'LogitRegression_2var/predicted_probabilties_plot_logit.pdf',width =7.5,height =7.5)
persp(x, y, z,theta = -60,phi =0,
      xlab = 'age',ylab = 'UKIP 2015',zlab = "Pr(Leave)",
      cex.lab = 1,col = adjustcolor('black',0.25),zlim = c(0,1),
      main ='regression plane',ticktype = 'detailed')
dev.off()


# latent mu to prbability
mu_pred_id309 = sapply(1:dim(extracted$beta)[1],function(s){t(data_list$X[309,]) %*% extracted$beta[s,]})
pi_pred_id309 = inv_logit(mu_pred_id309 )
mu_pred_id332 = sapply(1:dim(extracted$beta)[1],function(s){t(data_list$X[332,]) %*% extracted$beta[s,]})
pi_pred_id332 = inv_logit(mu_pred_id332 )

pdf(file = 'LogitRegression_2var/predicted_probabilties_id_309_332_logit.pdf',width =7.5,height =7.5)
par(mfrow = c(2,2))
plot(density(mu_pred_id309 ),main = 'latent leave support\nage: 72 & ukip15: 45%',xlim = c(-4,4))
abline(v = mean(mu_pred_id309 ))
abline(v = 0,lty = 2)
plot(density(inv_logit(mu_pred_id309 )),main = 'probability of leave support\nage: 72 & ukip15: 45%',xlim = c(0,1))
abline(v = 0.5,lty = 2)
abline(v = mean(inv_logit(mu_pred_id309 )))

plot(density(mu_pred_id332 ),main = 'latent leave support\nage: 30 & ukip15: 3%',xlim = c(-4,4))
abline(v = 0,lty = 2)
abline(v = mean(mu_pred_id332 ))
plot(density(inv_logit(mu_pred_id332 )),main = 'probability of leave support\nage: 30  & ukip15: 3%',xlim = c(0,1))
abline(v = 0.5,lty = 2)
abline(v = mean(inv_logit(mu_pred_id332 )))

dev.off()

## ## check a histogram of predicted probabilities to look for an optimal threshold
pred.mean = prediction(x = data_list$X[,2],y = data_list$X[,3])
pdf(file = 'LogitRegression_2var/predicted_probabilties_histogram.pdf',width =5,height =5)
hist(pred.mean,
     main = 'predicted probabilities histogram',
     xlab = 'probability of voting `Leave',
     breaks = 150,
     col = adjustcolor('orange',0.5),
     border = adjustcolor('black',0.25) 
     )
abline(v = 0.5,lwd = 1.5)
dev.off()


## ## use ROC to identify optimal threshold
library('ROCit')
ROCit_obj <- rocit(score=as.numeric(unlist(pred.mean)),class=data_list$y)

pdf(file = 'LogitRegression_2var/ROC_curve_logit_simple.pdf',width =10,height =5)
par(mfrow = c(1,2))
plot(ROCit_obj)
abline(v = 0,h = 1,lty = 2)
points(x = 0,y = 1,pch = 16)
kplot <- ksplot(ROCit_obj)
dev.off()


## ## Calculate confusion table 
library(caret)
confusion = 
caret::confusionMatrix(data = table(
  pred = ifelse(as.numeric(unlist(pred.mean))>kplot $`KS Cutoff`,1,0),
  class=data_list$y),positive = '1')

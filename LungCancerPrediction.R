rm(list = ls()) # clear global environment
graphics.off() # close all graphics
gc()
time.begin <- proc.time()[3]


if(!"pacman" %in% rownames(installed.packages())){
  install.packages(pkgs = "pacman",repos = "http://cran.us.r-project.org")
}
# p_load is equivalent to combining both install.packages() and library()
pacman::p_load(bnlearn,pROC,MLmetrics,caret,bestglm,DMwR,smotefamily,randomForest,unbalanced,e1071,RSNNS,C50,MASS,ROSE,snow,ranger,parallel,xgboost,gbm,naivebayes,kernlab,pls,glmnet)


time.begin <- proc.time()[3]
setwd("")
data<- data.frame(read.csv("",header=TRUE))
data<- data[,-c(1,2,3,29:42)]


data.variables<-data[,c(3:25)] 
data.variables[,] <- lapply(data.variables, factor) # the categorcal variables
data<-cbind(data[,c(1,2)],data.frame(predict(dummyVars(" ~ .", data = data.variables, fullRank=T),
                                             newdata = data.variables)),data$X1y)

data<- data[, -nearZeroVar(data[,-ncol(data)])]
age.scale<-data.frame(scale(data$Age))
data[,1]<-age.scale[,1]
colnames(data)[ncol(data)]<- "survival"
table(data$survival)

cooksd <- cooks.distance(lm(survival ~ ., data=data[,]))
influential <- data.frame(as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))]))
influential<- influential[complete.cases(influential), ]
data<- data[-c(influential),]
colnames(data)[ncol(data)]<- "survival"
table(data$survival)


#create test and training
set.seed(500)
trainIndex<- createDataPartition(data$survival, p=0.7, list=F)
pretrain<- data[trainIndex, ]
pretest<- data[-c(trainIndex),]
pretrain<-pretrain[,apply(pretrain, 2, var, na.rm=TRUE) != 0]
pretrain$survival<- as.factor(pretrain$survival)
table(pretrain$survival)
set.seed(333)
library(imbalance)
newData<- imbalance::oversample(
  pretrain,
  ratio =0.7,
  method = "ADASYN",
  classAttr = "survival"
)
table(newData$survival)



pretrain<-newData[,apply( newData, 2, var, na.rm=TRUE) != 0]
rownames(pretrain) <- NULL
for  (i in 2: (ncol(pretrain)-1)){
  pretrain [,i] <- ifelse(pretrain [,i] >= 0.5, 1, 0)
}
rownames(pretrain) <- NULL
colnames(pretrain)[ncol(pretrain)] <- "survival"
pretrain$survival<- as.factor(pretrain$survival)



set.seed(123)
library(glmnet)
lambda_seq <- 10^seq(2, -2, by = -.05)
x_vars <- model.matrix(survival~. , pretrain)[,-1]
y_var <- as.numeric(pretrain$survival)-1
cv_output <- cv.glmnet(x_vars[,], y_var[], 
                       alpha = 1, family = "binomial", type.measure = "class",nfolds =5,lambda = lambda_seq)

best_lam <- cv_output$lambda.min
lasso_best <- glmnet(x_vars[,], y_var[], alpha = 1,
                     lambda = best_lam, intercept = TRUE)
nvariables<- data.frame(lasso_best$df)
deviation<- data.frame(lasso_best$dev.ratio)
lambda<-data.frame(lasso_best$lambda)
c<-coef(lasso_best,s=best_lam,exact=TRUE)
inds<-which(c!=0)
BEST<-row.names(c)[inds][-1]
BEST <- c("survival", BEST)


features<-as.matrix(c)
features<- data.frame(features)
features <- tibble::rownames_to_column(features, "VALUE")
features<-features[features$X1!=0,]
features$X1<- round(features$X1,4)


pretrain<- pretrain[,BEST]
test <- pretest[, BEST]
levels(pretrain$survival) <- c("0", "1")
Train.0 <- which(pretrain$survival[]==0)
Train.1 <- which(pretrain$survival[]==1)



Mainfun<-function(j, pretrain, test, features){
  
  if(!"pacman" %in% rownames(installed.packages())){
    install.packages(pkgs = "pacman",repos = "http://cran.us.r-project.org")
 }
  # p_load is equivalent to combining both install.packages() and library()
  pacman::p_load(bnlearn,pROC,MLmetrics,caret,bestglm,DMwR,smotefamily,randomForest,unbalanced,e1071,RSNNS,C50,MASS,ROSE,snow,ranger,parallel,xgboost,gbm,naivebayes,kernlab,pls,glmnet)
  
  Train.0 <- which(pretrain$survival[]==0)
  Train.1 <- which(pretrain$survival[]==1)

  for (k in 1:m){
  set.seed((1987+j))
  index.train <- c(sample(Train.0, length(Train.0), replace=TRUE), sample(Train.1, length(Train.1), replace=TRUE))
  train <- pretrain[index.train, ]
  survival<- length(which(test$survival[]==1))
  not.survival<- nrow(test)- survival
  
  ctrl1 <- caret::trainControl(method = "cv", number = 2, returnResamp = "all",savePredictions = TRUE, 
                               classProbs = TRUE, summaryFunction=twoClassSummary,
                               verboseIter = TRUE)
  
  train$survival<- as.factor(train$survival)
  levels(train$survival) <- c("decease", "survival")
  
  xgbGrid <- expand.grid(nrounds = c(50,100,200),  
                         max_depth = c(10, 15, 20, 25),
                         colsample_bytree = seq(0.5, 0.9, length.out = 5),
                         eta = 0.1,
                         gamma=0,
                         min_child_weight = 1,
                         subsample = 1)
  
  mod <- caret::train(survival ~ ., method="glm", data = train,trControl= ctrl1,metric = 'ROC')
  #mod <- caret::train(survival ~ ., method="xgbTree", data = train,trControl= ctrl1,tuneGrid = xgbGrid,metric = 'ROC')
  #mod <- caret::train(survival ~ ., method="nnet", data = train,trControl= ctrl1,metric = 'ROC')
  
  roc.tr <- roc(train$survival, 
                predict(mod, train, type = "prob")[,1], 
                levels = rev(levels(train$survival)))
  pred <- predict(mod, train)
  comparison <- table(train$survival,pred)
  (comparison[1,1]+comparison[2,2])/nrow(train)
  comparison[1,1]/ length(which(train$survival[]=="decease"))
  comparison[2,2]/ length(which(train$survival[]=="survival"))
  

  test$survival<- as.factor(test$survival)
  levels(test$survival) <- c("decease", "survival")

  roc0 <- roc(test$survival, 
              predict(mod, test, type = "prob")[,1], 
              levels = rev(levels(test$survival)))
  pred <- predict(mod, test)
  comparison <- table(test$survival,pred)
  accuracy<- (comparison[1,1]+comparison[2,2])/nrow(test)
  sen.recall<- comparison[1,1]/ (not.survival)
  spe<- comparison[2,2]/ (survival)
  precision<-comparison[1,1]/(comparison[1,1]+comparison[2,1])
  f1<- 2*(sen.recall*precision)/(sen.recall+precision)
  gm  <- sqrt(sen.recall*spe)
  roc<- roc0$auc
  
  metric.1<- c(j,sen.recall, spe, accuracy, precision, f1, gm, roc)
  BEST<- c(colnames(train))
  ### save output values as a list
  Optimal_result <- list(metric.1,features)
  return(Optimal_result)
}
}


library(snow)   ### for the parallel computation
cl <- makeCluster(3, type="SOCK") ### number of cores in your computer
ncases <- 10
Result <- parSapply(cl, 1:ncases, Mainfun, pretrain, test, features)
stopCluster(cl)
time.end <- (proc.time()[3]-time.begin)/60
paste("It took", time.end, "minutes to run the program.")


metric <- data.frame(matrix(0, nrow = ncases, ncol = 8))
colnames(metric) <- c("model","sensitivity","specificity", "accuracy", "precision","f1", "gm", "AUC") 
best_features <- list()


for (i in 1:ncases){
  metric[i,] <- as.vector(Result[[2*i-1]])
  best_features[[1]] <- Result[[(2)]]
}

best_features<- (data.frame(plyr::ldply(best_features, rbind)))
mean.model <- apply(metric, 2, mean) 
sd.model <- apply(metric, 2, sd)
all.metrics <- rbind(metric,mean.model,sd.model)
best_features<-merge(data.frame(best_features, row.names=NULL), data.frame(features, row.names=NULL), 
                     by = 0, all = TRUE)[,-1]


modelname <- ""
sampling<- ""
feature.selection <- ""
year <- ""
name<-paste(year,modelname,sampling,feature.selection, sep="")
setwd("")
write.csv(all.metrics,paste(name,"_","model.csv", sep=""))
write.csv(best_features,paste(name,"_","feCo.csv"))


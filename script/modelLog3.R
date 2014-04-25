##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Logistic Regression 3: Ensemble
## 4/24/2014
## Updated: 4/25/2014

## cross validation on training set
## use sort.var as the variable selection list

library(caret)
library(caretEnsemble)
library(glmnet)
library(pROC)

# enable multicore parallelization
library(doParallel)
registerDoParallel(3)


# define trainControl
ensemble.control <- trainControl(method="repeatedcv", number=10, repeats=3, 
  classProb=T, summaryFunction = twoClassSummary, verboseIter=F, allowParallel=T,
  returnResamp='none', returnData=FALSE, savePredictions=TRUE)

# define tuning grid
glmnet.grid <- expand.grid(.alpha=0, .lambda=seq(0.0,0.2,by=0.01))

# preserved demographic variables
demo.var <- names(train.ex)[2:11]

# a grid of the number of variables to select 
var.lengths <- seq(10, 100, by=10)

# the iterative function for training a series of models
# arguments: dat = train data
#            var.set = full list of variables
#            nvar.grid = list of numbers of top variables to select at each iteraction
#            param.grid = if in the tuning mode, a data.frame of expand.grid; otherwise a list of fixed set of parameters
#            tuning = T/F, enable/disable tuning mode

iter.train <- function(dat, var.set, nvar.grid, param.grid=glmnet.grid, tuning=T){
  set.seed(86)
  ensemble.control$index <- createMultiFolds(dat$Happy, k=10, times=3)
  niter <- length(nvar.grid)

  if(tuning){
    pg.list <- rep(list(param.grid), niter)}
  else{
    pg.list <- param.grid}

  models <- list()
  # train a series of models according to the number of selected variables
  for(i in 1:niter){
    topn <- nvar.grid[i]
    vset <-  unique(c(demo.var, as.character(var.set$Var)[1:topn]))
    #print(vset)
    print(pg.list[[i]])
    model <- train(Happy ~. + Gender * EducationLevel, 
      data=dat[vset], method='glmnet', family="binomial",
      metric="ROC",
      tuneGrid=pg.list[[i]],
      trControl=ensemble.control)
    print(model)
    models <- c(models, list(model))
  }

  names(models) <- paste0("model", 1:length(nvar.grid))
  return(models)
}

log.models <- iter.train(train.ex, sort.var, var.lengths)

# use greedy ensemble and linear regression for blending
greedy.mix <- caretEnsemble(log.models)
lin.mix <- caretStack(log.models, method = "glm", trControl = trainControl(method = "cv"))

# in-sample ROC: 0.7745039
colAUC(predict(greedy.mix, newdata=train.ex), train.ex$Happy) # **** 0.7745039 ****
colAUC(predict(lin.mix, newdata=train.ex, type="prob")[,2], train.ex$Happy) # **** 0.7721397 ****

# out-of-sample ROC: 0.7374906
preds <- data.frame(sapply(log.models, function(x){predict(x, test.ex, type='prob')[,2]}))
preds$greedy <- predict(greedy.mix, newdata=test.ex)
preds$linear <- predict(lin.mix, newdata=test.ex, type='prob')[,1]
sort(data.frame(colAUC(preds, test.ex$Happy)))
#     model1   model10    model9    model3    model8    model5    model4    model6    model7    model2    greedy
#  0.7217596 0.7276473 0.7291923 0.7296113 0.7300854 0.7303415 0.7316875 0.7317002 0.7335309 0.7362758 0.7374229
#     linear
#  0.7374906

##### final model trained on soh with parameters retuned ####
## use sort.var.soh as the variable selection list
sort.var.soh

log.models.final <- iter.train(soh.ex, sort.var.soh, var.lengths)

# inspect the tuned parameters and cv ROCs of each model
sapply(log.models.final, function(X) X$bestTune)
sapply(log.models.final, function(X) max(X$results$ROC)) # best single model is model 6
#   model1    model2   model10    model9    model3    model8    model5    model7 
#0.7190967 0.7394903 0.7397630 0.7402042 0.7407011 0.7410861 0.7425521 0.7430176 
#   model4    model6 
#0.7434172 0.7438601 

# blend models into ensembles
greedy.mix.final <- caretEnsemble(log.models.final)
lin.mix.final <- caretStack(log.models.final, method = "glm", trControl = trainControl(method = "cv"))
greedy.mix.final$error # cv ROC: 0.7462623

# in-sample ROC: 0.770
colAUC(predict(greedy.mix.final, newdata=soh.ex), soh.ex$Happy) # **** 0.7701926 ****
colAUC(predict(lin.mix.final, newdata=soh.ex, type="prob")[,1], soh.ex$Happy) # **** 0.7706582 ****
colAUC(predict(log.models.final$model6, newdata=soh.ex, type="prob")[,2], soh.ex$Happy) # **** 0.7717141 ****

# predict on soh.test.ex
pred.mix <- as.data.frame(list(UserID=soh.test.ex$UserID)) 
pred.mix$greedy <- predict(greedy.mix.final, newdata=soh.test.ex)  # **** 0.74128 ****
pred.mix$linreg <- predict(lin.mix.final, newdata=soh.test.ex, type="prob")[,1]  # **** 0.74097 ****
pred.mix$model6 <- predict(log.models.final$model6, newdata=soh.test.ex, type="prob")[,2] # **** 0.73388 ****

# export
write.table(pred.mix[1:2], file="./result/greedyMix1.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1"))

write.table(pred.mix[c(1,3)], file="./result/linearMix1.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1"))

write.table(pred.mix[c(1,4)], file="./result/logNet3.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1"))

####### The final models with retuned parameters suffers from overfitting issue #######
## back to final models with optimal parameters given by cross validation
## use sort.var as the variable selection list

# get a list of best tuned parameters
best.param.list <- lapply(log.models, function(X) X$bestTune)

log.models.final2 <- iter.train(soh.ex, sort.var, var.lengths, best.param.list, tuning=F)
# inspect the tuned parameters and cv ROCs of each model
sapply(log.models.final2, function(X) X$bestTune)
sapply(log.models.final2, function(X) max(X$results$ROC))

# blend models into ensembles
# weights computed using the final models
greedy.mix.final2 <- caretEnsemble(log.models.final2)
greedy.mix.final$error # cv ROC: 0.7457306

# weights from the greedy.mix training
rbind(names(greedy.mix$models), greedy.mix$weight)

# in-sample ROC: 0.767971
colAUC(predict(greedy.mix.final2, newdata=soh.ex), soh.ex$Happy) # **** 0.767971 ****

pred2.tr.mix <- data.frame(sapply(log.models.final2[names(greedy.mix$models)], function(x){predict(x, soh.ex, type='prob')[,2]}))
pred2.tr.mix$greedy <- as.matrix(pred2.tr.mix) %*% greedy.mix$weights
colAUC(pred2.tr.mix$greedy, soh.ex$Happy) # **** 0.7656768 ****

# predict
pred.mix <- as.data.frame(list(UserID=soh.test.ex$UserID)) 
pred.mix$greedy <- predict(greedy.mix.final2, newdata=soh.test.ex)  # **** 0.74004 ****

pred2.mix <- data.frame(sapply(log.models.final2[names(greedy.mix$models)], function(x){predict(x, soh.test.ex, type='prob')[,2]}))
pred2.mix$greedy <- as.matrix(pred2.mix) %*% greedy.mix$weights  # **** 0.73951 ****
pred2.mix$id <- soh.test.ex$UserID  # **** 0.73951 ****

write.table(pred.mix[1:2], file="./result/greedyMix2.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1")) 

write.table(pred2.mix[c("id", "greedy")], file="./result/greedyMix3.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1"))

save(log.models, log.models.final, ensemble.control, demo.var, var.lengths, iter.train, greedy.mix, lin.mix, file="logMix.Rda")


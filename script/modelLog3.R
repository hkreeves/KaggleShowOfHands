##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Logistic Regression 3: Ensemble
## 4/24/2014

## cross validation on training set
## use sort.var as the variable selection list

library(caret)
library(caretEnsemble)

# define trainControl
ensemble.control <- trainControl(method="repeatedcv", number=10, repeats=3, 
  classProb=T, summaryFunction = twoClassSummary, verboseIter=FALSE,
  returnResamp='none', returnData=FALSE, savePredictions=TRUE)

# define tuning grid
glmnet.grid <- expand.grid(.alpha=0, .lambda=seq(0.0,0.2,by=0.01))

# preserved demographic variables
demo.var <- names(train.ex)[2:11]

# a grid of the number of variables to select 
var.lengths <- seq(10, 100, by=10)

# the iterative function for training a series of models
iter.train <- function(dat, var.set, nvar.grid){
  set.seed(86)
  ensemble.control$index <- createMultiFolds(dat$Happy, k=10, repeats=3)

  models <- list()
  # train a series of models according to the number of selected variables
  for(topn in nvar.grid){
    vset <-  unique(c(demo.var, as.character(var.set$Var)[1:topn]))
    model <- train(Happy ~. + Gender * EducationLevel, 
      data=dat[vset], method='glmnet', family="binomial",
      metric="ROC",
      tuneGrid=glmnet.grid,
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
preds$linear <- predict(lin.mix, newdata=test.ex, type='prob')[,2]
sort(data.frame(colAUC(preds, test.ex$Happy)))
#     model1   model10    model9    model3    model8    model5    model4    model6    model7    model2    greedy
#  0.7217596 0.7276473 0.7291923 0.7296113 0.7300854 0.7303415 0.7316875 0.7317002 0.7335309 0.7362758 0.7374229
#     linear
#  0.7374906

save(log.models, ensemble.control, demo.var, var.lengths, iter.train, greedy.mix, lin.mix, preds, file="logMix.Rda")


##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Model Selection 3: Logistic
## 4/22/2014

# As a countinuation of Model Selection 2

#############
# now try logistic regression
log1 <- glm(Happy ~ ., data=train[-1], family="binomial")
summary(log1)

# in-sample ROC: 0.79834353
pred.tr.log1 <- predict(log1, type="response")
roc.tr.log1 <- prediction(pred.tr.log1, train$Happy)
performance(roc.tr.log1, "auc")@y.values # **** 0.7984353 ****
plot(performance(roc.tr.log1, "sens", "fpr"))

# out-of-sample ROC: 0.712729
pred.te.log1 <- predict(log1, newdata=test, type="response")
roc.te.log1 <- prediction(pred.te.log1, test$Happy)
performance(roc.te.log1, "auc")@y.values # ****  0.712729 ****
plot(performance(roc.te.log1, "sens", "fpr"))

############
# now try a glmnet logistic regression
cv.control

### some failed attempts ##
myFunc <- caretFuncs
myFunc$summary <- twoClassSummary
myFunc$rank <- function(object, x, y) {
    vimp <- sort(object$finalModel$beta[, 1])
    vimp <- as.data.frame(vimp)
    vimp$var <- row.names(vimp)
    vimp$Overall <- seq(nrow(vimp), 1)
    vimp
}
rfe.control <- rfeControl(functions = myFunc, method = "cv", number = 10, 
    rerank = FALSE, returnResamp = "final", saveDetails = FALSE, verbose = F)

set.seed(138)
rfe1 <- rfe(x=train[-(1:2)], y=train$Happy, sizes = seq(10,40,by=10),
  metric = "ROC", maximize=TRUE, rfeControl = rfe.control,
  method='glmnet', family="binomial",
  tuneGrid = expand.grid(.alpha=c(0,1), .lambda=c(0.1, 0.15, 0.2)),
  trControl = cv.control)
#####

set.seed(139)
log.net <- train(Happy ~., data=train[-1], method='glmnet', family="binomial",
  metric = "ROC",
  tuneGrid = expand.grid(.alpha=c(0,0.2,0.4), .lambda=seq(0.15,0.17,by=0.005)),
  trControl=cv.control)
log.net # cv ROC: 0.738, alpha=0 (ridge), lambda=0.16
#  lambda  ROC    Sens   Spec   ROC SD  Sens SD  Spec SD
#  0.15    0.737  0.531  0.8    0.0231  0.0372   0.0348 
#  0.16 *  0.738  0.531  0.798  0.023   0.0471   0.046 

# in-sample ROC: 0.7819365
pred.tr.log2 <- predict(log.net, newdata=train, type="prob")[,2]
roc.tr.log2 <- prediction(pred.tr.log2, train$Happy)
performance(roc.tr.log2, "auc")@y.values # **** 0.7819365 ****
plot(performance(roc.tr.log2, "sens", "fpr"))

# out-of-sample ROC: 0.7275352
pred.te.log2 <- predict(log.net, newdata=test, type="prob")[,2]
roc.te.log2 <- prediction(pred.te.log2, test$Happy)
performance(roc.te.log2, "auc")@y.values # **** 0.7275352 ****
plot(performance(roc.te.log2, "sens", "fpr"))

##### final model trained on soh with optimal parameters ####
final.control <- trainControl(method="cv", number=10, classProb=T, summaryFunction = twoClassSummary)
set.seed(86)
log.net.final <- train(Happy ~., data=soh[-1], method='glmnet', family="binomial",
  metric = "ROC",
  tuneGrid = expand.grid(.alpha=0, .lambda=0.16),
  trControl=final.control)

pred.soh.log2 <- predict(log.net.final, newdata=soh, type="prob")[,2]
roc.soh.log2 <- prediction(pred.soh.log2, soh$Happy)
auc.soh.log2 <- as.numeric(performance(roc.soh.log2, "auc")@y.values) # **** 0.7711326 ****

# a list to save all final models
final.models <- list(LogNet=list(model=log.net.final, cv=log.net.final$results["ROC"], in.sam=auc.soh.log2))

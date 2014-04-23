##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Model Selection 5: Random Forests
## 4/22/2014

library(randomForest)

# define tuning grid for mtry
rf.grid <- expand.grid(.mtry=c(20, 40, 50, 60, 80))
rf.control <- trainControl(method="cv", number=10, 
  classProb=T, summaryFunction = twoClassSummary, verboseIter=T)

set.seed(1986)
rf2 <- train(Happy ~., 
  data=train[-1], method="rf", 
  metric = "ROC",
  trControl=rf.control,
  tuneLength=5)
  #tuneGrid=rf.grid)

rf1 # cv ROC: 0.726
#  mtry  ROC    Sens   Spec   ROC SD  Sens SD  Spec SD
#  12    0.72   0.435  0.841  0.0344  0.0437   0.0261 
#  16    0.725  0.459  0.837  0.035   0.0453   0.0307 
#  20 *  0.726  0.462  0.827  0.0363  0.0517   0.0291 

set.seed(1986)
rf3 <- randomForest(Happy ~ ., data=train[-1], ntree=1000, mtry=30)

# in-sample ROC: 1 ??
pred.tr.rf1 <- predict(rf1, newdata=train, type="prob")[,2]
roc.tr.rf1 <- prediction(pred.tr.rf1, train$Happy)
performance(roc.tr.rf1, "auc")@y.values # ****  1 ?? ****
plot(performance(roc.tr.rf1, "sens", "fpr"))

pred.tr.rf3 <- predict(rf3, type="prob")[,2]
roc.tr.rf3 <- prediction(pred.tr.rf3, train$Happy)
performance(roc.tr.rf3, "auc")@y.values # ****  0.7373881 ****
plot(performance(roc.tr.rf3, "sens", "fpr"))

# out-of-sample ROC: 0.7176675 
pred.te.rf1 <- predict(rf1, newdata=test, type="prob")[,2]
roc.te.rf1 <- prediction(pred.te.rf1, test$Happy)
performance(roc.te.rf1, "auc")@y.values # ****  0.7176675 ****
plot(performance(roc.te.rf1, "sens", "fpr"))

pred.te.rf3 <- predict(rf3, newdata=test, type="prob")[,2]
roc.te.rf3 <- prediction(pred.te.rf3, test$Happy)
performance(roc.te.rf3, "auc")@y.values # ****  0.728315 ****
plot(performance(roc.te.rf3, "sens", "fpr"))

##### final model trained on soh with optimal parameters ####
final.control <- trainControl(method="cv", number=10, classProb=T, summaryFunction = twoClassSummary, verboseIter=T)
set.seed(86)
rf.final <- train(Happy ~., 
  data=soh[-1], method="rf", 
  metric = "ROC",
  trControl=final.control,
  tuneGrid=expand.grid(.mtry=20))

set.seed(86)
rf.final2 <- randomForest(Happy ~ ., data=soh[-1], ntree=1000, mtry=30)

pred.soh.rf <- predict(rf.final, newdata=soh[-1], type="prob")[,2]
roc.soh.rf <- prediction(pred.soh.rf, soh$Happy)
auc.soh.rf <- as.numeric(performance(roc.soh.rf, "auc")@y.values) # **** 1 ****

pred.soh.rf2 <- predict(rf.final2, type="prob")[,2]
roc.soh.rf2 <- prediction(pred.soh.rf2, soh$Happy)
auc.soh.rf2 <- as.numeric(performance(roc.soh.rf2, "auc")@y.values) # **** 0.7355268 ****


final.models <- c(final.models, list(RF=list(model=rf.final, cv=rf.final$results["ROC"], in.sam=auc.soh.rf)))
final.models <- c(final.models, list(RF2=list(model=rf.final2, cv=rf.final2$results["ROC"], in.sam=auc.soh.rf2)))

save(log1, log.net, tree1, tree2, rf1, rf3, tree.base, tree.base2, file="models.Rda")

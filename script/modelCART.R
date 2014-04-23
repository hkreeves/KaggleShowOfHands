##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Model Selection 4: CART
## 4/22/2014

# As a countinuation of Model Selection 3: Logistic
set.seed(139)
tree1 <- train(Happy ~ ., data=train[-1],
  method="rpart", 
  metric="ROC", 
  tuneGrid=expand.grid(.cp=seq(0.005, 0.02, by=0.001)),
  trControl=cv.control)
tree1 # cv: 0.653, cp=0.013
#  cp     ROC    Sens   Spec   ROC SD  Sens SD  Spec SD
#  0.009  0.634  0.454  0.788  0.0763  0.0612   0.0314 
#  0.011  0.643  0.459  0.784  0.0612  0.0568   0.0387 
#  0.013* 0.653  0.47   0.775  0.0337  0.0567   0.045 
prp(tree1$finalModel, extra=1, varlen=0, faclen=20) 

# in-sample ROC: 0.6582607
pred.tr.tree1 <- predict(tree1, newdata=train, type="prob")[,2]
roc.tr.tree1 <- prediction(pred.tr.tree1, train$Happy)
performance(roc.tr.tree1, "auc")@y.values # ****  0.6582607 ****
plot(performance(roc.tr.base, "sens", "fpr"))

# out-of-sample ROC: 0.660338 
pred.te.tree1 <- predict(tree1, newdata=test, type="prob")[,2]
roc.te.tree1 <- prediction(pred.te.tree1, test$Happy)
performance(roc.te.tree1, "auc")@y.values # ****  0.660338 ****
plot(performance(roc.te.tree1, "sens", "fpr"))

##### final model trained on soh with optimal parameters ####
tree1$bestTune

final.control <- trainControl(method="cv", number=10, classProb=T, summaryFunction = twoClassSummary)
set.seed(86)
tree.final <- train(Happy ~ ., data=soh[-1],
  method="rpart", 
  metric="ROC", 
  tuneGrid=expand.grid(.cp=0.013),
  trControl=final.control)

pred.soh.tree <- predict(tree.final, newdata=soh, type="prob")[,2]
roc.soh.tree <- prediction(pred.soh.tree, soh$Happy)
auc.soh.tree <- as.numeric(performance(roc.soh.tree, "auc")@y.values) # **** 0.6589737 ****

# save model
final.models <- c(final.models, list(CART=list(model=tree.final, cv=tree.final$results["ROC"], in.sam=auc.soh.tree)))

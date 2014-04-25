##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Logistic Regression 2: Variable Selection
## 4/23/2014

# perform logistic regression on the reducted variable sets
log.red1 <- glm(Happy ~ . + Gender * EducationLevel, data=train.ex[var.set], family="binomial")
summary(log.red1)

# in-sample ROC: 0.769986
pred2.tr.log1 <- predict(log.red1, type="response")
roc2.tr.log1 <- prediction(pred2.tr.log1, train.ex$Happy)
performance(roc2.tr.log1, "auc")@y.values # **** 0.769986****
plot(performance(roc2.tr.log1, "sens", "fpr"))

# out-of-sample ROC: 0.7268495
pred2.te.log1 <- predict(log.red1, newdata=test.ex, type="response")
roc2.te.log1 <- prediction(pred2.te.log1, test.ex$Happy)
performance(roc2.te.log1, "auc")@y.values # **** 0.7268495 ****
plot(performance(roc2.te.log1, "sens", "fpr"))

####
cv.control$verboseIter <- TRUE
set.seed(139)
log.net2 <- train(Happy ~. + Gender * EducationLevel, data=train.ex[var.set], method='glmnet', family="binomial",
  metric = "ROC",
  tuneGrid = expand.grid(.alpha=0, .lambda=seq(0.01,0.08,by=0.005)),
  trControl=cv.control)
plot(log.net2)
log.net2$bestTune # cv ROC: 0.7415
#  alpha lambda
#5     0   0.05

# in-sample ROC: 0.7665414
pred2.tr.log2 <- predict(log.net2, newdata=train.ex, type="prob")[,2]
roc2.tr.log2 <- prediction(pred2.tr.log2, train.ex$Happy)
performance(roc2.tr.log2, "auc")@y.values # **** 0.7665414 ****
plot(performance(roc2.tr.log2, "sens", "fpr"))

# out-of-sample ROC: 0.7299246
pred2.te.log2 <- predict(log.net2, newdata=test.ex, type="prob")[,2]
roc2.te.log2 <- prediction(pred2.te.log2, test.ex$Happy)
performance(roc2.te.log2, "auc")@y.values # **** 0.7299246 ****
plot(performance(roc2.te.log2, "sens", "fpr"))

##### final model trained on soh with optimal parameters ####
set.seed(85)
rf.var <- randomForest(Happy ~ ., data=soh.ex[-1], ntree=500, mtry=30)
varImpPlot(rf.var)

var.soh <- sort(as.vector(rf.var$importance), decreasing=T, index.return=T)
sort.var.soh <- data.frame(list(Var=rownames(rf.var$importance)[var.soh$ix], Imp=var.soh$x))
head(sort.var.soh, 30)
var.set.soh <- unique(c(names(soh.ex[2:11]), as.character(sort.var.soh$Var[1:40])))


final.control <- trainControl(method="cv", number=10, classProb=T, summaryFunction = twoClassSummary)
set.seed(86)
log.net.final2 <- train(Happy ~. + Gender * EducationLevel, data=soh.ex[var.set.soh], method='glmnet', family="binomial",
  metric = "ROC",
  tuneGrid = expand.grid(.alpha=0, .lambda=0.05),
  trControl=final.control)
log.net.final2$results
#  0.7415531

pred2.soh.log2 <- predict(log.net.final2, newdata=soh.ex, type="prob")[,2]
roc2.soh.log2 <- prediction(pred2.soh.log2, soh.ex$Happy)
auc2.soh.log2 <- as.numeric(performance(roc2.soh.log2, "auc")@y.values) # **** 0.7636715 ****

final.models2 <- list(LogNet=list(model=log.net.final2, cv=log.net.final2$results$ROC, in.sam=auc2.soh.log2))

# apply ridge logistic regression model on test set
pred2.log <- predict(final.models2$LogNet$model, newdata=soh.test.ex, type="prob")[,2]
pred2.comb <- as.data.frame(list(UserID=soh.test.ex$UserID, Prob.Log=pred2.log))

## ridge Logistic: OOS=0.74177 ... 0.73910
write.table(pred2.comb[1:2], file="./result/logNet2.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1"))

save(final.models2, sort.var.soh, var.set.soh, file="finalModels2.Rda")

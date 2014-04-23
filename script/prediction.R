##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Prediction
## 4/22/2014

# final models in comparison
final.models

# apply ridge logistic regression model on test set
pred.log <- predict(final.models$LogNet$model, newdata=soh.test, type="prob")[,2]
pred.comb <- as.data.frame(list(UserID=soh.test$UserID, Prob.Log=pred.log))

# apply CART on test set
pred.tree <- predict(final.models$CART$model, newdata=soh.test, type="prob")[,2]
pred.comb <- cbind(pred.comb, pred.tree)

# apply CART on test set
pred.rf <- predict(final.models$RF$model, newdata=soh.test, type="prob")[,2]
pred.comb <- cbind(pred.comb, pred.rf)

pred.rf2 <- predict(final.models$RF2$model, newdata=soh.test, type="prob")[,2]
pred.comb <- cbind(pred.comb, pred.rf2)

####### export ######
## baseline: CART OOS=0.63546
write.table(pred.comb[c(1,3)], file="./result/CART.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1"))

## ridge Logistic: OOS=0.73410
write.table(pred.comb[1:2], file="./result/logNet.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1"))

## random Forest (caret tuned): OOS=0.72443
write.table(pred.comb[c(1,4)], file="./result/RFcaret.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1"))

## random Forest (standalone): OOS=0.72921
write.table(pred.comb[c(1,5)], file="./result/RF.csv", sep=",", quote=F, 
  row.names=F, col.names=c("UserID", "Probability1"))

#####################
sapply(final.models, function(x) c(as.numeric(x$cv), x$in.sam))

# Consistent with the CV evaluation, logistic regression with ridge regularity is the winner.
# scoring 0.739 ROC in CV and 0.734 ROC out of sample.
# The further improvement may lie on ensembling, or cluster-then-predict approach.

save(final.models, file="finalModels.Rda")

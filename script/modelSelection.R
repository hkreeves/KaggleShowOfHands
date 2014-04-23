##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Model Selection 1
## 4/16/2014
## updated: 4/22/2014

# The model selection scheme is the following.
# 
# Preprocessing: keep all "unanwered" as a choice for vote response variables.
# Two ways to impute the demographic missing information.
#
# 1) Use caret's internal knn imputation. Pros: imputation is trained from training set
# and applied to test/cross-validation set. No cheating. Cons: knn may not work properly for
# categorical variables.
# 2) Use multiple imputation (or any other imputation methods) globally with train and test
# data. Pros: Properly handle numeric and categorical variables. Cons: data snooping concern 
#
#
# Training & evaluation: 
# 1. Split data into training set and test set balancedly with a ration of 0.7.
# 2. Cross validation: caret will be used in model training. A 10-fold, 10-repeats 
# crovss-validation # with identical partitions (fix the seed) to evaluate the following 
# models: 1) logistic regression, 2) CART tree, 3) Random Forests, and 4) maybe an ensemble
# of 1-3.  A non-imputed version of CART will also be compared. Evaluation metric will be AUC.
#
# Further thoughts: 1) ensemble method to combine various model predictions, unweighted or weighted
# by performance (AUC in this case). 2) collaborative filtering on the vote reponses. 
# 3) take cluster-then-predict approach by segmenting observations based on vote responses. But
# need a clustering method that deals with factor vairables. Euclidean distance is undefined for
# factors. One way may be transform categorical variables to dummy variables.

library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(ROCR)
library(pROC)

load("soh.Rda")
soh$Happy <- factor(soh$Happy, levels=c(0,1), labels=c("No", "Yes"))

# data splitting
set.seed(334)
splits <- sample.split(soh$Happy, 0.7)
#train.index <- createDataPartition(soh$Happy, p=0.7, list=F)

############
# baseline model: CART tree without imputation.
train <- subset(soh[-1], splits) # nrow = 3233
test <- subset(soh[-1], !splits) # nrow = 1386

# define train control parameters
cv.control <- trainControl(method="repeatedcv", number=10, repeats=3, classProb=T, summaryFunction = twoClassSummary)

# train a CART tree: **** cv-ROC = 0.646 ***
set.seed(139)
tree.base <- train(Happy ~ ., data=train, method="rpart", metric="ROC", na.action=na.pass, trControl=cv.control, tuneLength=5)
tree.base
#  cp      ROC    Sens   Spec   ROC SD  Sens SD  Spec SD
#  *0.00674  0.646  0.478  0.762  0.0495  0.0757   0.0469 
#  0.0078   0.635  0.466  0.774  0.0669  0.0716   0.0452 
#  *0.0156   0.646  0.496  0.747  0.037   0.0635   0.0588 
#  0.0227   0.539  0.498  0.724  0.111   0.0623   0.0387 
#  0.135    0.46   0.212  0.868  0.042   0.216    0.135 
prp(tree.base$finalModel, extra=1, varlen=0, faclen=20)

# another way to get optimized full tree on the training set
tree.base2 <- rpart(Happy ~., data=train, method="class", cp=0.00674)
prp(tree.base2, extra=1, varlen=0, faclen=20)

######
#  most informative questions are Q118237, Q101162, Q102289, Q107869 ..
######

# test performance
# most frequent option
prop.table(table(test$Happy))
        0         1 
0.4365079 0.5634921 

# in-sample ROC: 0.6614109
pred.tr.base <- predict(tree.base, newdata=train, na.action=na.pass, type="prob")[,2]
roc.tr.base <- prediction(pred.tr.base, train$Happy)
performance(roc.tr.base, "auc")@y.values # ****  0.6614109 ****
plot(performance(roc.tr.base, "sens", "fpr"))

pred.tr.base2 <- predict(tree.base2, newdata=train, type="prob")[,2]
roc.tr.base2 <- prediction(pred.tr.base2, train$Happy)
performance(roc.tr.base2, "auc")@y.values # ****  0.6790477 ****
plot(performance(roc.tr.base2, "sens", "fpr"))

# out-of-sample ROC: 0.6553867 
pred.te.base <- predict(tree.base, newdata=test, na.action=na.pass, type="prob")[,2]
roc.te.base <- prediction(pred.te.base, test$Happy)
performance(roc.te.base, "auc")@y.values # ****  0.6553867 ****
plot(performance(roc.te.base, "sens", "fpr"))

pred.te.base2 <- predict(tree.base2, newdata=test, type="prob")[,2]
roc.te.base2 <- prediction(pred.te.base2, test$Happy)
performance(roc.te.base2, "auc")@y.values # ****  0.6535074 ****
plot(performance(roc.te.base2, "sens", "fpr"))


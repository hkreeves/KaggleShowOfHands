##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Feature Selection
## 4/23/2014

library(Hmisc)

# multipanel barcharts
par(mfrow=c(2,1), mar=c(2.6,10,2,1))
draw.bars <- function(var1, var2, data=soh){
  cut <- split(data, data[[var1]])
  nl <- nlevels(data[[var1]])
  lvl <- levels(data[[var1]])
  par(mfrow=c(nl, 1))
  for( i in 1:nl){
    grping <- lapply(cut, function(x) prop.table(table(x[[var2]], x$Happy),1))
    barplot(t(grping[[i]]), beside=F, horiz=T, las=2)
    mtext(lvl[i], side=3)
  }
}

draw.bars("Gender", "Income") # not obvious in Gender
draw.bars("Gender", "EducationLevel") # very obvious gender*Edu difference
draw.bars("Gender", "HouseholdStatus") # no obvious interaction difference
draw.bars("Gender", "Party") # no obvious interaction difference
draw.bars("Income", "HouseholdStatus") # strong interaction patterns

###### extract Age/Agegroup, Martial, HasKid information ######
age <- 2014 - comb.imp2$YOB
age[(age > 100) | (age < 0)] <- median(age, na.rm=T)
agegroup <- cut2(age, g=4) 

mar <- rep(NA, nrow(comb.imp2))
status <- comb.imp2$HouseholdStatus
mar[grep("Single", status)] <- "Single"
mar[grep("Married", status)] <- "Married"
mar[grep("Partners", status)] <- "Partners"

kid <- rep(NA,nrow(comb.imp2))
kid[grep("w/kids", status)] <- "Yes"
kid[grep("no kids", status)] <- "No"

mar <- as.factor(mar)
kid <- as.factor(kid)

# combine into original data
comb.ex <- cbind(agegroup, mar, kid, comb.imp2)
names(comb.ex)[1:3] <- c("AgeGroup", "Martial", "HasKid")
comb.ex <- comb.ex[c(4,5,1,6:8,2,3,9:ncol(comb.ex))]
comb.ex$YOB <- NULL

##### new barplots investigating interaction #####
soh.ex <- subset(comb.ex, !is.na(Happy))
soh.test.ex <- subset(comb.ex, is.na(Happy))

draw.bars("HasKid", "Martial", soh.ex) # not obvious in Gender
draw.bars("AgeGroup", "EducationLevel", soh.ex) # very obvious gender*Edu difference
draw.bars("Gender", "AgeGroup", soh.ex) # no obvious interaction difference
draw.bars("AgeGroup", "Party", soh.ex) # no obvious interaction difference
draw.bars("Income", "HouseholdStatus") # strong interaction patterns

##### logistic regression to select demographic features ####
train.ex <- subset(soh.ex, splits)
test.ex <- subset(soh.ex, !splits)

log.demo <- glm(Happy ~ . + Gender * EducationLevel, data=train.ex[2:11], family="binomial")
summary(log.demo)

log.demo2 <- glm(Happy ~ . + Gender * EducationLevel - HouseholdStatus, 
  data=train.ex[2:11], family="binomial")
summary(log.demo2)

pred2.te.demo1 <- predict(log.demo, newdata=test.ex, type="response")
roc2.te.demo1 <- prediction(pred2.te.demo1, test.ex$Happy)
performance(roc2.te.demo1, "auc")@y.values 

pred2.te.demo2 <- predict(log.demo2, newdata=test.ex, type="response")
roc2.te.demo2 <- prediction(pred2.te.demo2, test.ex$Happy)
performance(roc2.te.demo2, "auc")@y.values 

############## variable selection #############
# 1. fit a random Forests model to select important variables

set.seed(85)
rf.var.sel <- randomForest(Happy ~ ., data=train.ex[-1], ntree=500, mtry=30)
rf.var.sel2 <- randomForest(Happy ~ ., data=train.ex[-1], ntree=500, mtry=20)
varImpPlot(rf.var.sel)
varImpPlot(rf.var.sel2)

var.imp <- sort(as.vector(rf.var.sel$importance), decreasing=T, index.return=T)
sort.var <- data.frame(list(Var=rownames(rf.var.sel$importance)[var.imp$ix], Imp=var.imp$x))
head(sort.var, 30)

# decide the variable set
var.set <- unique(c(names(train.ex[2:11]), as.character(sort.var$Var[1:30])))

save(comb.ex, soh.ex, soh.test.ex, train.ex, test.ex, sort.var, var.set, file="sohExtented.Rda")

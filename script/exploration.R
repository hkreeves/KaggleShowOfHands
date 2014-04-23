##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Data exploration
## 4/16/2014

library(caret)

# load Show Of Hands dataset
soh <- read.csv("./data/train.csv", stringsAsFactors=F)
soh.test <- read.csv("./data/test.csv", stringsAsFactors=F)
nrow(soh) # 4619
nrow(soh.test) # 1980

# add an NA column to test such that soh.test and soh match each other in variables
Happy <- NA
soh.test <- cbind(Happy, soh.test)[colnames(soh)]

# combine soh and soh.test for preprocessing
soh <- rbind(soh, soh.test)

# assign NA to missing values ("") in the demographic variables (2:7)
demographic <- soh[2:7]
demographic[demographic == ""] <- NA
soh[2:7] <- demographic
summary(soh)

# replace "" in all vote response variables as "Skipped"
vote.response <- soh[9:109]
vote.response[vote.response == ""] <- "Skipped"
soh[9:109] <- vote.response

# convert variables into factors
vars.convert <- c(3:7, 9:109)
soh[vars.convert] <- as.data.frame(lapply(soh[vars.convert], as.factor))
summary(soh)

# relevels and make as ordered factors (might improve imputation performance)
inc.lvl <- levels(soh$Income)
inc.lvl <- inc.lvl[c(6, 2, 3, 4, 1, 5)]
soh$Income <- ordered(soh$Income, levels=inc.lvl)
edu.lvl <- levels(soh$EducationLevel)
edu.lvl <- edu.lvl[c(3, 6, 4, 1, 2, 7, 5)]
soh$EducationLevel <- ordered(soh$EducationLevel, levels=edu.lvl)

# split up data back to soh and soh.test
soh.test <- soh[4620:nrow(soh),]
soh <- soh[1:4619,]

# another way of importing
#soh2 <- read.csv("./data/train.csv", na.strings = "")
#soh2$YOB <- as.integer(as.character(soh2$YOB))
#str(soh2)
#summary(soh2[1:8])

# to avoid the naming problem in caret, convert Happy to factor with "Yes/No" labels
soh$Happy <- factor(soh$Happy, levels=c(0,1), labels=c("No", "Yes"))

# save the preproccesed dataset
save(soh, soh.test, file="soh.Rda")

# the YOB has outliers, with minimum 1900 (2) and maximum 2039 (1)
prop.table(table(soh$Happy, soh$vote > 40), 2)
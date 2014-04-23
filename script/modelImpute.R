##
## Kaggle X Show Of Hands: Happiness Prediction
##
## Author: Kai He
##

## Model Selection 2: Imputation
## 4/22/2014

# part II: preprocess data (combining soh and test; might be questionable)
# use Multivariate Imputations by Chained Equations (mice)
# Detail methods as follows (from 'mice' help page). 
#
#         ??pmm?．, predictive mean matching (numeric data) ??logreg?．, logistic
#          regression imputation (binary data, factor with 2 levels)
#          ??polyreg?．, polytomous regression imputation for unordered
#          categorical data (factor >= 2 levels) ??polr?．, proportional
#          odds model for (ordered, >= 2 levels)

library(mice)
library(FactoMineR)

load("soh.Rda")

# combine soh and soh.test, leave out the id and outcome variable
comb <- rbind(soh, soh.test)

# extract the vote part (columns: 7:107)
vote <- comb[9:109]
demo <- comb[c(2:7,110)]

# perform a Multiple Correspondence Analysis (similar to PCA, but for factor variables)
# aim to reduce the dimension of vote, and include the components in mice imputation
fit <- MCA(vote, ncp=25) # top 25 components explains 50% of variance
comb.mca <- cbind(demo, fit$svd$U)

# impute
mice1 <- mice(demo, seed=116) # imputation with only demo data
mice2 <- mice(comb.mca, seed=119) # imputation with demo + MCA.components

demo.imp1 <- complete(mice1)
demo.imp2 <- complete(mice2)[1:7]

comb.imp1 <- cbind(comb[c(1,8)], demo.imp1, vote)
comb.imp2 <- cbind(comb[c(1,8)], demo.imp2, vote)

# divide the imputed data back to soh and soh.test
soh <- subset(comb.imp2, !is.na(Happy))
soh.test <- subset(comb.imp2, is.na(Happy))

# partition the new soh into train and test using the same "splits"
# set.seed(334)
train <- subset(soh, splits)
test <- subset(soh, !splits)

save(comb.imp1, comb.imp2, splits, soh, soh.test, train, test, file="sohImputed.Rda")
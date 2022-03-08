# Besm Allah
# Arwa Ashi
# March 6th, 2022
# Harvard PH125.9x
# Data Science: Capstone
# Final Project
# -------------------------

# ==================================
# Record Linkage Project
# https://cran.r-project.org/web/packages/RecordLinkage/vignettes/BigData.pdf
# ==================================
#------------------------------
# Calling packages 
#------------------------------
#install.packages('RecordLinkage')
#install.packages("randomForest")
#install.packages('reclin2')
library(RecordLinkage)
library(tidyverse)
library(dslabs)
library(diyar)
library(caret)
library(gridExtra)
library(randomForest)
library(reclin2)

showClass("RLBigData")
showClass("RLBigDataDedup")
showClass("RLBigDataLinkage")
showClass("RLResult")

#------------------------------
# Calling the data
#------------------------------
data(RLdata500)
data(RLdata10000)

#------------------------------
# Exploring the data
#------------------------------
head(RLdata500)
head(RLdata10000)

#------------------------------
# generate the feature - record pairs
#------------------------------
pairs_feature_500 <- compare.dedup(RLdata500,
                                   blockfld = list(1,5:7), # match in 5,6,7 columns
                                   strcmp = c(2,3,4),
                                   strcmpfun = levenshteinSim)
pairs_feature_500
str(pairs_feature_500) # return a list of all feature
matches_500 <- pairs_feature_500$pairs
matches_500

pairs_feature_10000 <- compare.dedup(RLdata10000,
                                   blockfld = list(1,5:7), # match in 5,6,7 columns
                                   strcmp = c(2,3,4),
                                   strcmpfun = levenshteinSim
                                   )
pairs_feature_10000
str(pairs_feature_1000) # return a list of all feature

#------------------------------
# Probabilistic Method
# Fellegi-Sunter Model
#------------------------------
# Define a cut off for string comparing at 80% by using EM algorithm

# calculating M and U weights using EM algorithm
cut_off_500 <- emWeights(pairs_feature_500, cutoff = 0.8)
summary(cut_off_500) 

# intial matches
allPairs_500 <- getPairs(cut_off_500)
head(allPairs_500)

# threshold is 30
finalPairs_500 <- getPairs(cut_off_500, max.weight = 30, min.weight = 0)
head(finalPairs_500)

# calculating M and U weights using EM algorithm
cut_off_10000 <- emWeights(pairs_feature_10000, cutoff = 0.8)
summary(cut_off_10000)

# initial matches 40
allPairs_10000 <- getPairs(cut_off_10000)
head(allPairs_10000)

finalPairs_10000 <- getPairs(cut_off_10000, max.weight = 40, min.weight = 0)
head(finalPairs_10000)

#------------------------------
# machine learning model()
#------------------------------
dataset1 <- RLdata500
dataset1['id'] <- identity.RLdata500

dataset2 <- RLdata10000
dataset2['id'] <- identity.RLdata10000

# 1 Creating a paris by using blocking fields
ML_pairs <- pair_blocking(dataset1,dataset1,c("id"))

# 2 Comparing the pairs and get comparing score for each feature
compare_pairs(ML_pairs, on = c("fname_c1","lname_c1","by","bm", "bd"), inplace = TRUE,
              comparators = list(fname_c1 = jaro_winkler(), 
                                 lname_c1 = jaro_winkler(), 
                                 by = jaro_winkler(),
                                 bm = jaro_winkler(),
                                 bd = jaro_winkler()))

# 3 preparing the binary parameters 'TRUE' or 'FALS'
dataset1['known_id'] <- dataset1['id']
setDT(dataset1)

compare_vars(ML_pairs, "y", on_x = "id", on_y = "known_id", 
             y = dataset1, inplace = TRUE)

compare_vars(ML_pairs, "y_true", on_x = "id", 
             on_y = "id", inplace = TRUE)

# 4 ML model - the logistic regression
glm_fit <- glm(y ~ fname_c1 + lname_c1 + by + bm + bd, 
               data = ML_pairs,family = binomial())

# 5 predicting the matching probability
ML_pairs[, `:=`(prob, predict(glm_fit, type = "response", newdata = ML_pairs))]

# 6 selecting a matching probability that is greater than 50%
ML_pairs[, `:=`(select, prob > 0.5)]

# 7 generating a FALSE TRUE table for the result evaluation
table(ML_pairs$select > 0.5, ML_pairs$y_true)

# 8 generate the Final matching pair
ML_FinalPairs <- link(ML_pairs, selection = "select", all_y = TRUE)
ML_FinalPairs
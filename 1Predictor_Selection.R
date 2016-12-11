
### 4. Variable Selection and Elimination

#############################
# References
# 1. Variable selection using rfe, recursive feature selection
# https://github.com/MichaelPluemacher/Big-Mart-Sales/blob/master/AnalyzeAndClean.R
# 2. Boruta vs RFE (Good read)
# https://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/
# 3. Excellent article on cleanup, preprocessing
# https://www.mql5.com/en/articles/2029
#############################

#Read files:
#setwd("Downloads/Kaggle/Bigmart_Sales_3/")
trainm = read.csv("trainmodified.csv",stringsAsFactors = TRUE)
testm = read.csv("testmodified.csv",stringsAsFactors = TRUE)


# check variable importance with 
# random feature elimination (RFE)
# from caret 

# scale Sales to be in interval [0,1]
maxSales <- max(trainm$Item_Outlet_Sales)
trainm$Item_Outlet_Sales <- trainm$Item_Outlet_Sales/maxSales
names(trainm)[names(trainm)=='Item_Outlet_Sales'] <- "target"
# ========================================================================
# ===============Reusable Code starts here================================
# ========================================================================
set.seed(0)

# one-hot encoding of the factor variables
# leave out the intercept column
trainm$Item_Identifier <- NULL
testm$Item_Identifier <- NULL
trainm$Outlet_Identifier <- NULL
testm$Outlet_Identifier <- NULL

trainm <- as.data.frame(model.matrix( ~ . + 0 , data = trainm))
testm <- as.data.frame(model.matrix( ~ . + 0, data = testm))

#str(trainm)
#########################################
#REmove Near Zero Variance Variables 
#
#########################################
nzv_cols <- nearZeroVar(trainm)
nzv_colnames <- names(trainm[nzv_cols])
if(length(nzv_cols) > 0) {
  trainm <- trainm[!names(testm) %in% nzv_colnames]
  testm <- testm[!names(testm) %in% nzv_colnames]
}

#########################################
#Identification and removal of correlated predictors (numeric).
# x1 = b1x2
#########################################
corcol <- findCorrelation(cor(trainm), cutoff = .90, verbose = FALSE)
nzv_colnames <- names(trainm[corcol])

if(length(corcol) > 0) {
  trainm <- trainm[!names(testm) %in% nzv_colnames]
  testm <- testm[!names(testm) %in% nzv_colnames]
}
#########################################
#Identification and removal of linear dependencies (factors).
# x3 = b1x1 + b2x2
#########################################
comboInfo <- findLinearCombos(trainm)


# define a vector of Item_Outlet_Sales
# and a dataframe of predictors
#target <- trainm$Item_Outlet_Sales
#predictors <- subset(trainm, select=-c(Item_Outlet_Sales))

#########################################
#
# check relative importance of predictors
# with caret rfe
# A simple backwards selection, a.k.a. recursive feature selection (RFE), algorithm
#########################################

# do it in parallel
library(doParallel)
cl <- makeCluster(detectCores()); registerDoParallel(cl)

subsetSizes <- c(1:20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 121)
N <- 5 # number of resamples
seeds <- vector(mode = "list", length = N+1)
for(i in 1:N) seeds[[i]] <- sample.int(1000, length(subsetSizes) + 1)
seeds[[N+1]] <- sample.int(1000, 1)

library(caret)
control <- rfeControl(functions=rfFuncs,
                      method="cv",
                      seeds = seeds,
                      number = N,
                      repeats = 3,
                      verbose=TRUE,
                      allowParallel=TRUE
)
# Start the clock!
ptm <- proc.time()
# run the RFE algorithm
results2 <- rfe(x = predictors,
                y = target,
                sizes = subsetSizes,
                preProc=c("center", "scale"),
                rfeControl=control)
# Stop the clock
proc.time() - ptm

# stop the parallel processing and register sequential front-end
stopCluster(cl); registerDoSEQ();


# summarize the results
print(results2)
# list all features in descending order of importance
listOfPreds <- pickVars(results2$variables, 120)
listOfPreds
# plot the results
plot(results2, type=c("g", "o") )

# build a data frame containing the predictors 
# ordered by their importance

ordered.preds <- predictors[,listOfPreds[1]]
for (i in 2:length(listOfPreds)) {
  ordered.preds <- cbind(ordered.preds, predictors[,listOfPreds[i]])
}
colnames(ordered.preds) <- listOfPreds
ordered.preds <- as.data.frame(ordered.preds)

ordered.test <- testm[,listOfPreds[1]]
for (i in 2:length(listOfPreds)) {
  ordered.test <- cbind(ordered.test, testm[,listOfPreds[i]])
}
colnames(ordered.test) <- listOfPreds
ordered.test <- as.data.frame(ordered.test)

#remove the scaling to [0,1] in target

trainm$Item_Outlet_Sales <- trainm$Item_Outlet_Sales*maxSales


###################################
# Feature Selection Using Boruta
# 
###################################
library(Boruta)
  
set.seed(123)
boruta.train <- Boruta(target~., data = trainm, doTrace = 2)
print(boruta.train)

plot(boruta.train, xlab = "", xaxt = "n")

lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i) 
    boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))

axis(side = 1,las=2,labels = names(Labels),
       at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)


final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)
boruta.predictors <- getSelectedAttributes(final.boruta, withTentative = F)

boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)
######################################################################
boruta.predictors[boruta.predictors=='`Outlet_Location_TypeTier 2`'] <-  "Outlet_Location_TypeTier 2"
boruta.predictors[boruta.predictors=='`Outlet_Location_TypeTier 3`'] <-  "Outlet_Location_TypeTier 3"
boruta.predictors[boruta.predictors=='`Outlet_TypeSupermarket Type1`'] <-  "Outlet_TypeSupermarket Type1"
boruta.predictors[boruta.predictors=='`Outlet_TypeSupermarket Type2`'] <-  "Outlet_TypeSupermarket Type2"
boruta.predictors[boruta.predictors=='`Outlet_TypeSupermarket Type3`'] <-  "Outlet_TypeSupermarket Type3"

ordered.preds <- trainm[boruta.predictors]
ordered.test <- testm[boruta.predictors]
# save those dataframes to disk

write.csv(ordered.preds, file="ordered_predictors.csv", row.names=FALSE, quote = FALSE)
write.csv(ordered.test, file="ordered_test.csv", row.names=FALSE, quote = FALSE)
write.csv(trainm$target, file="sales.csv", row.names=FALSE, quote = FALSE)



# free up some memory
gc(verbose = TRUE)
ls(all = TRUE)
rm(list = ls(all = TRUE)) 
ls(all = TRUE)
gc(verbose = TRUE)


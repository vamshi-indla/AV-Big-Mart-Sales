
### 4. Variable Selection and Elimination

# check variable importance with 
# random feature elimination (RFE)
# from caret

# scale Sales to be in interval [0,1]
maxSales <- max(trainm$Item_Outlet_Sales)
trainm$Item_Outlet_Sales <- trainm$Item_Outlet_Sales/maxSales

set.seed(0)

# one-hot encoding of the factor variables
# leave out the intercept column

trainm <- as.data.frame(model.matrix( ~ . + 0, data = trainm))
testm <- as.data.frame(model.matrix( ~ . + 0, data = testm))

str(trainm)

# define a vector of Item_Outlet_Sales
# and a dataframe of predictors
target <- trainm$Item_Outlet_Sales
predictors <- subset(trainm, select=-c(Item_Outlet_Sales))


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

ordered.test <- new_test[,listOfPreds[1]]
for (i in 2:length(listOfPreds)) {
  ordered.test <- cbind(ordered.test, new_test[,listOfPreds[i]])
}
colnames(ordered.test) <- listOfPreds
ordered.test <- as.data.frame(ordered.test)

#remove the scaling to [0,1] in target

target <- target*maxSales


### 4. Model Building



### 5. Submission
################################
#
# preparing the final submissions
#
################################

final12 <- data.frame(test$Item_Identifier, test$Outlet_Identifier, prediction12)

names(final12) <- c("Item_Identifier",
                    "Outlet_Identifier",
                    "Item_Outlet_Sales")

write.csv(final12, file="final12.csv", row.names=FALSE, quote = FALSE)

# free up some memory
gc(verbose = TRUE)
ls(all = TRUE)
rm(list = ls(all = TRUE)) 
ls(all = TRUE)
gc(verbose = TRUE)


#Read files:
setwd("Downloads/Kaggle/Bigmart_Sales_3/")
train = read.csv("train.csv",stringsAsFactors = TRUE)
test = read.csv("test.csv",stringsAsFactors = TRUE)

# Merge train and test
train$source <-  'train'
test$source <- 'test' 
test$Item_Outlet_Sales <- 0

data <- rbind(train,test)

cat("combined data:", dim(data),"train:", dim(train),"test:",dim(test))

# 1. Data Exploration
# 2. Data Imputation
#Determine the average weight per item:
library(sqldf)
item_avg_weight = sqldf("select Item_Identifier,AVG(Item_Weight) as 'Item_Weight' from data Group by data.Item_Identifier")

#Get a boolean variable specifying missing Item_Weight values
miss_bool = is.na(data['Item_Weight'])

#Impute data and check #missing values before and after imputation to confirm
cat('Orignal #missing: ', sum(miss_bool))
data[miss_bool,'Item_Weight'] = item_avg_weight[data[miss_bool,'Item_Identifier'],'Item_Weight']
cat('Final #missing: ', sum(is.na(data['Item_Weight'])))


#Determing the mode for each Outlet_Type
item_mode_size  = sqldf("select Outlet_Type, Outlet_Size, count(*) as count from data group by Outlet_Type,Outlet_Size order by count(*) desc")

item_mode_size <- item_mode_size[!duplicated(item_mode_size$Outlet_Type),]

cat('Mode for each Outlet_Type') 
item_mode_size

#Get a boolean variable specifying missing Item_Weight values
miss_bool =  (data$Outlet_Size=='') 

#Impute data and check #missing values before and after imputation to confirm
cat('Orignal #missing: ', sum(miss_bool))
data[miss_bool,'Outlet_Size'] = item_mode_size[data[miss_bool,'Outlet_Type'],'Outlet_Size']
data$Outlet_Size  <- droplevels(data$Outlet_Size) 
cat('Final #missing: ', sum(data$Outlet_Size==''))


## 4. Feature Engineering
#4.1 Item Visibility

#Determine average visibility of a product
visibility_avg = aggregate(data$Item_Visibility, list(data$Item_Identifier),mean)
names(visibility_avg) <- c("Item_Identifier","Item_visibility_avg")

#Impute 0 values with mean visibility of that product:
miss_bool = (data['Item_Visibility'] == 0)

cat('Number of 0 values initially:',sum(miss_bool))
data[miss_bool,'Item_Visibility'] = visibility_avg[data[miss_bool,'Item_Identifier'],'Item_visibility_avg']
cat('Number of 0 values after modification:',sum(data['Item_Visibility'] == 0))

#Determine another variable with means ratio
data$Item_Visibility_MeanRatio = data$Item_Visibility/visibility_avg[data$Item_Identifier,'Item_visibility_avg'] 
summary(data$Item_Visibility_MeanRatio)

#4.2
#Get the first two characters of ID:
data$Item_Type_Combined = substr(data$Item_Identifier,1,2)

#Rename them to more intuitive categories:
library(plyr)
data$Item_Type_Combined = revalue(data$Item_Type_Combined,c("FD"="Food","NC"="Non-Consumable","DR"="Drinks"))
count(data$Item_Type_Combined)

#4.3
#Years:
data$Outlet_Years = 2013 - data$Outlet_Establishment_Year
summary(data$Outlet_Years)


#4.4
#Change categories of low fat:
cat('Original Categories:')
count(data$Item_Fat_Content)

cat('\nModified Categories:')
data$Item_Fat_Content = revalue(data$Item_Fat_Content,c('LF'='Low Fat',
                                                        'reg'='Regular',
                                                        'low fat'='Low Fat'))
count(data$Item_Fat_Content)

#Mark non-consumables as separate category in low_fat:
levels(data$Item_Fat_Content) <- c(levels(data$Item_Fat_Content),"Non-Edible")
data$Item_Fat_Content[data$Item_Type_Combined=="Non-Consumable"] <- "Non-Edible"

count(data$Item_Fat_Content)

#4.5
data$Saleno <- round(data$Item_Outlet_Sales/data$Item_MRP)

#4.5
#Drop the columns which have been converted to different types:
data$Item_Type <- NULL
data$Outlet_Establishment_Year <- NULL

#(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
trainm = data[data$source=="train",]
testm = data[data$source=="test",]

#Drop unnecessary columns:
trainm$source <- NULL

testm$Item_Outlet_Sales <- NULL
testm$source <- NULL


  
#=================
#5. Variable Selection 
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
nzv_colnames
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
nzv_colnames
if(length(corcol) > 0) {
  trainm <- trainm[!names(testm) %in% nzv_colnames]
  testm <- testm[!names(testm) %in% nzv_colnames]
}
#########################################
#Identification and removal of linear dependencies (factors).
# x3 = b1x1 + b2x2
#########################################
comboInfo <- findLinearCombos(trainm)
comboInfo

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

subsetSizes <- c(20, 25)

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
predictors <- subset(trainm, select=-c(Item_Outlet_Sales))
Saleno <- trainm$Saleno
results2 <- rfe(x = predictors,
                y = Saleno,
                sizes = subsetSizes,
                preProc=c("center", "scale"),
                rfeControl=control)
# Stop the clock
proc.time() - ptm

# stop the parallel processing and register sequential front-end
stopCluster(cl); registerDoSEQ()


# summarize the results
print(results2) 

# rfe tells these 4 variables are important
feature.names <- c('Item_Visibility_MeanRatio', 'Outlet_TypeSupermarket Type1', 'Outlet_Years', 'Outlet_SizeSmall')
#=========== 

#Boruta Feature selection
feature.names <- boruta.predictors

#=========== model 
# XGboost
library(caret)
MyTrainControl=trainControl(
  method = "cv",
  number=5,
  savePredictions = TRUE
)

library(xgboost)
xgb_model <- train(
  x = data.matrix(trainm[,feature.names]),
  y = trainm$Saleno,
  trControl = MyTrainControl,
  objective   = "reg:linear",
  eval_metric = "rmse"
  )

xgb_result1 <- predict(xgb_model,data.matrix(trainm[,feature.names]))
rmse(round(xgb_result1)*train$Item_MRP - train$Item_Outlet_Sales)
                      
# Predict the model
xgb_result <- predict(xgb_model,data.matrix(testm[,feature.names]))
xgb_result <- round(xgb_result)*testm$Item_MRP

# Model Submission
submission <- data.frame(Item_Identifier=test$Item_Identifier,Outlet_Identifier=test$Outlet_Identifier,
                         Item_Outlet_Sales=xgb_result)
write.csv(submission,"3.xgb_result.csv",row.names=F)

#==================== Ensemble
library(h2o); library(h2oEnsemble)
predictors <- trainm[,feature.names]
ordered_test <- testm[,feature.names]
# rescale sales to interval [0,1]
maxSales <- max(trainm$Saleno)
sales <- as.vector(trainm$Saleno/maxSales)

set.seed(42)

# do it in parallel
cl <- makeCluster(detectCores()); registerDoParallel(cl)


###############################################################
#
# H2O
#
###############################################################

## Start a local cluster with 8GB RAM
localH2O = h2o.init(ip = "localhost",
                    port = 54321,
                    startH2O = TRUE,
                    nthreads = -1,     # use all CPUs
                    max_mem_size = '8g') # maximum memory

# import dataframe into h2o
train.hex <- as.h2o(cbind(predictors,sales), destination_frame="train.hex")
test.hex <- as.h2o(ordered_test, destination_frame="test.hex")

######################################################
#
# h2o ensemble using the 12 most important predictors
#
######################################################

# # number of predictors we use to buld the model:
predictorCount <- length(feature.names)

# glm base learners
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha, solver = "IRLSM")
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha, solver = "IRLSM")
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha, solver = "IRLSM")
h2o.glm.4 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha, solver = "L_BFGS")
h2o.glm.5 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha, solver = "L_BFGS")
h2o.glm.6 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha, solver = "L_BFGS")

# random forest base learners
h2o.randomForest.1 <- function(...,
                               ntrees = 500,
                               mtries = predictorCount,
                               sample_rate = 0.8,
                               col_sample_rate = 1,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.2 <- function(...,
                               ntrees = 500,
                               mtries = -1,
                               sample_rate = 0.8,
                               col_sample_rate = 1,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.3 <- function(...,
                               ntrees = 500,
                               mtries = predictorCount,
                               sample_rate = 0.8,
                               col_sample_rate = 0.5,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.4 <- function(...,
                               ntrees = 500,
                               mtries = -1,
                               sample_rate = 0.8,
                               col_sample_rate = 0.5,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.5 <- function(...,
                               ntrees = 500,
                               mtries = 12,
                               sample_rate = 0.8,
                               col_sample_rate = 1,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.6 <- function(...,
                               ntrees = 500,
                               mtries = 12,
                               sample_rate = 0.8,
                               col_sample_rate = 0.5,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)

# gbm base learners
h2o.gbm.1 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  seed = seed)
h2o.gbm.2 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      nbins = 50,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  nbins = nbins,
                  seed = seed)
h2o.gbm.3 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      max_depth = 5,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.4 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.8, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  seed = seed)
h2o.gbm.5 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.7, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  seed = seed)
h2o.gbm.6 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.6, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  seed = seed)
h2o.gbm.7 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      balance_classes = TRUE,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  balance_classes = balance_classes,
                  seed = seed)
h2o.gbm.8 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      max_depth = 3,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.9 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      max_depth = 5,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.10 <- function(...,
                       ntrees = 100,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 2, # Minimum number of rows to assign to teminal nodes
                       max_depth = 5,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.11 <- function(...,
                       ntrees = 100,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 2, # Minimum number of rows to assign to teminal nodes
                       max_depth = 3,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.12 <- function(...,
                       ntrees = 150,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 2, # Minimum number of rows to assign to teminal nodes
                       max_depth = 3,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.13 <- function(...,
                       ntrees = 80,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 2, # Minimum number of rows to assign to teminal nodes
                       max_depth = 3,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.14 <- function(...,
                       ntrees = 80,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 1, # Minimum number of rows to assign to teminal nodes
                       max_depth = 5,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)

# deep learning neural net base learners
h2o.deeplearning.1 <- function(...,
                               hidden = c(50,50),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.2 <- function(...,
                               hidden = c(30,30,30),
                               activation = "Tanh",
                               epochs = 500,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.3 <- function(...,
                               hidden = c(13,13,13,13),
                               activation = "Rectifier",
                               epochs = 500,
                               max_w2 = 50,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.4 <- function(...,
                               hidden = c(30,30,30),
                               activation = "Rectifier",
                               epochs = 500,
                               max_w2 = 50,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.5 <- function(...,
                               hidden = c(30,30,30),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               max_w2 = 10,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.6 <- function(...,
                               hidden = c(13,13,13,13),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               max_w2 = 10,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.7 <- function(...,
                               hidden = c(13,13,13,13),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               max_w2 = 10,
                               rate = 0.5,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.8 <- function(...,
                               hidden = c(14,14,14,14),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               max_w2 = 10,
                               rate = 0.5,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.9 <- function(...,
                               hidden = c(20,20),
                               activation = "Tanh",
                               epochs = 500,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.10 <- function(...,
                                hidden = c(20,20),
                                activation = "Rectifier",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.11 <- function(...,
                                hidden = c(13,13,13,13),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.12 <- function(...,
                                hidden = c(10,10,10,10),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.13 <- function(...,
                                hidden = c(8,8,8,8),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.14 <- function(...,
                                hidden = c(8,8,8,8),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.2, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.15 <- function(...,
                                hidden = c(15,15,15,15),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.16 <- function(...,
                                hidden = c(15,15,15,15),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.17 <- function(...,
                                hidden = c(10,10,10,10),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.18 <- function(...,
                                hidden = c(30,30,30),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.19 <- function(...,
                                hidden = c(50,50,50),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.20 <- function(...,
                                hidden = c(100,100),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.21 <- function(...,
                                hidden = c(50,50),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.22 <- function(...,
                                hidden = c(10,10,10,10,10),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)

# which of the base learners defined above do we want to include 
# in the ensemble?
# Play around with various combinations, you'll see some rather
# important differences

learner <- c("h2o.randomForest.2", 
  "h2o.randomForest.3", 
  "h2o.randomForest.4", 
  "h2o.randomForest.6",
  "h2o.gbm.8",
  "h2o.gbm.11",
  "h2o.gbm.12",
  "h2o.gbm.13",
  "h2o.deeplearning.2",
  "h2o.deeplearning.3",
  "h2o.deeplearning.4",
  "h2o.deeplearning.9",
  "h2o.deeplearning.11",
  "h2o.deeplearning.12",
  "h2o.deeplearning.15",
  "h2o.deeplearning.16",
  "h2o.deeplearning.17",
  "h2o.deeplearning.18",
  "h2o.deeplearning.19",
  "h2o.deeplearning.20",
  "h2o.deeplearning.21"
)

# define the metalearner
#metalearner <- "h2o.glm.wrapper"
metalearner <- "h2o.gbm.wrapper"
#metalearner <- "h2o.deeplearning.wrapper"
#metalearner <- "h2o.randomForest.wrapper"

# train the ensemble
fit <- h2o.ensemble(x = 1:predictorCount,  # column numbers for predictors
                    y = ncol(predictors)+1,   # column number for label
                    training_frame = train.hex, # data in H2O format
                    family = "AUTO", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 3)) # 3-fold cross validation

# generate predictions on the test set

pred_train <- predict(fit, train.hex)
pred <- pred_train$pred*maxSales
rmse(round(pred$predict)*trainm$Item_MRP - trainm$Item_Outlet_Sales)

pred <- predict(fit, test.hex)
prediction12 <- as.data.frame(pred$pred*maxSales)

min(prediction12)
max(prediction12)

# predicted values smaller than zero are set to zero
prediction12[prediction12<0] <- 0


#=================

# free up some memory
gc(verbose = TRUE)
ls(all = TRUE)
rm(list = ls(all = TRUE)) 
ls(all = TRUE)
gc(verbose = TRUE)

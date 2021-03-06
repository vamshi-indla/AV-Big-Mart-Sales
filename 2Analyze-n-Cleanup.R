---
  title: "Approach and Solution to break in Top 20 of Big Mart Sales prediction in R"
output: github_document
Author: Vamshi Indla
--
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## References

Below analysis and model ideas are heavily borrowed from:
  https://www.analyticsvidhya.com/blog/2016/02/bigmart-sales-solution-top-20/
  
  Onehot Encoding
https://github.com/dmlc/xgboost/blob/master/R-package/demo/create_sparse_matrix.R

R-code with h2o Models:
  https://github.com/MichaelPluemacher/Big-Mart-Sales/blob/master/AnalyzeAndClean.R


## Introduction

BigMart Sales Prediction practice problem was launched on Analytics Vidhya in 2016. I am going to take you through the entire journey of getting started with this data set, using R.

We will explore the problem in following stages:
  
  1. Hypothesis Generation – understanding the problem better by brainstorming possible factors that can impact the outcome
2. Data Exploration – looking at categorical and continuous feature summaries and making inferences about the data.
3. Data Cleaning – imputing missing values in the data and checking for outliers
4. Feature Engineering – modifying existing variables and creating new ones for analysis
5. Model Building – making predictive models on the data

## 1. Hypothesis Generation

This is a very pivotal step in the process of analyzing data. This involves understanding the problem and making some hypothesis about what could potentially have a good impact on the outcome. This is done BEFORE looking at the data, and we end up creating a laundry list of the different analysis which we can potentially perform if data is available. Read more about hypothesis generation here.

## The Problem Statement

Understanding the problem statement is the first and foremost step. You can view this in the competition page but I’ll iterate the same here:
  
  So the idea is to find out the properties of a product, and store which impacts the sales of a product. Let’s think about some of the analysis that can be done and come up with certain hypothesis.

## The Hypotheses

I came up with the following hypothesis while thinking about the problem. These are just my thoughts and you can come-up with many more of these. Since we’re talking about stores and products, lets make different sets for each.

### Store Level Hypotheses:

1. City type: Stores located in urban or Tier 1 cities should have higher sales because of the higher income levels of people there.
2. Population Density: Stores located in densely populated areas should have higher sales because of more demand.
3. Store Capacity: Stores which are very big in size should have higher sales as they act like one-stop-shops and people would prefer getting everything from one place
4. Competitors: Stores having similar establishments nearby should have less sales because of more competition.
Marketing: Stores which have a good marketing division should have higher sales as it will be able to attract customers through the right offers and advertising.
5. Location: Stores located within popular marketplaces should have higher sales because of better access to customers.
6. Customer Behavior: Stores keeping the right set of products to meet the local needs of customers will have higher sales.
7. Ambiance: Stores which are well-maintained and managed by polite and humble people are expected to have higher footfall and thus higher sales.

### Product Level Hypotheses:

1. Brand: Branded products should have higher sales because of higher trust in the customer.
2. Packaging: Products with good packaging can attract customers and sell more.
3. Utility: Daily use products should have a higher tendency to sell as compared to the specific use products.
4. Display Area: Products which are given bigger shelves in the store are likely to catch attention first and sell more.
5. Visibility in Store: The location of product in a store will impact sales. Ones which are right at entrance will catch the eye of customer first rather than the ones in back.
6. Advertising: Better advertising of products in the store will should higher sales in most cases.
7. Promotional Offers: Products accompanied with attractive offers and discounts will sell more.
These are just some basic 15 hypothesis I have made, but you can think further and create some of your own. Remember that the data might not be sufficient to test all of these, but forming these gives us a better understanding of the problem and we can even look for open source information if available.

Lets move on to the data exploration where we will have a look at the data in detail.

### 2. Data Exploration

We’ll be performing some basic data exploration here and come up with some inferences about the data. We’ll try to figure out some irregularities and address them in the next section. 

The first step is to look at the data and try to identify the information which we hypothesized vs the available data. A comparison between the data dictionary on the competition page and out hypotheses is shown below:
  
  ### Insert Image
  
  You will invariable find features which you hypothesized, but data doesn’t carry and vice versa. You should look for open source data to fill the gaps if possible. Let’s start by loading the required libraries and data. You can download the data from the competition page.

```{r,echo=TRUE}
#Read files:
setwd("Downloads/Kaggle/Bigmart_Sales_3/")
train = read.csv("train.csv",stringsAsFactors = TRUE)
test = read.csv("test.csv",stringsAsFactors = TRUE)

```

Its generally a good idea to combine both train and test data sets into one, perform feature engineering and then divide them later again. This saves the trouble of performing the same steps twice on test and train. Lets combine them into a dataframe ‘data’ with a ‘source’ column specifying where each observation belongs.

```{r,echo=TRUE}

train$source <-  'train'
test$source <- 'test' 
test$Item_Outlet_Sales <- NA

data <- rbind(train,test)

cat("combined data:", dim(data),"train:", dim(train),"test:",dim(test))
```


Thus we can see that data has same #columns but rows equivalent to both test and train. One of the key challenges in any data set is missing values. Lets start by checking which columns contain missing values.

```{r,echo=TRUE}
unlist(lapply (data, function (x) ifelse(class(x)=="numeric",sum(is.na(x)),sum(x==""))))
```

Note that the Item_Outlet_Sales is the target variable and missing values are ones in the test set. So we need not worry about it. But we’ll impute the missing values in Item_Weight and Outlet_Size in the data cleaning section.

Lets look at some basic statistics for numerical variables.

```{r,echo=TRUE}
summary(data)
}
```

Some observations:
  
  1. Item_Visibility has a min value of zero. This makes no practical sense because when a product is being sold in a store, the visibility cannot be 0.
2. Outlet_Establishment_Years vary from 1985 to 2009. The values might not be apt in this form. Rather, if we can convert them to how old the particular store is, it should have a better impact on sales.
3. The lower ‘count’ of Item_Weight and Item_Outlet_Sales confirms the findings from the missing value check.

Moving to nominal (categorical) variable, lets have a look at the number of unique values in each of them.

```{r,echo=TRUE}
unlist(lapply (data, function (x) length(unique(x))))
```

This tells us that there are 1559 products and 10 outlets/stores (which was also mentioned in problem statement). Another thing that should catch attention is that Item_Type has 16 unique values. Let’s explore further using the frequency of different categories in each nominal variable. I’ll exclude the ID and source variables for obvious reasons.

```{r,echo=TRUE}

#Filter categorical variables
categorical_columns <- names(data)[which(sapply(data, is.factor))]

#Exclude ID cols and source:
categorical_columns <- categorical_columns[-which(categorical_columns %in% c('Item_Identifier','Outlet_Identifier'))]

##Print frequency of categories
library(plyr)
for(col in which(names(data) %in% categorical_columns)){
  cat('\nFrequency of Categories for varible')
  print(count(data[col]))
}
```

The output gives us following observations:
  
  1. Item_Fat_Content: Some of ‘Low Fat’ values mis-coded as ‘low fat’ and ‘LF’. Also, some of ‘Regular’ are mentioned as ‘regular’.
2. Item_Type: Not all categories have substantial numbers. It looks like combining them can give better results.
3. Outlet_Type: Supermarket Type2 and Type3 can be combined. But we should check if that’s a good idea before doing it.

## 3. Data Cleaning

This step typically involves imputing missing values and treating outliers. Though outlier removal is very important in regression techniques, advanced tree based algorithms are impervious to outliers. So I’ll leave it to you to try it out. We’ll focus on the imputation step here, which is a very important step.

## Imputing Missing Values

We found two variables with missing values – Item_Weight and Outlet_Size. Lets impute the former by the average weight of the particular item. This can be done as:
  
  ```{r}
#Determine the average weight per item:
library(sqldf)
item_avg_weight = sqldf("select Item_Identifier,AVG(Item_Weight) as 'Item_Weight' from data Group by data.Item_Identifier")

#Get a boolean variable specifying missing Item_Weight values
miss_bool = is.na(data['Item_Weight'])

#Impute data and check #missing values before and after imputation to confirm
cat('Orignal #missing: ', sum(miss_bool))
data[miss_bool,'Item_Weight'] = item_avg_weight[data[miss_bool,'Item_Identifier'],'Item_Weight']
cat('Final #missing: ', sum(is.na(data['Item_Weight'])))
```

This confirms that the column has no missing values now. Lets impute Outlet_Size with the mode of the Outlet_Size for the particular type of outlet.


```{r}

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
```

This confirms that there are no missing values in the data. Lets move on to feature engineering now.

## 4. Feature Engineering

We explored some nuances in the data in the data exploration section. Lets move on to resolving them and making our data ready for analysis. We will also create some new variables using the existing ones in this section.

## Step 1: Consider combining Outlet_Type

During exploration, we decided to consider combining the Supermarket Type2 and Type3 variables. But is that a good idea? A quick way to check that could be to analyze the mean sales by type of store. If they have similar sales, then keeping them separate won’t help much.

```{r}
tmp <- aggregate(train$Item_Outlet_Sales,list(train$Outlet_Type),mean)
names(tmp) <- c("Outlet_Type","Mean")
tmp
rm(tmp)
```

This shows significant difference between them and we’ll leave them as it is. Note that this is just one way of doing this, you can perform some other analysis in different situations and also do the same for other features.

## Step 2: Modify Item_Visibility

We noticed that the minimum value here is 0, which makes no practical sense. Lets consider it like missing information and impute it with mean visibility of that product.

```{r}
#Determine average visibility of a product
visibility_avg = aggregate(data$Item_Visibility, list(data$Item_Identifier),mean)
names(visibility_avg) <- c("Item_Identifier","Item_visibility_avg")

#Impute 0 values with mean visibility of that product:
miss_bool = (data['Item_Visibility'] == 0)

cat('Number of 0 values initially:',sum(miss_bool))
data[miss_bool,'Item_Visibility'] = visibility_avg[data[miss_bool,'Item_Identifier'],'Item_visibility_avg']
cat('Number of 0 values after modification:',sum(data['Item_Visibility'] == 0))
```

So we can see that there are no values which are zero.

In step 1 we hypothesized that products with higher visibility are likely to sell more. But along with comparing products on absolute terms, we should look at the visibility of the product in that particular store as compared to the mean visibility of that product across all stores. This will give some idea about how much importance was given to that product in a store as compared to other stores. We can use the ‘visibility_avg’ variable made above to achieve this.

```{r}
#Determine another variable with means ratio
data$Item_Visibility_MeanRatio = data$Item_Visibility/visibility_avg[data$Item_Identifier,'Item_visibility_avg'] 
summary(data$Item_Visibility_MeanRatio)
```

Thus the new variable has been successfully created. Again, this is just 1 example of how to create new features. I highly  encourage you to try more of these, as good features can drastically improve model performance and they invariably prove to be the difference between the best and the average model.

## Step 3: Create a broad category of Type of Item

Earlier we saw that the Item_Type variable has 16 categories which might prove to be very useful in analysis. So its a good idea to combine them. One way could be to manually assign a new category to each. But there’s a catch here. If you look at the Item_Identifier, i.e. the unique ID of each item, it starts with either FD, DR or NC. If you see the categories, these look like being Food, Drinks and Non-Consumables. So I’ve used the Item_Identifier variable to create a new column:
  
  ```{r}
#Get the first two characters of ID:
data$Item_Type_Combined = substr(data$Item_Identifier,1,2)

#Rename them to more intuitive categories:
library(plyr)
data$Item_Type_Combined = revalue(data$Item_Type_Combined,c("FD"="Food","NC"="Non-Consumable","DR"="Drinks"))
count(data$Item_Type_Combined)
```

## Another idea could be to combine categories based on sales. The ones with high average sales could be combined together. I leave this for you to try.

## Step 4: Determine the years of operation of a store

We wanted to make a new column depicting the years of operation of a store. This can be done as:
  ```{r}
#Years:
data$Outlet_Years = 2013 - data$Outlet_Establishment_Year
summary(data$Outlet_Years)
```

This shows stores which are 4-28 years old. Notice I’ve used 2013. Why? Read the problem statement carefully and you’ll know.

## Step 5: Modify categories of Item_Fat_Content

We found typos and difference in representation in categories of Item_Fat_Content variable. This can be corrected as:
  
  ```{r}
#Change categories of low fat:
cat('Original Categories:')
count(data$Item_Fat_Content)

cat('\nModified Categories:')
data$Item_Fat_Content = revalue(data$Item_Fat_Content,c('LF'='Low Fat',
                                                        'reg'='Regular',
                                                        'low fat'='Low Fat'))
count(data$Item_Fat_Content)
```

Now it makes more sense. But hang on, in step 4 we saw there were some non-consumables as well and a fat-content should not be specified for them. So we can also create a separate category for such kind of observations

```{r}
#Mark non-consumables as separate category in low_fat:
levels(data$Item_Fat_Content) <- c(levels(data$Item_Fat_Content),"Non-Edible")
data$Item_Fat_Content[data$Item_Type_Combined=="Non-Consumable"] <- "Non-Edible"

count(data$Item_Fat_Content)
```

## Step 6: Numerical and One-Hot Coding of Categorical variables

Convert all categories of nominal variables into numeric types. Also, I wanted Outlet_Identifier as a variable as well. So I created a new variable ‘Outlet’ same as Outlet_Identifier and coded that. Outlet_Identifier should remain as it is, because it will be required in the submission file.

Lets start with coding all categorical variables as numeric using ‘LabelEncoder’ from sklearn’s preprocessing module.

```{r}
# #New variable for outlet
# library(CatEncoders)
# 
# #usage for LabelEncoder
# fit=LabelEncoder.fit(data$Outlet_Identifier)
# data$Outlet=transform(fit,data$Outlet_Identifier)
#  
# 
# #### usage for OneHotEncoder ###
# var_mod = c('Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet')
# le = LabelEncoder()
# for (i in var_mod){
#     fit=OneHotEncoder.fit(data[i])
#     data[i]=transform(fit,data[i])
# }
```

One-Hot-Coding refers to creating dummy variables, one for each category of a categorical variable. For example, the Item_Fat_Content has 3 categories – ‘Low Fat’, ‘Regular’ and ‘Non-Edible’. One hot coding will remove this variable and generate 3 new variables. Each will have binary numbers – 0 (if the category is not present) and 1(if category is present). This can be done using ‘get_dummies’ function of Pandas.

Lets look at the datatypes of columns now:
  
  ```{r}
str(data)
```

# Here we can see that all variables are now float and each category has a new variable. Lets look at the 3 columns formed from Item_Fat_Content.
# 
# head(data['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2'])
# You can notice that each row will have only one of the columns as 1 corresponding to the category in the original variable.

## Step 7: Exporting Data

Final step is to convert data back into train and test data sets. Its generally a good idea to export both of these as modified data sets so that they can be re-used for multiple sessions. This can be achieved using following code:
  
  ````{r}

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

#Export files as modified versions:
write.csv(trainm, "trainmodified.csv",row.names=FALSE)
write.csv(testm, "testmodified.csv",row.names=FALSE))
```

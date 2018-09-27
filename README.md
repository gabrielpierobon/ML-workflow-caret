# A comprehensive Machine Learning Workflow with multiple modelling using R's caretEnsemble

### Introduction

I'm going to use a very interesting dataset presented in the book *Machine Learning with R* from *PACKT Publishing*, written by *Brett Lantz*. My intention is to expand the analysis on this dataset by executing a full data science workflow which I've been laying out for some time now.

If you are thinking this is nothing new, then you're absolutely right! I'm not coming up with anything new here, but making sure I have all the tools necessary to follow a machine learning process without leaving behind any detail. Hopefully you will find it useful and be sure you are going to find some judgment errors from my part and/or things you would do differently. Feel free to leave me a comment and help me improve!

So moving forward and just in case, whenever the text is written in *italics*, that means the quote is from the author of the book, meaning I just copied and pasted some parragraphs (mostly for the description of the dataset, nothing in terms of actual analysis).

Why did I chose this specific dataset? The reason is that my father is an engineer and I wanted to show him how useful data science can be in modeling something related to that field.

Let's jump ahead and begin to understand what information we are going to work with:

### Modeling the strength of concrete

*In the field of engineering, it is crucial to have accurate estimates of the performance of building materials. These estimates are required in order to develop safety guidelines governing the materials used in the construction of building, bridges, and roadways.
Estimating the strength of concrete is a challenge of particular interes. Although it is used in nearly every construction project, concrete performance varies greatly due to a wide variety of ingredients that interact in complex ways. As a result, it is difficult to accurately predict the strength of the final product. A model that could reliably predict concrete strength given a listing of the composition of the input materials could result in safer construction practices.*

*For this analysis, we will utilize data on the compressive strength of concrete donated to the UCI Machine Learning Data Repository (http://archive.ics.uci.edu/ml) by I-Cheng Yeh*

*According to the website, the concrete dataset contains 1,030 examples of concrete with eight features describing the components used in the mixture. These features are thought to be related to the final compressive strength and they include the amount(in kilograms per cubic meter) of cement, slag, ash, water, superplasticizer, coarse aggregate, and fine aggregate used in the product in addition to the aging time (measured in days).*

### A machine learning workflow

I've found that splitting the task in 6 parts works the best for me. In that sense, I'll describe this instances as:  
1) Setting  
2) Exploratory Data Analysis  
3) Feature Engineering  
4) Data Preparation  
5) Modeling  
6) Conclusion  

In practice, I end up jumping from one to another and many times regardless of order. So think of this just as an initial structure and more of a checklist, rather than a step by step guide.

Again, this is something that I put together from many other sources, so you probably find this trivial or obvious. Hopefully you can take something from here anyways!

The checklist is very comprehending and covers practically everything that you would follow in supervised learning, for both classification and regression problems. In practice, many of the points from the list you will end up skipping. Even some of them are sometimes a little redundant (although double checking is always useful). I'm sure that with more practice this list will get tuned and I'm hoping I can also share that in the future.

I'll limit myself to short-direct answers, and will expand whenever necessary to put more work. Otherwise I'm affraid I can make this too long for anyone to stick around!

You will see that even though I'm including my full checklist, in many cases we don't have to do absolutely anything. This is because this dataset is pretty straightforward. However, I'll make sure to mention why we don't have to do anything and the idea of that checkpoint.

The last thing I wanted to mention is that when I get to the Modelling phase, I will change my approach and showcase some useful modelling tools with caret and caretEnsemble. I recon that the modelling part is very dependent on one's particular choice of algorithm and library to use. Since this is already a long read, I'll try to not spend so much thime there and leave that for a future publication.

Enough said, let's begin:

### 1) SETTING

##### 1.1) What are we trying to predict?

We need to accurately estimate the performance of building materials for the field of engineering. 

##### 1.2) What type of problem is it? Supervised or Unsupervised Learning? Classification or Regression? Binary or Multiclass? Univariate or Multivariate? Clustering?

This is a multivariate supervised machine learning problem in which we have to predict numeric outcomes, thus we'll be using regression techniques.

##### 1.3) What type of data do we have? Asses it's size

Our data is in csv format. It presents a header row with the column names. It seems to contain only numeric information.

##### 1.4) Import the dataset

We can simply load it with this code:

```{r}
concrete <- read.csv("/Users/Gabriel Pierobon/Desktop/Datasets para cargar/concrete.csv")
```

This is the size of our dataset:

```{r}
dim(concrete)
```

1,030 rows and 9 columns, one of which is our response/target variable.

We can see that it was imported as a data frame which is the format that we require to work with this data:

```{r}
class(concrete)
```

We can also check that the names of our columns were correctly imported.

```{r}
names(concrete)
```

From that list, we identify "strength" as our response/target variable.

We can conclude that our data was correctly imported and thus end our "Setting" phase. 

Before we do that, I'd like to require all our libraries alltogether. It's useful to do it all at once at the beggining, to improve readability of this work:

```{r message=FALSE, warning=FALSE}
library(dplyr)
library(PerformanceAnalytics)
library(ggplot2)
library(ggthemes)
library(corrplot)
library(car)
library(psych)
library(caret)
library(caretEnsemble)
library(doParallel)
```

With that taken care of, let's move on to exploratory data analysis:

### 2) Exploratory Data Analysis (EDA)

##### 2.1) View Data (str or dplyr's glimpse). First look. Anything strange?

The fist way we want to double check what we did in the "Setting" phase is to quickly view our full dataset. We do it like this:

```{r}
View(concrete)
```

This will display a window with the dataset. I like to quickly look at it from top to bottom and left to right to reassure the fact that the data was loaded. It's also a quick and dirty way to detect issues that could be observable at first glance. You don't want to do this for significantly large datasets.

Next, we are going to take a glimpse at our data and observe just the first few rows of the data frame:

```{r}
glimpse(concrete)
```

```{r}
head(concrete)
```

##### 2.2) Is it a "tidy" dataset? Need to *gather* or *spread* it? Is it presented in a way we can work with?

We need our data to be presented with individual observations as rows and features as columns. Fortunately for us, this is the case of this dataset, so we won't have to transform it. Otherwise, we would have to use some sort of function to "pivot" the data frame to accomodate to our needs.

##### 2.3) Rownames and colnames ok? Should we change them?

We have already checked how these were loaded. Do we need to update/change anything to have clearer understanding of our variables? I don't think this is the case, we can move on. The idea here is to make sure we move forward comfortably with our feature names, avoiding unnecessary long names or any other sort of confusing situation.

##### 2.4) Check data types (numeric, integr, dbl, char, factor, date). Are data types ok? If not, convert

As we have seen, all our variables are of the "double" type, except for the variable "age" which is "integer". Fortunately, we don't have to convert anything! It's very important that we check this in order to avoid some type of error when loading the data. Sometimes a single character in a numeric column can result in the whole column being loaded as "character".

##### 2.5) What is our Response Variable? Class imbalance? Study it.

Our response variable is "strength". Let's look at some useful statistics:

```{r}
summary(concrete$strength)
```

We can see that it ranges from 2.33 to 82.60

The median and mean are really close, but since median is actually smaller, this results in a skight skew to the right.

We can observe that with a plot using ggplot2:

```{r}
getmode <- function(v) {
   uniqv <- unique(v)
   uniqv[which.max(tabulate(match(v, uniqv)))]}


ggplot(data = concrete) +
  geom_histogram(mapping = aes(x = strength), bins = 15, boundary = 0, fill = "gray", col = "black") +
  geom_vline(xintercept = mean(concrete$strength), col = "blue", size = 1) +
  geom_vline(xintercept = median(concrete$strength), col = "red", size = 1) +
  geom_vline(xintercept = getmode(concrete$strength), col = "green", size = 1) +
  annotate("text", label = "Median = 34.45", x = 23, y = 100, col = "red", size = 5) +
  annotate("text", label = "Mode = 33.4", x = 23, y = 125, col = "black", size = 5) +
  annotate("text", label = "Mean = 35.82", x = 45, y = 45, col = "blue", size = 5) +
  ggtitle("Histogram of strength") +
  theme_bw()
```

As we can see, the distribution of the "strength" variable is not perfectly normal, although we are going to proceed regardless. This shouldn't be a problem because it's close.

Additionally, since this is a regression problem, we don't have to worry about class imbalance. In classification, you want to have a balanced class in your response variable.

##### 2.6) Rest of the features. Summary statistics. Understand your data

Here we extend the statistic analyisis to our other variables. We want to pay attention at the minimums and maximums. Also, mean and median difference is something to be concerned about. We would like all our variables to follow the normal distribution as much as possible.

```{r}
summary(concrete)
```

We can follow our analysis with a correlation plot. This will present us with a chart showing the correlation between all variables. It will also let us think for the first time if we need all our variables in our model. We don't want our feature variables to present a high correlation between them. We will take care of this later.

```{r}
corrplot(cor(concrete), method = "square")
```

```{r}
chart.Correlation(concrete)
```


At this point my first conclusion is that the variable "ash" has low correlation with our response variable "strentgh" and a high correlation with most of the other features. It is thus a strong candidate to be removed.


##### 2.7) Categorical data/Factors: create count tables to understand different categories. Check all of them.

In this case we are not working with categorical features. We can move on.

##### 2.8) Unnecessary columns? Columns we can quickly understand we don't need. Drop them

Here we want to look for columns which are totally useless. Anything that would come in the dataset that is really uninformative and we can determine that we should drop. Additional index columns, uninformative string columns, etc. This is not the case for this dataset. You could perfectly do this as soon as you import the dataset, no need to do it specifically at this point.

##### 2.9) Check for missing values. How many? Where? Delete? Impute?

Let's first do an overall check:

```{r}
anyNA(concrete)
```

Wonderful! no missing values in the whole set! Let me also show you a way we could have detected this for every column:

```{r}
sapply(concrete, {function(x) any(is.na(x))})
```

##### 2.10) Check for outliers and other inconsistent data points. Boxplots. DBSCAN for outlier detection?

Outlier detection is more of a craft than anything else, in my opinion. I really like the approach of using DBSCAN clustering for outlier detection but I'm not going to proceed with this so I don't overextend this analysis. DBSCAN is a clustering algorithm that can detect "noise" points in the data and not assign them to any cluster. I find it very compelling for outlier detection.

Instead, I'll just go ahead with a boxplot and try to work with the points I consider relevant just by sight:

```{r}
boxplot(concrete[-9], col = "orange", main = "Features Boxplot")
```

We see that there are several potential outliers, however I consider that the "age" feature could be the most problematic. Let's look at it isolated:

```{r}
boxplot(concrete$age, col = "red")
```

Are these just 4 outliers? If so, we could just get rid of them. Or shouldn't we?

Let's find out what this values are and how many of them are there.

```{r}
age_outliers <- which(concrete$age > 100)
concrete[age_outliers, "age"]
```

Oh my! so there were 62 of them instead of just 4! This is because the numbers are repeated several times. This makes me think that this age points should be relevant and we wouldn't want to get rid of them. 62 data points from our 1,030 dataset seems too high a number to just eliminate (we would be losing plenty of information).


##### 2.11) Check for multicollinearity in numeric data. Variance Inflation Factors (VIF)

We've already seen in the correlation plot presented before that there seems to be significant correlation between features. We want to make sure that multicollinearity is not an issue that prevents us to move forward. In order to do this, we will compute  a score called Variance Inflation Factor (VIF) which measures how much the variance of a regression coefficient is inflated due to multicollinearity in the model. If VIF is more than 10, multicolinearity is strongly suggested and we should try to get rid of the features that are causing it.

First, we generate a simple linear regression model of our target variable explained by each other feature. Afterwards, we call function vif() on the model object and we take a look at the named list output:

```{r}
simple_lm <- lm(strength ~ ., data = concrete)
vif(simple_lm)
```

Even though there are many variables scoring 5 and higher, none of them surpasses the threshold of 10 so we will consider multicollinearity is not to be a big issue. Howerver, some would argue that it could indeed be a problem to have as many features with scores of 7. We will not worry about that at this time.

With this, we consider the EDA phase over and we move on to Feature Engineering. Feel free to disagree with this being over! I'm sure there still plenty of exploration left to do.


### 3) FEATURE ENGINEERING

Feature engineering could also be considered a very important craft for a data scientist. It involves the production of new features obtained from the rest of the features present in the dataset. This could be as simple as extracting some dates information from string columns, or producing interaction terms. Moreover, it will certainly require some degree of subject matter expertise, since coming up with new informative features is something intrinsec to the nature of the data and the overall field of application.

Here we will go super quick, since our data seems to be already very informative and complete.

I wanted to mention that engineering new features will most likely require that we repeat some of the previous analysis we have done, we don't want to introduce new features and add problems to the dataset. So bear in mind that we must always look at our checklist back again and define if we need to analyze things one more time.

##### 3.1) Create new useful features. Interaction Terms. Basic math and statistics. Create categories with ifelse structures.

Because I'm not a subject matter expert in egineering, I won't create new features from interaction terms. I'll just limit myself to verify if any of the variables require any type of transformation.

The first thing that I want to do is to check two variables which seem to have unusual distributions. It is the case of "age" and "superplastic". Let's plot their original form and logs below.

```{r}
par(mfrow = c(2,2))
hist(concrete$age)
hist(concrete$superplastic)
hist(log(concrete$age), col = "red")
hist(log(concrete$superplastic), col = "red")
```

While I feel comfortable with converting "age" to "log of age", in the case of "superplastic" with as many observations being 0, I'll have some issues when taking the log of 0, so I'll manually set those to 0.

Below, the code to convert both features:

```{r}
concrete$age <- log(concrete$age)

concrete$superplastic <- log(concrete$superplastic)
concrete$superplastic <- ifelse(concrete$superplastic == -Inf, 0, concrete$superplastic)

head(concrete)
```

I spent quite some time at this point trying to create a new superplastic feature by binning the original superplastic feature into 3 numeric categories. However I didn't have much success in terms of importance to explain the target variable. I won't show those attempts, but just know that I indeed tried to work on that for some time unsuccesfully. Failing is also part of learning!


##### 3.2) Create dummies for categorical features. Preferred One-Hot-Encoding

We are not working with categorical features at this time, so this section will be skipped. However, if you were presented with some factor columns, you will have to make sure your algorithm can work with those, or otherwise proceed with one-hot-encoding them.

##### 3.3) Can we extract some important text from string columns using regular expressions?

We are not working with sting data at this time, so this section will be skipped. This should be a very useful tool when you have some relevan information in character type from which you can create new useful categories.

##### 3.4) can we create new datetime columns from date features?

We are not working with datetime columns at this time, so this section will be skipped. Here we are thinking in terms of extracting month names, week days, time, etc. 

Thankfully we didn't have to go through all that! However, those last 4 sections are a MUST if we are working with that type of information. I'll make sure to analyze a dataset in which I can show some of that.

Let's move on to the Data Preparation phase:


### 4) DATA PREPARATION

##### 4.1) Manual feature selection. Remove noisy, uninformative, highly-correlated, or duplicated features.

Here's where we take spend time for the second time in this workflow looking at our features and detecting if any of them are uninformative enough or should be dropped because of introducing problems as could be multicollinearity.

As I already had decided, I'll first drop the "ash" feature.

```{r}
concrete$ash <- NULL

head(concrete)
```

Besides that, I won't remove anything else.

##### 4.2) Transform data if needed. Scale or normalize if required.

Most of the algorithms we will use require our numeric features to be scaled or normalized.

We won't be doing that in this precise section, but leave it for later, since caret allows us to do some pre-processing within it's training functionallity. This is specifically useful because the algorithm will automatically transform back to it's original scale when presenting us with the results. If I manually normalize the datset here, then I have to remember to convert back the predictions.

##### 4.3) Automatic Feature extraction. Dimensionality Reduction (PCA, NMF, t-SNE, Factor Analysis)

If removing features manually isn't straightforward enough but we still consider our dataset to contain too many features, we can use some dimensonality reduction techniques. Principal Components Analysis is one of those techinques, and a really useful one to both simplify our dataset, and also remove the issue of multicollinearity for good. 

With just 8 features in our dataset, and 1 already dropped manually, I don't consider we require to reduce dimensionallity any further.

##### 4.4) Is our dataset randomized?

We are not sure it is randomized, so we will shuffle it just in case:

```{r}
set.seed(123)
concrete_rand <- concrete[sample(1:nrow(concrete)), ]
dim(concrete_rand)
```

##### 4.5) Define an evaluation protocol: how many samples we have? Hold out method. Cross validation needed?

We have 1,030 samples, so this is definately a small dataset. We will divide the dataset into train and test sets and make sure we use cross validation when we train our model. In that way we ensure we are using our few observations as well as we can.

##### 4.6) Split dataset into train & test sets (set seed for replicability)

We first create a set of predictors and a set of our target variable

```{r}
X = concrete_rand[, -8]
y = concrete_rand[, 8]
```

We check everything is ok:

```{r}
str(X)
```

```{r}
str(y)
```

We then proceed to split our new "X" (predictors) and "y" (target) sets into training and test sets.

Note: you don't have to separate the sets, and can perfectly go ahead using the "formula" method. I just prefer to do it this way, because this way I'm sure I allign my proceeding to how I'd work with Python's scikit-learn.

We will use caret's createDataPartition() function, which generates the partition indexes for us, and we will use them to perform the splits:

```{r}
set.seed(123)

part.index <- createDataPartition(concrete_rand$strength, p = 0.75, list = FALSE)

X_train <- X[part.index, ]
X_test <- X[-part.index, ]
y_train <- y[part.index]
y_test <- y[-part.index]
```

So now we have 4 sets. Two predictors sets splitted into train and test, and two target sets splitted into train and test. All of them using the same index for partitioning.

Once again, let's check everything worked just fine:

```{r}
str(X_train)
str(X_test)
str(y_train)
str(y_test)
```

Ok! We are good to go! Now to the modeling phase!

### 5) MODELING

As I mentioned in the introduction, in this phase I will change my approach, and instead of going through a check list of things to do, I will summarise altogether how we will proceed.

* We will use package caretEnsemble in order to train a list of models all at the same time  
* This will allow us to use the same 5 fold cross validation for each model, thanks to caret's functionallity  
* We will allow parallel processing to boost speed 
* We won't focus on the nature of the algorithms. We will just use them comment on the results  
* We will use a linear model, a support vector machines with radial kernel, a random forest, a gradient boosting tree and a gradient boosting linear model 
* We won't do manual hyperparameter tuning, instead we will allow caret to go through some default tuning in each model  
* We will compare performance over training and test sets, focusing on RMSE as our metric (root mean squared error)  
* We will use a very cool functionallity from caretEnsemble package and will ensemble the model list and then stack them in order to try to produce an ultimate combination of models to hopefully improve perfomance even more

So let's move on

We first set up parallel processing and cross validation in trainControl()

```{r}
registerDoParallel(4) # here I'm using 4 cores from my computer
getDoParWorkers()

set.seed(123) # for replicability

my_control <- trainControl(method = "cv", # for "cross-validation"
                           number = 5, # number of k-folds
                           savePredictions = "final",
                           allowParallel = TRUE)
```

We then train our list of models using the caretList() function by calling our X_train and y_train sets. We specify trControl with our trainControl object created above, and set methodList to a list of algorithms (check the caret package information to understand what models are available and how to use them).

```{r message=FALSE, warning=FALSE}
set.seed(222)

model_list <- caretList(X_train, # can perfectly use y ~ x1 + x2 + ... + xn formula instead
                        y_train,
                        trControl = my_control, # remember, 5 fold cross validation + allowparallel
                        methodList = c("lm", "svmRadial", "rf", "xgbTree", "xgbLinear"), # our 5 models
                        tuneList = NULL, # no manual hyperparameter tuning
                        continue_on_fail = FALSE, # stops if something fails
                        preProcess  = c("center","scale")) # as mentioned before, here we scale the dataset
```

Now that our caretList was trained, we can take a look at the results. We can access each separate model. Here's the SVM result:

```{r}
model_list$svmRadial
```

We won't go through each one of them. That's for you to check!

Let's go to our objective, which is finding the model that has the lowest root mean squared error. We first asses this for the training data.

```{r}
options(digits = 3)

model_results <- data.frame(LM = min(model_list$lm$results$RMSE),
                            SVM = min(model_list$svmRadial$results$RMSE),
                            RF = min(model_list$rf$results$RMSE),
                            XGBT = min(model_list$xgbTree$results$RMSE),
                            XGBL = min(model_list$xgbLinear$results$RMSE))


print(model_results)
```

In terms of RMSE, the extreme gradient boosting tree offers the best result, with 4.36 (remember the mean strength was 35.8)

caretEnsemble offers a functionality to resample this model list and plot the performance:

```{r}
resamples <- resamples(model_list)
dotplot(resamples, metric = "RMSE")
```

We can also see that the xgbTree is also presenting a smaller variance compared to the other models.

Next we will attempt to create a new model by ensembling our model_list, in order to find the best possible model, hopefully a model that takes the best things of the 5 we have trained and boosts performance.

Ideally, we would ensemble models with low correlation between them. In this case we will see that some high correlation is present, but we will choose to move on regardless, just for the sake of showcasing this amazing feature:

```{r}
modelCor(resamples)
```

Firstly, we train an ensemble of our models using caretEnsemble(), which will perform a linear combination of all the models in our model list.

```{r}
set.seed(222)
ensemble_1 <- caretEnsemble(model_list, metric = "RMSE", trControl = my_control)
summary(ensemble_1)
```

As we can see, we managed to reduce RMSE for the training set to 4.15

Here's a plot of our ensemble

```{r}
plot(ensemble_1)
```

The red dashed line is the ensemble's RMSE performance

Next, we can be more specific and try to do an ensemble using other algorithms. I tried some but wasn't able to improve performance. We will leave both regardless, to check which does best with unseen data.

```{r}
set.seed(222)

ensemble_2 <- caretStack(model_list, method = "glmnet", metric = "RMSE", trControl = my_control)

print(ensemble_2)
```

Finally, it's time to evaluate the performance of our models over unseen data, which is in our test set.

We first predict the test set with each model and then compute RMSE:

```{r}
pred_lm <- predict.train(model_list$lm, newdata = X_test)
pred_svm <- predict.train(model_list$svmRadial, newdata = X_test)
pred_rf <- predict.train(model_list$rf, newdata = X_test)
pred_xgbT <- predict.train(model_list$xgbTree, newdata = X_test)
pred_xgbL <- predict.train(model_list$xgbLinear, newdata = X_test)
predict_ens1 <- predict(ensemble_1, newdata = X_test)
predict_ens2 <- predict(ensemble_2, newdata = X_test)


pred_RMSE <- data.frame(ensemble_1 = RMSE(predict_ens1, y_test),
                        ensemble_2  = RMSE(predict_ens2, y_test),
                        LM = RMSE(pred_lm, y_test),
                        SVM = RMSE(pred_svm, y_test),
                        RF = RMSE(pred_rf, y_test),
                        XGBT = RMSE(pred_xgbT, y_test),
                        XGBL = RMSE(pred_xgbL, y_test))

print(pred_RMSE)
```

Surprisingly, the extreeme gradient boosting linear model out performs every other model on the test set, including our ensemble_1 and matching the performance of the ensemble_2

We see also that in general , there is a difference in performance compared to the training set. This is to be expected. We could still try to tune hyperparameters in order to reduce some overfitting, but at this point I believe we have achieved very strong performance over unseen data and I will leave further optimization for a future publication.

The last thing I wanted to show is variable importance. In order to do this, I will calculate our "xgbLinear" model separately, indicating I want to retain the variable importance and then plot it:

```{r}
set.seed(123)

xgbTree_model <- train(X_train,
                       y_train,
                       trControl = my_control,
                       method = "xgbLinear",
                       metric = "RMSE",
                       preProcess  = c("center","scale"),
                       importance = TRUE)

plot(varImp(xgbTree_model))

```

Here we can see the high importance of variables "age" and "cement" to the prediction of concrete's "stength". This was to be expected since we had already observed a high correlation between them in our correlation plot.

Working with the log of "age" has also allowed us to improve predictability (as verified separately)

Notice some "unimportant" features present in the chart. Should we have dropped them? Can we simplify our model and still get the same strong performance? Those are questions we could ask and try to resolve.

### 6) Conclusion

This has been quite the journey! We moved along most of the necessary steps in order to execute a complete and careful machine learning project. Even though we didn't have to do a whole lot of changes and transformation to the original data as imported from the csv, we made sure we understood why and what we should have done otherwise.

Moreover, I was able to showcase the fantastic work you can do with caret and caretEnsemble in terms of doing multiple modelling, quick and dirty, all at once and being able to quickly compare model performance. The more advanced data scientists and machine learning enthusiasts could possibly take this as a first draft before proceeding with more advanced algorithms and fine hyperparameter tuning that will get those extra bits of performance. For us, this proved to be really strong even with the basic configuration.

In the mentioned book, the author calculates correlation between predictions using a neural network with 5 hidden layers and the true values, obtaining a score of 0.924. It also mentions that compared to the original publication (the one his work was based on), this was a significant improvement (original publication achieved 0.885 using a similar neural network)

So how did we do computing the same correlation as the book's author?

```{r}
pred_cor <- data.frame(ensemble_1 = cor(predict_ens1, y_test),
                       ensemble_2 = cor(predict_ens2, y_test),
                       LM = cor(pred_lm, y_test),
                       SVM = cor(pred_svm, y_test),
                       RF = cor(pred_rf, y_test),
                       XGBT = cor(pred_xgbT, y_test),
                       XGBL = cor(pred_xgbL, y_test))

print(pred_cor)
```

Pretty strong performance!

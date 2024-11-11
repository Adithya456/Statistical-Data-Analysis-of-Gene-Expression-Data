---
title: 'Gene Expression Statistical Data Analysis'
author: "Nagadithya Bathala"
Registration number: ""
output:
  html_document:
    df_print: paged
  pdf_document: default
  word_document: default
---


\textbf{Explore the dataset and make the subset}

```{r, eval=TRUE}
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE) # turning off warnings

df <- read.csv(file="gene-expression-invasive-vs-noninvasive-cancer.csv") # reading csv file
print("Dimensions of the original data :") 
print(dim(df)) # finding the dimensions of the dataset

set.seed(2315880) # fixing the random seed using registration number
team.gene.subset <- rank(runif(1:4948))[1:2000] # getting indices of columns randomly using fixed seed

sub_df <- df[c(team.gene.subset)] # make a subset from data frame using randomly generated indices
sub_df['Class'] <- df['Class']

print("Dimensions of the sub data :") 
print(dim(sub_df))

print("Total missing values in data :") 
sum(is.na(sub_df))
```


```{r, eval=TRUE}

#sub_df$NM_005069 <- ifelse(is.na(sub_df$NM_005069),
 #                          median(sub_df$NM_005069, na.rm = TRUE),
  #                         sub_df$NM_005069) #replacing null value with median
```

```{r, eval=TRUE}

library("DMwR")
new_df <- knnImputation(data = sub_df, k = 10, scale = TRUE, meth = "weighAvg", distData = NULL)
dim(new_df)
sum(is.na(new_df))
```

```{r, eval=TRUE}

pca_ <- prcomp(new_df, cor = TRUE) # create principal components

summary(pca_, loadings = TRUE,cutoff=.1) # summarize created principal components
plot(pca_) # plot principal components

```

```{r, eval=TRUE}

plot(density(data$Contig57173_RC))
hist(data$Contig57173_RC)

```

```{r eval=TRUE}
library('dplyr')
new_df$Class <- case_when(
  new_df$Class == 1 ~ 0,
  new_df$Class == 2 ~ 1
)
```


```{r eval=TRUE}
library("caTools")

whole_ <- pca_$x[, 1:8]
whole_$Class <- new_df$Class

whole_
#sample = sample.split(new_df$Class, SplitRatio = .8)
#train = subset(new_df, sample == TRUE)
#test  = subset(new_df, sample == FALSE)

sample = sample.split(new_df$Class, SplitRatio = .8)
train = subset(whole_, sample == TRUE)
test  = subset(whole_, sample == FALSE)

print(dim(train))
print(dim(test))
```
```{r, eval=TRUE}

library("ggplot2")
ggplot(data = train, aes(x=Class))+ geom_bar()

```

```{r, eval=TRUE}
# Function to calculate misclassification and accuracy
library(MASS) # library for creating table

misclass <- function(target, pred){
  conf <- table(list(predicted=pred, observed=target))
  sensitivity <- conf[1] / (conf[1]  + conf[2])
  specificity <- conf[4] / (conf[4] + conf[3])
  misclassification <- (conf[3]+conf[2])/(conf[1]+conf[2]+conf[3]+conf[4])
  accuracy <- 1 - misclassification
  print("Confusion Matrix: ")
  print(conf)
  print("Sensitivity: ")
  print(sensitivity)
  print("Specificity: ")
  print(specificity)
  print("Misclassification Error: ")
  print(misclassification)
  print("Accuracy: ")
  print(accuracy)
}

# Prediction converter
convert_to_class <- function(output){
  output <- case_when(
    output > 0.5 ~ 0,
    output < 0.5 ~ 1,
    output == 0.5 ~ 0)
}
```



```{r, eval=TRUE}
# Logistic Regression
LR <- glm(train$Class ~., data = train, family = binomial(link = logit))

LR_prediction <- predict(LR, newdata = test, type = 'response')

LR_prediction <- case_when(
  LR_prediction > 0.5 ~ 0,
  LR_prediction < 0.5 ~ 1,
  LR_prediction == 0.5 ~ 0
)

misclass(test$Class, LR_prediction)

```

```{r, eval=TRUE}

# Poisson Regression
PR <- glm(train$Class ~., data = train, family = poisson(link = log))

PR_prediction <- predict(PR, newdata = test, type = 'response')
PR_prediction <- case_when(
  PR_prediction > 0.5 ~ 0,
  PR_prediction < 0.5 ~ 1,
  PR_prediction == 0.5 ~ 0
)
misclass(test$Class, PR_prediction)

```

```{r, eval=TRUE}
# Linear Discriminant Analysis 

LDA <- lda(Class ~.  , data=train) # perform LDA on subset
lda_prediction <- predict(LDA, newdata = test) 

misclass(test$Class, lda_prediction$class)
```


```{r, eval=TRUE}

measure_pca <- princomp(sub_df, cor = TRUE) # create principal components

summary(measure_pca, loadings = TRUE,cutoff=.1) # summarize created principal components
plot(measure_pca) # plot principal components
pairs(measure_pca$scores) # create pair plots of scores to explain clustering

```


```{r, eval=TRUE}
# Quadratic Discriminant Analysis 
train.subset <- rank(runif(1:2001))[1:50] # getting indices of columns randomly using fixed seed

sub_train <- df[c(train.subset)]
  
QDA <- qda(train$Class ~.  , data=sub_train) # perform LDA on subset
qda_prediction <- predict(QDA, newdata = test) 

misclass(test$Class, qda_prediction$class)

length(sub_train$NM_013396)
length(train$NM_013396)

```


```{r, eval=TRUE}

manova_pca <- manova(measure_pca$score[,1:2] ~ df$Class) # fitting MANOVA on 1st and 2nd principal components
summary(manova_pca,intercept=TRUE) # summarizing MANOVA

```


```{r, eval=TRUE}

library(MASS) # library for creating table
sub_df[["class"]] <- df[["Class"]] # adding "class" column to the subset

lda1 <- lda(class ~.  , data=sub_df) # perform LDA on subset
lda_prediction <- predict(lda1) # predict classes using created LDA

conf <- table(list(predicted=lda_prediction$class,
                   observed=sub_df$class)) # create confusion matrix
print(conf) # Confusion matrix for LDA

sensitivity <- conf[1] / (conf[1]  + conf[2]) # calculate sensitivity
print(sensitivity) # Sensitivity for LDA

specificity <- conf[4] / (conf[4] + conf[3]) # calculate specificity
print(specificity) # Specificity for LDA

misclassification <- (conf[3]+conf[2])/
  (conf[1]+conf[2]+conf[3]+conf[4]) # calculate misclassification error
print(misclassification) # misclassification for LDA

```


```{r, eval=TRUE}

qda1 <- qda(class ~.  , data=sub_df) # perform QDA on subset
qda_prediction <- predict(qda1) # predict using created QDA

qda_conf <- table(list(predicted=qda_prediction$class,
                       observed=sub_df$class)) # create confusion matrix for QDA
print(qda_conf) # Confusion for QDA

qda_sensitivity <- qda_conf[1] / (qda_conf[1]  + qda_conf[2]) # Calculate sensitivity 
print(qda_sensitivity) # Sensitivity for QDA

qda_specificity <- qda_conf[4] / (qda_conf[4] + qda_conf[3]) # Calculate specificity
print(qda_specificity) # Specificity for QDA

qda_misclassification <- (qda_conf[3]+qda_conf[2])/
  (qda_conf[1]+qda_conf[2]+qda_conf[3]+qda_conf[4]) # calculate misclassification error
print(qda_misclassification) # Misclassification for QDA

```



```{r, eval=TRUE}

median(measure_pca$score[,1]) # median of first principal component

sub_df[["predicted"]] <- NULL # creating new empty prediction column
for (x in 1:78){
  if (measure_pca$score[,1][x] > median(measure_pca$score[,1])){
    sub_df[["predicted"]][x] = 2}
  else{ sub_df[["predicted"]][x] = 1
  }
} # storing class 1 for eigen value less than median of principal component and class 2 for eigen value less than median of principal component

pca_conf <- table(list(predicted=sub_df$predicted,
                       observed=sub_df$class)) # create confusion matrix for pca
print(pca_conf) # Confusion matrix for 1st and 2nd principal component

pca_sensitivity <- pca_conf[1] /(pca_conf[1]  + pca_conf[2]) # calculate sensitivity for pca
print(pca_sensitivity) # Sensitivity matrix for 1st and 2nd principal component

pca_specificity <- pca_conf[4] / (pca_conf[4] + pca_conf[3]) # calculate specificity for pca
print(pca_specificity) # Specificity matrix for 1st and 2nd principal component

result <- fisher.test(pca_conf) # performing Fisher's test on PCA confusion matrix
print(result) # result of Fisher's test

youden_index <- pca_sensitivity + pca_specificity -1 # Calculate Youden Index for PCA
print(youden_index) # Youden index

```

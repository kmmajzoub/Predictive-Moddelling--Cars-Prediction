

cars = read.csv("Cars-dataset.csv")



### Problem: We need to need to predict if employees will use car as a transport
## and which are the variables that are significant 

## Libraries needed: 
library(car) # use for multicollinearity test (i.e. Variance Inflation Factor(VIF))
library(MASS) # use for basic statistics
library(dummies) # use for dummy variable transformation(i.e. One-Hot Encoding)
library(ggplot2) # use for visualisation
library(caret) # use for LM model training i.e Naive bayes (train() function )
library(Information) # use for calculating WOE and Information value
library(caTools)
library(ROCR) # use for ROC curve 
library(dplyr) # use for basic data wrangling
library(tidyr) # Converting data shape- long to wide or wide to long format
library(corrplot) # for correlation analysis 
library(ggplot2) # for visualization 
library(GGally) # for better visualization of multiple plots in one grid
library(factoextra) # use for PCA techinque
library(e1071) # using for machine learning models(i.e.Naive Bayes,KNN Models)

##Perform an EDA on the data - (7 marks)
### we have 418 obs and 9 variables 

head(cars)
table(cars$Transport)
str(cars)
# Insights:  
########## 1.Transport: Dependent variable, It is a factor type.
#          2.All the independent variables are numeric type or integers / gender is a factor 
#          3.Few of the variables are having a range of 0 to 1.
#-----------------------------------------------------------------------------------------
anyNA(cars)
sum(is.na(cars))
sum(is.na(cars$Age))
sum(is.na(cars$Gender))
sum(is.na(cars$Engineer))
sum(is.na(cars$MBA))
## we only have 1 na , so i will delete it , 1 out of 418 obs
cars = na.omit(cars)
anyNA(cars)
### we deleted 1 obs and now it is 417 obs
# Let's see the distribution of each variables 

summary(cars)

cars$Engineer = as.factor(cars$Engineer)
cars$MBA = as.factor(cars$MBA)
cars$license = as.factor(cars$license)
str(cars)
attach(cars) 
par(mfrow=c(1,1))
hist(Age)
boxplot(Age, horizontal = TRUE)
plot(density(Age))
## we have outliers here 

plot(Gender, main = "Gender")
plot(Engineer, main = "Engineer")
plot(MBA, main = "MBA")

boxplot(Work.Exp)
plot(density(Work.Exp))
hist(Work.Exp)
## it is not normal 
### we have outliers here 

boxplot(Salary)
plot(density(Salary))
hist(Salary)
## it is not normal 
### we have outlier here

boxplot(Distance)
hist(Distance)
plot(density(Distance))
### we have outlier here

plot(license)
plot(Transport)
### most employees use public transport 
### we want to predict on using the car 
## I think license and distance are the miost important factors related to getting a car

qqnorm(Age, main = "Age")
qqnorm(Work.Exp, main = "Work Exp")
qqnorm(Salary, main = "Salary")
qqnorm(Distance, main = "Distance")

boxplot(Age,Work.Exp,Salary,Distance, names = c( "Age" , "Work Exp", "Salary", "Distance") )
#multivariate 
par(mfrow=c(1,1))
?par

plot(Transport, Age, main= "Age vs Transport")
plot(Transport,Work.Exp, main ="Work Exp vs Transport" )
plot(Transport, Salary, main = "Salary vs Transport")
plot(Transport, Distance, main = "Distance vs Transport")

plot(Transport,license, main = "License vs Trans")
plot(Transport,Gender, main = "Gender vs Trans")
plot(Transport,Engineer , main = "Engineer vs Trans")
plot(Transport,MBA, main = "MBA vs Trans")


### Treating outliers 
boxplot(Age, horizontal = TRUE)
cars$Age = cars$Age[cars$Age<36]
summary(cars$Age)

new_vars <- c("Age","Gender","Engineer","MBA","Work.Exp","Salary","Distance","license","Transport")

str(cars)
cars$Gender = ifelse(cars$Gender=='Male',1,0)
cars$Gender = as.integer(cars$Gender)
cars$Engineer = as.integer(cars$Engineer)
cars$MBA = as.integer(cars$MBA)
cars$license = as.integer(cars$license)
str(cars)


outlier_treatment_fun = function(data,var_name){
  capping = as.vector(quantile(data[,var_name],0.9))
  flooring = as.vector(quantile(data[,var_name],0.01))
  data[,var_name][which(data[,var_name]<flooring)]<- flooring
  data[,var_name][which(data[,var_name]>capping)]<- capping
  #print('done',var_name)
  return(data)
}

for(i in new_vars[1:8]){
  cars =  outlier_treatment_fun(cars,i)
}

boxplot(Age,Gender,Engineer, MBA,Work.Exp,Salary,Distance,license, main = "Treated" , names = c("Age","Gender","Engineer","MBA", "Work.Exp", "Salary", "Distance", "license"))

cars$Transport = ifelse(cars$Transport == "Car","1","0")
cars$Transport = as.integer(cars$Transport)
corrplot(cor(cars[1:9]))

35/417 ## cars employes rate 

### Create multiple models and explore how each model perform using appropriate model performance metrics (15 marks)

################### Logistic Regression ( no need to treat the outliers) ################
## 70-30

log.cars = cars
str(log.cars)
## changing the Transport into binomial 

log.cars$Transport = as.factor(log.cars$Transport)
log.cars$Engineer = as.factor(log.cars$Engineer)
log.cars$MBA = as.factor(log.cars$MBA)
log.cars$Gender = as.factor(log.cars$Gender)
log.cars$license = as.factor(log.cars$license)

summary(log.cars)
str(log.cars)

set.seed(123)

split.indices = sample.split(log.cars$Transport,SplitRatio = .7)
logistic.train.cars = log.cars[split.indices,]
logistic.test.cars = log.cars[!split.indices,]
print(nrow(logistic.test.cars)/nrow(log.cars))
print(nrow(logistic.train.cars)/nrow(log.cars))
summary(logistic.test.cars)
summary(logistic.train.cars)

logistic.test.model = glm(Transport~Gender+Engineer+MBA+license ,data = logistic.test.cars,family = "binomial")
logistic.train.model = glm(Transport~Gender+Engineer+MBA+license,data = logistic.train.cars,family = "binomial")
logistic.test.model
logistic.train.model
 
summary(logistic.test.model)
summary(logistic.train.model)

vif(logistic.test.model)
vif(logistic.train.model)
    
## Factors that are more than 5 are considered a concern 
## As shown we will build the model on Distance and license 
logistic.test.model = glm(Transport~license,data = logistic.test.cars,family = "binomial")
summary(logistic.test.model)
vif(logistic.test.model)

logistic.train.model = glm(Transport~license,data = logistic.train.cars,family = "binomial")
summary(logistic.train.model)
vif(logistic.train.model)

## I am happy with this model 
# for every extra KM in distance the car wantinf increase by 90% 
plot(log.cars$Transport,log.cars$license)
## this shows exactly the result
logistic.test.model$fitted.values
logistic.train.model$fitted.values
par(mfrow=c(1,2))
plot(logistic.test.cars$Transport,logistic.test.model$fitted.values)
plot(logistic.train.cars$Transport,logistic.train.model$fitted.values)


### this shows that below probabilty of 0.9 it is more likely to prepfer a car 
logistic.test.Predict = ifelse(logistic.test.model$fitted.values<.9,"no car" , "yes car")
logistic.train.Predict = ifelse(logistic.train.model$fitted.values<.9,"no car" , "yes car")
summary(logistic.test.Predict)
summary(logistic.train.Predict)
table(logistic.test.cars$Transport,logistic.test.Predict)
table(logistic.train.cars$Transport,logistic.train.Predict)
summary(logistic.test.cars$Transport)
summary(logistic.train.cars$Transport)

library(pROC)
roc(logistic.test.cars$Transport,logistic.test.model$fitted.values)
plot(roc(logistic.test.cars$Transport,logistic.test.model$fitted.values))
roc(logistic.train.cars$Transport,logistic.train.model$fitted.values)
plot(roc(logistic.train.cars$Transport,logistic.train.model$fitted.values))


# The ROC = 0.87 which is an exceelent model for both models 

############################################## Naive Bais ##############################


nb.cars = cars

nb.cars$Engineer = as.factor(nb.cars$Engineer)
nb.cars$MBA = as.factor(nb.cars$MBA)
nb.cars$Gender = as.factor(nb.cars$Gender)
nb.cars$license = as.factor(nb.cars$license)
nb.cars$Transport = ifelse(nb.cars$Transport == "1","car","no car")
nb.cars$Transport = as.factor(nb.cars$Transport)

str(nb.cars)
summary(nb.cars)


## train - test 
spilt.indices = sample.split(nb.cars$Transport,SplitRatio = .7)

NB.train.cars = nb.cars[split.indices,]
NB.test.cars = nb.cars[!split.indices,]
print(nrow(NB.test.cars)/nrow(nb.cars))
print(nrow(NB.train.cars)/nrow(nb.cars))


## model
set.seed(123)

NB.model.train = naiveBayes(Transport~Age+Work.Exp+Salary+Distance , data = NB.train.cars) 
NB.model.test = naiveBayes(Transport~Age+Work.Exp+Salary+Distance , data = NB.test.cars) 

NB.model.train
NB.model.test

predict.NB.model.train = predict(NB.model.train,type = "raw",newdata = nb.cars)
plot(nb.cars$Transport,predict.NB.model.train[,2])

predict.NB.model.test = predict(NB.model.test,type = "raw",newdata = nb.cars)
plot(nb.cars$Transport,predict.NB.model.test[,2])

summary(predict.NB.model.test)
summary(predict.NB.model.train)

pred.NB.test = predict(NB.model.test,NB.test.cars,type = "raw")
pred.NB.train = predict(NB.model.train,NB.train.cars,type = "raw")
summary(pred.NB.test)
summary(pred.NB.train)

NB.predict.response.train= factor(ifelse(pred.NB.train >= 0.9, "car","no car"))
NB.predict.response.test= factor(ifelse(pred.NB.test >= 0.9, "car","no car"))

summary(NB.predict.response.test)
summary(NB.predict.response.train)
table(NB.test.cars$Transport,NB.predict.response.test)
table(logistic.train.cars$Transport,logistic.train.Predict)
summary(logistic.test.cars$Transport)
summary(logistic.train.cars$Transport)


NB.test.matrix = confusionMatrix(NB.predict.response.test, NB.test.cars$Transport , positive = "car")

######## KNN ###### 

knn.cars = cars
knn.cars$Gender = as.factor(knn.cars$Gender)
knn.cars$Transport = ifelse(knn.cars$Transport == "1","car","not car")
knn.cars$Transport = as.factor(knn.cars$Transport)
knn.cars$Engineer = as.factor(knn.cars$Engineer)
knn.cars$MBA = as.factor(knn.cars$MBA)
knn.cars$license = as.factor(knn.cars$license)

str(knn.cars)
summary(knn.cars)

set.seed(1)
dim(knn.cars)
index=sample(417,317)
KNN.train.cars = knn.cars[index,]
KNN.test.cars = knn.cars[-index,]
dim(KNN.test.cars)
dim(KNN.train.cars)


library(class)
knn.model = knn (KNN.train.cars[,c(6,7)],KNN.test.cars[,c(6,7)], KNN.train.cars$Transport ,k=5)

table(KNN.test.cars$Transport,knn.model)

summary(knn.model)

conf_KNN = confusionMatrix(knn.predicted.response,KNN.test.cars$Transport)

##################Apply both bagging and boosting modeling procedures to create 2 models ##########
library(gbm)
library(xgboost)
library(caret)
library(ipred)
library(rpart)

### Bagging 
bag.cars = cars
bag.cars$Transport= as.factor(bag.cars$Transport)
str(bag.cars)
split = sample.split(bag.cars$Transport, SplitRatio = .75)

bag.cars.train = subset(bag.cars, split == FALSE)
bag.cars.test = subset(bag.cars, split == TRUE)


table(bag.cars.test$Transport)
table(bag.cars.train$Transport)
str(bag.cars.test)
str(bag.cars.train)

cars.bagging.train = bagging(Transport~. , data = bag.cars.train, rpart.control(maxdepth = 5 , minsplit = 15))
cars.bagging.test = bagging(Transport~.,data = bag.cars.test,rpart.control(maxdepth = 5 
                                                                               , minsplit = 15))  


bagging.cars.train$pred.class = predict (cars.bagging.test, bagging.cars.train)
bagging.cars.test$pred.class = predict (cars.bagging.rain, bagging.cars.test)

table(bagging.cars.test$Transport,bagging.cars.test$pred.class)
table(bagging.cars.train$Transport,bagging.cars.train$pred.class)




################### Boosting################
?gbm.fit

gbm.fit = gbm(formula = Transport~. , distribution = "gaussian",
              data = bag.cars.train , n.trees = 1000, interaction.depth=1 
              , shrinkage = .001 , cv.folds = 5 , n.cores = NULL , verbose = FALSE )


bag.cars.test$pred.class = predict(gbm.fit,bag.cars.test, type = "response")

table(bag.cars.test$Transport, bag.cars.test$pred.class>.5)



### XG Boost 
?xgboost


feature.train = as.matrix(bag.cars.train[,1:8])
label_train = as.matrix(bag.cars.train[,9])
feature.test = as.matrix(bag.cars.test[,1:8])


xgb.fit = xgboost(data = feature.train, label = label_train, eta = .001,
                  max_depth = 3, min_child_weight = 3 , nrounds = 10000, nfold = 5,
                  objective = "binary:logistic" , verbose = 0 , early_stopping_rounds =10)


bag.cars.test$xgb.pred.class = predict(xgb.fit,bag.cars.test)

  table(bag.cars.test$Transport, bag.cars.test$pred.class>.5)



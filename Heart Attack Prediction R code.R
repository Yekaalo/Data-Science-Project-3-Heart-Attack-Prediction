library(caret)
library(pROC)

# Read in Dataset
data<-read.csv("heart.csv")

#See summary of data
summary(data)

#Convert numeric variables into categoricals
data$sex<-as.factor(data$sex)
data$cp<-as.factor(data$cp)
data$fbs<-as.factor(data$fbs)
data$restecg<-as.factor(data$restecg)
data$exng<-as.factor(data$exng)
data$oldpeak<-as.factor(data$oldpeak)
data$slp<-as.factor(data$slp)
data$caa<-as.factor(data$caa)
data$thall<-as.factor(data$thall)

#Output will be class variable for the following experiment:convert to factor
data$output<-as.factor(data$output)

#Remove unwanted variables
data2<-data[,-(10:13)]
summary(data2)




#Partition the data into training and testing using the hold out method
#First, we need to set the random seed for repeatability
set.seed(1234)
#Create an index variable to perform a 70/30 split 
trainIndex <- createDataPartition(data2$output, p=.7, list=FALSE, times = 1)
data2_train <- data2[trainIndex,]
data2_test <- data2[-trainIndex,]

#Check the proportion of each origin in training and testing partitions
prop.table(table(data2_train$output)) * 100
prop.table(table(data2_test$output)) * 100

#Inspect the discriptive statistics for each variable in the training and testing partition
summary(data2_train)
summary(data2_test)



#Train K-Nearest Neighbor Classifier with one model (preProcess="scale" normalizes the data)
###########################################################################################
ctrl_none <- trainControl(method="none")
knnFit_none <- train(output ~ ., data = data2_train, method = "knn", trControl = ctrl_none, preProcess = "scale")
knnFit_none

#Now we're ready to predict our test data in order to evaluate the performance of the model
knnPredict_none <- predict(knnFit_none, newdata = data2_test)
knnPredict_none

#Now evaluate the classifier using the confusionMatrix() function
confusionMatrix(knnPredict_none, data2_test$output, mode="everything")



#Train K-Nearest Neighbor Classifier with repeated 10-fold cross validation
###################################################################
ctrl_cv <- trainControl(method="repeatedcv", repeats = 3)
knnFit_cv <- train(output ~ ., data = data2_train, method = "knn", trControl = ctrl_cv, preProcess = "scale")
knnFit_cv

#Visualize the result of the cross-validation
plot(knnFit_cv)

#Now we're ready to predict our test data in order to evaluate the performance of the model
knnPredict_cv <- predict(knnFit_cv, newdata = data2_test)
knnPredict_cv

#Now evaluate the classifier using the confusionMatrix() function
confusionMatrix(knnPredict_cv, data2_test$output, mode="everything")




#Train K-Nearest Neighbor Classifier with .632 bootstrap
#########################################################
ctrl_boot <- trainControl(method="boot632")
knnFit_boot <- train(output ~ ., data = data2_train, method = "knn", trControl = ctrl_boot, preProcess = "scale")
knnFit_boot

#You can also visualize the model with plot
plot(knnFit_boot)

#Now we're ready to predict our test data in order to evaluate the performance of the model
knnPredict_boot <- predict(knnFit_boot, newdata = data2_test)
knnPredict_boot

#Now evaluate the classifier using the confusionMatrix() function
confusionMatrix(knnPredict_boot, data2_test$output, mode="everything")


#Train Decision Tree using the CART algorithm and cross validation
###########################################################################################

ctrl <- trainControl(method="cv")
treeFit <- train(output ~ ., data = data2_train, method = "rpart", trControl = ctrl)
treeFit
summary(treeFit$finalModel)

#Check the attribute importance of the tree. This will show you the attribute 
varImp(treeFit,scale=FALSE)

#Plot a simple representation of the decision tree
plot(treeFit$finalModel, uniform=TRUE)
text(treeFit$finalModel, all=TRUE, cex=.8)

#The rpart.plot library creates a more visually appealing tree (install.packages("rpart.plot")) 
library(rpart.plot)
rpart.plot(treeFit$finalModel)

#Now we're ready to predict our test data in order to evaluate the performance of the model
treePredict <- predict(treeFit, newdata = data2_test)
treePredict

#Now evaluate the classifier using the confusionMatrix() function
confusionMatrix(treePredict, data2_test$output, mode="everything")



#Train Naive Bayes Classifier using cross validation
###########################################################################################
ctrl <- trainControl(method="cv")
nbFit <- train(output ~ ., data = data2_train, method = "nb", trControl = ctrl, preProcess="scale")
nbFit

#Now we're ready to predict our test data in order to evaluate the performance of the model
nbPredict <- predict(nbFit, newdata = data2_test)
nbPredict

#Now evaluate the classifier using the confusionMatrix() function
confusionMatrix(nbPredict, data2_test$output, mode="everything")

#Our classifier isn't performing so well. Let's look at the importance of the variables to see if 
#we can remove variables that don't contribute to the classifier.

varImp(nbFit,scale=FALSE)



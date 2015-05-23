#Practical Mashine Learning. Course Project, Coursera
#Philipp Kats, May 2015

#Setup

library(caret)
library(ggplot2)
library(corrplot)
library(randomForest)
library(rpart)
library(rpart.plot)
dir <- '/Users/casy/Dropbox/My_Projects/Coursera/Coursera_ML/course_project/ML_course_project'
setwd(dir)

#download train data
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}

#now load both sets
trainRaw <- read.csv("./data/pml-training.csv")
testRaw <- read.csv("./data/pml-testing.csv")
dim(trainRaw)
dim(testRaw)

trainRaw$classe <- as.factor(trainRaw$classe)
trainRaw$new_window<- as.factor(trainRaw$new_window)

testRaw$new_window<- as.factor(testRaw$new_window)

#Cleaning Data
sum(complete.cases(trainRaw))

#First, lets get rid of all NA-containing
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0]

#now remove insignificant columns
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]

#Now lets split our training set into pure training and testing sets
set.seed(333) # setting the seed
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]

#Data modelling
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf

#Check model perfomance on validation dataset
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)

table(predictRf, testData$classe)
accuracy <- postResample(predictRf, testData$classe)
accuracy

oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose

#Predicting
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result

#Appendix: Figures
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")

treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) # fast plot

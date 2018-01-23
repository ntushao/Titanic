library(tidyverse)
library(caret)
library(doSNOW)

setwd("~/Downloads")

#Read in and combine the data
Titanic.test <- read.csv("test.csv",stringsAsFactors = FALSE)
Titanic.train <- read.csv("train.csv",stringsAsFactors = FALSE)

View(Titanic.test)
View(Titanic.train)

full <- bind_rows(Titanic.train,Titanic.test)
View(full)

# Feature engineering and fit in the missing values
sapply(full,function(x) sum(is.na(x)))
sapply(full,function(x) sum(x==""))

full$Famsize <- full$SibSp+full$Parch+1


full$Title <- sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
 full$Title <- sub(' ', '', full$Title)
 full$Title[full$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
 full$Title[full$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
 full$Title[full$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
 
 
 
full$agegroup <- ifelse(is.na(full$Age),0,1)

table(full$Embarked)
full$Embarked[full$Embarked==""] <- "S"
table(full$Embarked)

full$Survived <- as.factor(full$Survived)
full$Pclass <- as.factor((full$Pclass))
full$Sex <- as.factor((full$Sex))
full$Embarked <- as.factor((full$Embarked))
full$Title <- as.factor(full$Title)
full <- select(full,Survived,Pclass,Sex,Age,Famsize,Fare,Embarked,Title)




#Fill in age using bagged tree imputation
dummy <- dummyVars(~.,data=full[,-1])
trainDummy <- predict(dummy,full[,-1])
View(trainDummy)

Pre <- preProcess(trainDummy,method="bagImpute")
imputed <- predict(Pre,trainDummy)
View(imputed)

#Categorize the age variable
full$Age <- imputed[,4]
full$age_new[full$Age<=18] <- "Young"
full$age_new[full$Age>18] <- "Old"
full <- select(full,-Age)

#Split the data
Train <- full[1:891,]
Test <- full[892:1309,]
Test %>%
  filter(!is.na(Fare)) %>%
  group_by(Pclass,Embarked) %>%
  summarise(median=median(Fare))
Test$Fare[is.na(Test$Fare)] <- 8.05

#Hyperparameter tuning, data training and prediction using random forest
train.control <- trainControl(method="repeatedcv",number=10,repeats=5,
                            search="grid")
tune <- expand.grid(mtry=1:7)

cl <- makeCluster(2, type = "SOCK")
registerDoSNOW(cl)


training <- train(Survived~.,data=Train,method="rf",tuneGrid=tune,trControl=train.control)
stopCluster(cl)

predicting <- predict(training,Test)
solution <- data.frame(PassengerID = Titanic.test$PassengerId, Survived = predicting)
write.csv(solution,file="luck123.csv",row.names = F)

#Trying to fit a neural network
tune.grid <- expand.grid(size=1:20,decay=5e-4)


training <- train(Survived~.,data=Train,method="nnet",trControl=train.control)
predicting <- predict(training,Test)
solution <- data.frame(PassengerID = Titanic.test$PassengerId, Survived = predicting)
write.csv(solution,file="luck.csv",row.names = F)


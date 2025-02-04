#setting the working dirctory
setwd("C:/Users/Prabhu Adithya/Desktop/Class Documents/Semester 3/Business Analytics/R - Files")

#******************************************************************************************************#

#Clearing the console and the environment  
rm(list=ls())

#******************************************************************************************************#

#Reading the data set
data.df=read.csv("UCI_Credit_Card_Dataset_Sec4_Group8.csv",na.strings = c(""," ","NA"))
head(data.df)
tail(data.df)
str(data.bcup)
#******************************************************************************************************#

#Library:
library(caret) 
library(rpart) 
library(rpart.plot) 
library(MASS) 
library(NeuralNetTools) 
library(ROCR)

#******************************************************************************************************#

#Pre-Procssing of data
summary(data.bcup)
str(data.df)

#Since the dataset has less that 10% of the data as NA's, we are just removing the data
#Totaly 229 data points are removed
data.df=na.omit(data.df)
data.bcup=data.df

#Converting continuos variables in to factor variables 
colsasfact=c("EDUCATION","SEX","MARRIAGE","PAY_0","PAY_2",
             "PAY_3","PAY_4","PAY_5","PAY_6","default.payment.next.month")

data.df[colsasfact]=lapply(data.df[colsasfact], as.factor)

#Columns to be removed
#ID field in not required for prediction
data.df=data.df[,c(-1)]

#Removing levels 0,5 and 6 from the education level, because it is unknown levels
#the number of values is also only 1% of the total (176 rows)

#Educaiton
X=table(data.df$EDUCATION)
X
ind <- with(data.df, which(data.df$EDUCATION==0 | data.df$EDUCATION==5 | data.df$EDUCATION==6, 
                                                   arr.ind=TRUE))
data.df[ind, ]
data.df <- data.df[-ind, ]

#Removing level 0 from mARRAIGE, because it is unknown level
#the number of values is also only 0.5% of the total

#MARRIAGE
Y=table(data.df$MARRIAGE)
Y
ind1 = with(data.df, which(data.df$MARRIAGE== 0, arr.ind=TRUE))
ind1
data.df = data.df[-ind1, ]

#AGE (Removing Outliers in age i.e., greater taht 100)

ind2 = with(data.df, which(data.df$AGE >= 100, arr.ind=TRUE))
ind2
data.df = data.df[-ind2, ]

#Exporting the cleaned file

write.csv(data.df,"C:/Users/Prabhu Adithya/Desktop/Class Documents/Semester 3/Business Analytics/R - Files/credit_card_final.csv")

#******************************************************************************************************#

#set the seed
set.seed(77850)

#Splitting the dataset
trainsplit<- createDataPartition(data.df$default.payment.next.month,p=0.7,list = FALSE)
train.df=data.df[trainsplit,]
test.df=data.df[-trainsplit,]
str(train.df)

#******************************************************************************************************#

#Decision-Tree
default.ct=rpart(default.payment.next.month~.,data=train.df,method = "class")

#plot tree
prp(default.ct, type=5)

#Deep tree
deeper.ct=rpart(default.payment.next.month~., data = train.df, method="class", cp=0, minsplit=1)
deeper.ct

#plot tree
prp(deeper.ct, type = 5)

#Check the accuracy models
default.model=predict(default.ct,train.df,type="class")
default.model

#generate confusion matrix for training data 
confusionMatrix(default.model,train.df$default.payment.next.month)


#Check the accuracy models (Testing data)
default1.model=predict(default.ct,test.df,type="class")

#generate confusion matrix for training data 
confusionMatrix(default1.model,test.df$default.payment.next.month)

#Check the accuracy models (Deeper model - Train data)
deeper.model=predict(deeper.ct,train.df,type="class")


#generate confusion matrix for training data 
confusionMatrix(deeper.model,train.df$default.payment.next.month)

#Check the accuracy models (Deeper model - Testing data)
deeper1.model=predict(deeper.ct,test.df,type="class")

#generate confusion matrix for training data 
confusionMatrix(deeper1.model,test.df$default.payment.next.month)


#Cross-Validation (Training Data)
cv.ct=rpart(default.payment.next.month~., data = train.df, method="class", cp=0.00001, minsplit=5, xval=5)
printcp(cv.ct)
cv.ct
head(train.df)


cv.ct1=rpart(default.payment.next.month~., data = train.df, method="class", cp=0.0012654, minsplit=28, xval=5)
prp(cv.ct1)

#******************************************************************************************************#

#scaling the data (age and salaries are in different scale)
str(data.df)

data.scaled=data.df

data.scaled$Age_scaled<-scale(data.scaled$AGE)
data.scaled$Limit_Bal_scaled<-scale(data.scaled$LIMIT_BAL)
data.scaled$Bill_Amt1_scaled<-scale(data.scaled$BILL_AMT1)
data.scaled$Bill_Amt2_scaled<-scale(data.scaled$BILL_AMT2)
data.scaled$Bill_Amt3_scaled<-scale(data.scaled$BILL_AMT3)
data.scaled$Bill_Amt4_scaled<-scale(data.scaled$BILL_AMT4)
data.scaled$Bill_Amt5_scaled<-scale(data.scaled$BILL_AMT5)
data.scaled$Bill_Amt6_scaled<-scale(data.scaled$BILL_AMT6)
data.scaled$Pay_Amt1_scaled<-scale(data.scaled$PAY_AMT1)
data.scaled$Pay_Amt2_scaled<-scale(data.scaled$PAY_AMT2)
data.scaled$Pay_Amt3_scaled<-scale(data.scaled$PAY_AMT3)
data.scaled$Pay_Amt4_scaled<-scale(data.scaled$PAY_AMT4)
data.scaled$Pay_Amt5_scaled<-scale(data.scaled$PAY_AMT5)
data.scaled$Pay_Amt6_scaled<-scale(data.scaled$PAY_AMT6)

str(data.scaled)

#Remove un-scaled variables

data.scaled = data.scaled[,-c(1,5,12,13,14,15,16,17,18,19,20,21,22,23)]

#******************************************************************************************************#
#set the seed
set.seed(77850)

#Splitting the dataset
trainsplit1<- createDataPartition(data.scaled$default.payment.next.month,p=0.7,list = FALSE)
train1.df=data.scaled[trainsplit1,]
test1.df=data.scaled[-trainsplit1,]
str(train1.df)

#******************************************************************************************************#

#Logistic Regression

training_logit=glm(default.payment.next.month~., data = train1.df, family = binomial(link="logit"))
training_logit

#Model_Diagnostics
##Overall model significance
#Chi-square test result -p value should be less than 0.05

with(training_logit, pchisq(null.deviance - deviance, df.null-df.residual, lower.tail=FALSE))

#Pseudo R2 (%variance explained by the model)

with(training_logit,1-(deviance/null.deviance))

#WALD statistic value is signigicant, Independent variables are significant

summary(training_logit)
varImp(training_logit)
#Step-AIC

m1.step<-stepAIC(training_logit,direction = "both")
m1.step

colnames(train1.df)
str(train1.df)
training_logit1=glm(formula = default.payment.next.month ~ SEX + PAY_2 + 
                      Limit_Bal_scaled + Bill_Amt1_scaled + Bill_Amt2_scaled + Pay_Amt1_scaled + 
                      Pay_Amt6_scaled, family = binomial(link = "logit"), data = train1.df)


#Model_Diagnostics
##Overall model significance
#Chi-square test result -p value should be less than 0.05
with(training_logit1, pchisq(null.deviance - deviance, df.null-df.residual, lower.tail=FALSE))

#Pseudo R2 (%variance explained by the model)
with(training_logit1,1-(deviance/null.deviance))

#WALD statistic value is signigicant, Independent variables are significant
summary(training_logit1)

#Step-AIC 
m2.step<-stepAIC(training_logit1,direction = "both")
m2.step
#There is no change in the number of variables after running this step wise AIC method

#TESTING DATA
test1.df$predicted_val<-predict(training_logit1, newdata=test1.df,type="response")
test1.df$score[test1.df$predicted_val >=0.5]="1"
test1.df$score[test1.df$predicted_val <0.5]="0"
test1.df$score<-as.factor(test1.df$score)
confusionMatrix(test1.df$score,test1.df$default.payment.next.month,positive="0")

#evaluating the performance of the model
pred<-prediction(test1.df$predicted_val,test1.df$default.payment.next.month)
perf <- performance(pred,"tpr","fpr")
plot(perf)
abline(a=0,b=1)


#Create AUC data
aucval<-performance(pred,"auc")
#Calcualte AUC
logistic_auc<-as.numeric(aucval@y.values)
#Display the auc value
logistic_auc

#******************************************************************************************************#

#Neural Network
nnmodel1<-train(default.payment.next.month ~ SEX + MARRIAGE + PAY_0 + PAY_2 + 
                  PAY_3 + PAY_4 + PAY_5 + PAY_6 + Limit_Bal_scaled + Bill_Amt1_scaled + 
                  Bill_Amt2_scaled + Pay_Amt1_scaled + Pay_Amt2_scaled + Pay_Amt6_scaled,train1.df,method="nnet",
                  trControl=trainControl(method = "none"),tuneGrid=expand.grid(.size=c(5),.decay=0.1))

summary(nnmodel1)
plotnet(nnmodel1)

#predict and check model accuracy (Testing data)
nnpredicted<-predict(nnmodel1,test1.df)
cmat<-confusionMatrix(nnpredicted,test1.df$default.payment.next.month,positive = "0")
cmat
nnpredicted

#model performance
nnprob<-predict(nnmodel1,test.df,type="prob")
pred<-prediction(nnprob[,2],test.df$default.payment.next.month)
nnperf<-performance(pred,"tpr","fpr")
nnperf
plot(nnperf)


#AUC
aucval<-performance(pred,"auc")

#calculate AUC
nn_auc<-as.numeric(aucval@y.values)
nn_auc

#including cross-validation to avoid overfitting
TrainingParameters<-trainControl(method = "repeatedcv",number = 10, repeats=10) #10 fold CV repeated 10 times

nnmodel2<-train(default.payment.next.month ~ SEX + MARRIAGE + PAY_0 + PAY_2 + 
                  PAY_3 + PAY_4 + PAY_5 + PAY_6 + Limit_Bal_scaled + Bill_Amt1_scaled + 
                  Bill_Amt2_scaled + Pay_Amt1_scaled + Pay_Amt2_scaled + Pay_Amt6_scaled,train1.df,
                  method="nnet",trControl=TrainingParameters,tuneGrid=expand.grid(.size=c(1,5,10),
                  .decay=c(0,0.001,0.1)))
nnmodel2
nnmodel1
nnmodel2$modelType
nnmodel2$bestTune
nnmodel2$finalModel
plotnet(nnmodel2)

#predict and check model accuracy (Testing data)
nnpredicted<-predict(nnmodel2,test1.df)
cmat<-confusionMatrix(nnpredicted,test1.df$default.payment.next.month,positive = "0")
cmat
nnpredicted

#model performance
nnprob1<-predict(nnmodel2,test1.df,type="prob")
pred1<-prediction(nnprob1[,2],test1.df$default.payment.next.month)
nnperf1<-performance(pred1,"tpr","fpr")
nnperf
plot(nnperf1)
abline(a=0,b=1)

#AUC
aucval1<-performance(pred1,"auc")

#calculate AUC
nn_auc1<-as.numeric(aucval1@y.values)
nn_auc1


#relationship with other models
varImp(nnmodel2)
varImp(default.ct)

#******************************************************************************************************#



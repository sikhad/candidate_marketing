# Find best candidates to market to based on candidate demographics

library(mice) # imputation
library(gdata) # unknown to NA
library(arm) # binned residual plot
library(e1071) # svm
library(boot) # cv

# Data Understanding / Preparation 

# Read in data and verify:

training = read.csv("training.csv")
testing = read.csv("testing.csv")

summary(training)
summary(testing)

# Numerical Data:

par(mfrow=c(3,4))
hist(training$custAge, main="custAge")
hist(training$campaign, main="campaign")
hist(training$pdays, main="pdays")
hist(training$previous, main="previous")
hist(training$emp.var.rate, main="emp.var.rate")
hist(training$cons.price.idx, main="cons.price.idx")
hist(training$cons.conf.idx, main="cons.conf.idx")
hist(training$euribor3m, main="euribor3m")
hist(training$nr.employed, main="nr.employed")
hist(training$pastEmail, main="pastEmail")

# Categorical Data:

table(training$profession) 
table(training$marital) 
table(training$schooling) 
table(training$default)   
table(training$housing)  
table(training$loan) 
table(training$contact)
table(training$month)
table(training$day_of_week)
table(training$poutcome)  

# Convert all the types of missing to just NA:

training$profession = unknownToNA(training$profession, unknown="unknown")
training$marital = unknownToNA(training$marital, unknown="unknown")
training$schooling = unknownToNA(training$schooling, unknown="unknown")
training$default = unknownToNA(training$default, unknown="unknown")
training$housing = unknownToNA(training$housing, unknown="unknown")
training$loan = unknownToNA(training$loan, unknown="unknown")
training$poutcome = unknownToNA(training$poutcome, unknown="nonexistent")
training$pdays = unknownToNA(training$pdays, unknown=999)
training$pmonths = unknownToNA(training$pmonths, unknown=999)

testing$profession = unknownToNA(testing$profession, unknown="unknown")
testing$marital = unknownToNA(testing$marital, unknown="unknown")
testing$schooling = unknownToNA(testing$schooling, unknown="unknown")
testing$default = unknownToNA(testing$default, unknown="unknown")
testing$housing = unknownToNA(testing$housing, unknown="unknown")
testing$loan = unknownToNA(testing$loan, unknown="unknown")
testing$poutcome = unknownToNA(testing$poutcome, unknown="nonexistent")
testing$pdays = unknownToNA(testing$pdays, unknown=999)
testing$pmonths = unknownToNA(testing$pmonths, unknown=999)

# Impute missing values:

trainingImpute = mice(training, m=1, maxit=1, meth='pmm', seed=234)
trainingImputed = complete(trainingImpute)

testingImpute = mice(testing, m=1, maxit=1, meth='pmm', seed=234)
testingImputed = complete(testingImpute)

# Split train into training and validation sets (60-40 split):

set.seed(234)

train_ind = sample(seq_len(nrow(trainingImputed)), size=floor(0.6*nrow(trainingImputed)))

train_split1 = trainingImputed[train_ind,] # actual training
train_split2 = trainingImputed[-train_ind,] # validation

# Remove unnecessary columns: 

train_split1 = train_split1[train_split1$schooling!="illiterate",]
train_split1 = subset(train_split1, select=-c(pmonths))

train_split2 = train_split2[train_split2$schooling!="illiterate",]
train_split2 = subset(train_split2, select=-c(pmonths))

# Modeling

## Part 1

# Run logistic regression on training set and test on the validation set:

FIT1 = glm(responded~custAge+factor(profession)+factor(marital)+
             factor(schooling)+factor(housing)+
             factor(loan)+factor(contact)+factor(month)+factor(day_of_week)+
             campaign+pdays+previous+factor(poutcome)+emp.var.rate+
             cons.price.idx+cons.conf.idx+euribor3m+nr.employed+
             pastEmail, family="binomial", data=train_split1)
summary(FIT1)
binnedplot(predict(FIT1), residuals(FIT1, type="pearson")) # check model assumption

fitted.FIT1 = predict(FIT1,train_split2[1:20],type="response")
fitted.FIT1 = ifelse(fitted.FIT1 > 0.5, "yes", "no")
classifyError = mean(fitted.FIT1 != train_split2$responded)
1-classifyError # accuracy 

# Similarly, try SVM:

FIT2 = svm(responded~custAge+factor(profession)+factor(marital)+
             factor(schooling)+factor(housing)+
             factor(loan)+factor(contact)+factor(month)+factor(day_of_week)+
             campaign+pdays+previous+factor(poutcome)+emp.var.rate+
             cons.price.idx+cons.conf.idx+euribor3m+nr.employed+
             pastEmail, probability=TRUE, data=train_split1)

fitted.FIT2 = predict(FIT2, train_split2[1:20], probability=TRUE)

table(fitted.FIT2, train_split2$responded)
1-((85+229)/(2825+156)) # accuracy from table 

prob.FIT2 = attr(fitted.FIT2,"probabilities")

## Part 2

# Subset those who responded:

respondedProfit = subset(train_split1, train_split1$responded == "yes")

par(mfrow=c(2,2))

# did not include default because all values are 'no'
M1 = lm(profit~custAge+factor(profession)+factor(marital)+
          factor(schooling)+factor(housing)+
          factor(loan)+factor(contact)+factor(month)+factor(day_of_week)+
          campaign+pdays+previous+factor(poutcome)+emp.var.rate+
          cons.price.idx+cons.conf.idx+euribor3m+nr.employed+
          pastEmail, data=respondedProfit)
summary(M1)
plot(M1) # check model assumptions

fitted.M1 = predict(M1, respondedProfit)

plot(respondedProfit$profit, fitted.M1)
abline(0,1, col="red")

# Look at testing file now:
# Multiply probability from the logistic regression model (slightly better than the SVM) with profit from the multiple regression model to find expected profit. Find those candidates whose expected profits are greater than the marketing cost.

testingImputed_noilliterate = testingImputed[testingImputed$schooling!="illiterate",]

predProfit = predict(M1, testingImputed_noilliterate)

predProb = invlogit(predict(FIT1, testingImputed_noilliterate))
expectedProfit = testingImputed_noilliterate[predProfit*predProb > 30,]

# Add column to the testingCandidates files of the ones to target/not target:

finalCandidatesIndex = as.numeric(rownames(expectedProfit))
testing$market = -1
testing[finalCandidatesIndex,]$market = 1
testing[-finalCandidatesIndex,]$market = 0

# Look at statistics of the ones to target:

toTarget = testing[finalCandidatesIndex,]

par(mfrow=c(3,4))
hist(toTarget$custAge, main="custAge")
hist(toTarget$campaign, main="campaign")
hist(toTarget$pdays, main="pdays")
hist(toTarget$previous, main="previous")
hist(toTarget$emp.var.rate, main="emp.var.rate")
hist(toTarget$cons.price.idx, main="cons.price.idx")
hist(toTarget$cons.conf.idx, main="cons.conf.idx")
hist(toTarget$euribor3m, main="euribor3m")
hist(toTarget$nr.employed, main="nr.employed")
hist(toTarget$pastEmail, main="pastEmail")

notToTarget = testing[-finalCandidatesIndex,]

par(mfrow=c(3,4))
hist(notToTarget$custAge, main="custAge")
hist(notToTarget$campaign, main="campaign")
hist(notToTarget$pdays, main="pdays")
hist(notToTarget$previous, main="previous")
hist(notToTarget$emp.var.rate, main="emp.var.rate")
hist(notToTarget$cons.price.idx, main="cons.price.idx")
hist(notToTarget$cons.conf.idx, main="cons.conf.idx")
hist(notToTarget$euribor3m, main="euribor3m")
hist(notToTarget$nr.employed, main="nr.employed")
hist(notToTarget$pastEmail, main="pastEmail")

# Output to csv:

write.csv(testing, file="testing.csv")
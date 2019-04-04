# Title     : Movie review classifier
# Objective : Classify movei reviews in postive and negative
# Created by: tarikvandenberg
# Created on: 04/04/2019

#Load packages
library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)
library(data.table)
library(doMC)
registerDoMC(cores=detectCores())
#Set this path to your working directory
setwd("C:/Users/Tarik van den Berg/Documents/finalstretch")

#Getting rid of magic numbers
#Set the amount of data you would like to import from each dataset
rowsDfOne <- as.numeric(4000)
rowsDftwo <- as.numeric(2000)

#Should be the first 2/3
trainAmountStart <- as.numeric(1)
trainAmountEnd <-as.numeric(4000)

#Should be the last 1/3
TestAmountStart <- as.numeric(4001)
TestAmountEnd <- as.numeric(6000)

#Import the first DF
dfOne <- fread("labeledTrainData.tsv", stringsAsFactors = F, nrows = rowsDfOne)

#Remove the row that containts ID
dfOne$id <- NULL

#Read the second DF
dfTwo <- read.csv("movie-pang02.csv", stringsAsFactors = F, header = T, sep= ",", nrows = rowsDftwo)

#Add the column names for rbind
colnames(dfTwo) <- c("sentiment", "review")

#Change the Pos and Neg in the sentiment column to 1 and 0 to match the other DF
dfTwo$sentiment[dfTwo$sentiment=="Pos"] <- "1"
dfTwo$sentiment[dfTwo$sentiment=="Neg"] <- "0"

#Combine the two dataframes
df <- rbind(dfOne, dfTwo)

#Set seed and randomize our data
set.seed(1)
df <- df[sample(nrow(df)), ]
df <- df[sample(nrow(df)), ]

#Make sure that sentiment are vectors
df$sentiment <- as.factor(df$sentiment)

#Inspect the corpus
corpus <- Corpus(VectorSource(df$review))
corpus
inspect(corpus[1:3])

#Clean the texts of the following: Whitespaces, Punctuations, numbers, stopword and make it all Lower case
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

#Create a DocumentTermMatrix where we plug the Reviews into
dtm <- DocumentTermMatrix(corpus.clean)

inspect(dtm[40:50, 10:15])

#Split the data into different amounts for the testing purposes
df.train <- df[trainAmountStart:trainAmountEnd,]
df.test <- df[TestAmountStart:TestAmountEnd,]
dtm.train <- dtm[trainAmountStart:trainAmountEnd,]
dtm.test <- dtm[TestAmountStart:TestAmountEnd,]
corpus.clean.train <- corpus.clean[trainAmountStart:trainAmountEnd]
corpus.clean.test <- corpus.clean[TestAmountStart:TestAmountEnd]


#Restrict the words were gonna use to words that are used more than 5 times in all the reviews combined
fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))


#Set the presence of a the word to yes or no
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

#Apply our conversion
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

#Train the classifier
system.time( classifier <- naiveBayes(trainNB, df.train$sentiment, laplace = 1) )

#Create predictions using our classifier
system.time( pred <- predict(classifier, newdata=testNB) )

#Create the truth table
table("Predictions"= pred,  "Actual" = df.test$sentiment )

#Build confusion matrix
conf.mat <- confusionMatrix(pred, df.test$sentiment)

#Print how well we've done (should be above 75 :-) )
conf.mat$overall['Accuracy']

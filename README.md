# Prediction Loan repayments by Classification Models

#Supervised Learning: 
Supervised learning is the most common subbranch of machine learning today. Typically, new machine learning practitioners will begin their journey with supervised learning algorithms.

#How supervised learning works:
Supervised machine learning algorithms are designed to learn by example. The name “supervised” learning originates from the idea that training this type of algorithm is like having a teacher supervise the whole process.
When training a supervised learning algorithm, the training data will consist of inputs paired with the correct outputs. During training, the algorithm will search for patterns in the data that correlate with the desired outputs. After training, a supervised learning algorithm will take in new unseen inputs and will determine which label the new inputs will be classified as based on prior training data. The objective of a supervised learning model is to predict the correct label for newly presented input data. At its most basic form, a supervised learning algorithm can be written simply as:
Y=F(X)
 
Where Y is the predicted output that is determined by a mapping function that assigns a class to an input value x. The function used to connect input features to a predicted output is created by the machine learning model during training.
Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized.
Supervised learning can be separated into two types of problems when data mining classification and regression

#Classification
Classification uses an algorithm to accurately assign test data into specific categories. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined. Common classification algorithms are linear classifiers, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.


This repository contains Python implementation of the supervised learning algorithms devised in the Supervised Learning Algorithms for Predicting Loan repayments.

For this project we will be exploring publicly available data from LendingClub.com. Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
Lending club had a very interesting year in 2016, so let's check out some of their data and keep the context in mind. This data is from before they even went public.
We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from here or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
Here are what the columns represent:
•	credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
•	purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
•	int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
•	installment: The monthly installments owed by the borrower if the loan is funded.
•	log.annual.inc: The natural log of the self-reported annual income of the borrower.
•	dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
•	fico: The FICO credit score of the borrower.
•	days.with.cr.line: The number of days the borrower has had a credit line.
•	revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
•	revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
•	inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
•	delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
•	pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

 
![](images.jpg)

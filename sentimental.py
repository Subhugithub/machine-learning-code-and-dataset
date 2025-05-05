import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load dataset
msg = pd.read_csv(r"C:\Users\Shubham\Desktop\dicision_tree\sentiment_dataset.csv")

# Print dataset dimensions
print('The dimensions of the dataset', msg.shape)

# Map labels to numeric
msg['labelnum'] = msg['Label'].map({'pos': 1, 'neg': 0})

# Features and target
X = msg['Text']
y = msg['labelnum']

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(X, y)
print('\nThe total number of Training Data:', ytrain.shape)
print('\nThe total number of Test Data:', ytest.shape)

# Convert text to feature vectors
cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm = cv.transform(xtest)

# Show tokens
print('\nThe words or Tokens in the text documents:\n')
print(cv.get_feature_names_out())

# Convert training data to DataFrame (optional, for inspection)
df = pd.DataFrame(xtrain_dtm.toarray(), columns=cv.get_feature_names_out())

# Train Naive Bayes classifier
clf = MultinomialNB().fit(xtrain_dtm, ytrain)

# Predict on test data
predicted = clf.predict(xtest_dtm)

# Evaluation metrics
print('\nAccuracy of the classifier is:', metrics.accuracy_score(ytest, predicted))
print('\nConfusion matrix:')
print(metrics.confusion_matrix(ytest, predicted))
print('\nPrecision:', metrics.precision_score(ytest, predicted))
print('\nRecall:', metrics.recall_score(ytest, predicted))

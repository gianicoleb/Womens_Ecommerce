import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# read the data from CSV file
data = pd.read_csv("womensclothing.csv")

# drop the columns that are not required for sentiment analysis
data.drop(['Clothing ID', 'Age', 'Title', 'Positive Feedback Count', 'Division Name', 'Department Name', 'Class Name'], axis=1, inplace=True)

# drop rows with missing values
data.dropna(inplace=True)

# create a binary sentiment column based on the ratings
data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x > 3 else 0)

# create a CountVectorizer to convert text to a matrix of token counts
vectorizer = CountVectorizer(stop_words='english')

# transform the text to a matrix of token counts
X = vectorizer.fit_transform(data['Review Text'])

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, data['Sentiment'], test_size=0.2, random_state=42)

# train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# predict the likelihood of good reviews for clothing items
new_text = ["This dress is amazing, I love it!"]
new_X = vectorizer.transform(new_text)
new_pred = clf.predict(new_X)
print("Prediction:", new_pred)

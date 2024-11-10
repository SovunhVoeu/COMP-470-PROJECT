from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset (two categories)
categories = ['comp.os.ms-windows.misc', 'rec.autos']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

vectorizer.get_feature_names_out()
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)
print(clf.predict(X))

#This is the part I could not get running
print(classification_report(X_train, X_test, y_train, y_test))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

real['label'] = 0
fake['label'] = 1

data = pd.concat([real, fake], axis=0).sample(frac=1).reset_index(drop=True)
data = data[['title', 'text', 'label']]
data['content'] = data['title'] + " " + data['text']


X_train, X_test, y_train, y_test = train_test_split(
    data['content'], data['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

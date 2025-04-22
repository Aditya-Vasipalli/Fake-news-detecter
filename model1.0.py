import pandas as pd

real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

real['label'] = 0
fake['label'] = 1

data = pd.concat([real, fake], axis=0).sample(frac=1).reset_index(drop=True)
data = data[['title', 'text', 'label']]
data['content'] = data['title'] + " " + data['text']

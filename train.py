from preprocess import read_data, create_dictionary, transform_text
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

train_messages, train_labels = read_data('UIT-VSFC/train/sents.txt', 'UIT-VSFC/train/sentiments.txt')
dictionary = create_dictionary(train_messages)
train_data = transform_text(train_messages, dictionary)

model = MultinomialNB()
model.fit(train_data, train_labels)

test_messages, test_labels = read_data('UIT-VSFC/test/sents.txt', 'UIT-VSFC/test/sentiments.txt')
test_data = transform_text(test_messages, dictionary)

y_pred = model.predict(test_data)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(test_labels, y_pred)*100)
cf_matrix = metrics.confusion_matrix(test_labels, y_pred)
lb = np.sum(cf_matrix, axis=1, dtype=float)
lb = lb.reshape(-1, 1)

sns.heatmap(cf_matrix/lb*100, annot=True, cmap='Blues', fmt=".2f")
plt.show()

import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import logging

# LOG = logging.Logger(__name__)

x = pd.read_csv('X_hyperplane.txt')
y = pd.read_csv('y_hyperplane.txt')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

model = SVC(verbose=True)
y_train = y_train.values
model.fit(x_train, y_train)
mean_acc = model.score(x_test, y_test)

y_predicted = model.predict(x_test)

# Get Metrics
precision = precision_score(y_test, y_predicted)
f1 = f1_score(y_test, y_predicted)
auc = roc_auc_score(y_test, y_predicted)
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)

sns.set(style="darkgrid")
data = np.vstack([fpr, tpr])
data = pd.DataFrame(data.T, columns=['fpr', 'tpr'])

plt.figure()
sns.lineplot(data=data, x = 'fpr', y='tpr')
plt.title('ROC Curve')
plt.show()

print(f'Precision: {precision:0.4f}\n'
      f'F1 Score: {f1:0.4f}\n'
      f'AUC: {auc:0.4f}')
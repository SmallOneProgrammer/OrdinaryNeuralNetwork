from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from main import ONN
import numpy as np



#testing classifier
data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

model = ONN(X, y, [130, 130])

model.fit()

#comparison with MLPClassifier
sklearnModel = MLPClassifier()
sklearnModel.fit(X, y)
print("MLP Classifier: ", sklearnModel.score(X, y))
cost, predictions = model.predict()

#round predictions
rounding = lambda x: np.round(x)
pred = rounding(predictions)

#check accuracy
print('accuracy score is: ', np.mean(pred == y))
print(y.shape)
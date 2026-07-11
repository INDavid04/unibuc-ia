print("In progess...")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1042)

with open('../train_samples.txt', 'r', encoding='utf-8') as f:
    train_text = f.readlines()
train_coords = np.load('../train_coordinates.npy')

with open('../test_samples.txt', 'r', encoding='utf-8') as f:
    test_text = f.readlines()

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(train_coords)

X_train, X_val, y_train, y_val = train_test_split(X, y_scaled, test_size=0.1, random_state=1042)

model = MLPRegressor(
    hidden_layer_sizes=(128, 128, 128),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=1042
)

model.fit(X_train, y_train)

val_preds_scaled = model.predict(X_val)
val_preds = scaler.inverse_transform(val_preds_scaled)
y_val_orig = scaler.inverse_transform(y_val)

mse = mean_squared_error(y_val_orig, val_preds)

final_preds_scaled = model.predict(X_test)
final_predictions = scaler.inverse_transform(final_preds_scaled)
np.save('Irimia_David_subiect1_solutia1.npy', final_predictions)

print(f"mse: {mse}")

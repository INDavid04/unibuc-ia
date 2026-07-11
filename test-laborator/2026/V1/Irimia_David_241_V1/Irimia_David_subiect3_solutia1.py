print("In progress...")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

with open('Irimia_David_subiect2_solutia1.txt', 'r', encoding='utf-8') as f:
    custom_vocab = [line.strip() for line in f.readlines()]

with open('../train_samples.txt', 'r', encoding='utf-8') as f:
    train_text = f.readlines()
train_coords = np.load('../train_coordinates.npy')

with open('../test_samples.txt', 'r', encoding='utf-8') as f:
    test_text = f.readlines()

vectorizer = TfidfVectorizer(vocabulary=custom_vocab)
X = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(train_coords)

X_train, X_val, y_train, y_val = train_test_split(X, y_scaled, test_size=0.1, random_state=1042)

model = KernelRidge(kernel='rbf', gamma=0.1, alpha=1.0)
model.fit(X_train, y_train)

val_preds_scaled = model.predict(X_val)
val_preds = scaler.inverse_transform(val_preds_scaled)
y_val_orig = scaler.inverse_transform(y_val)
mse = mean_squared_error(y_val_orig, val_preds)

final_preds_scaled = model.predict(X_test)
final_predictions = scaler.inverse_transform(final_preds_scaled)

np.save('Irimia_David_subiect3_solutia1.npy', final_predictions)

print(f"MSE pe validare: {mse}")

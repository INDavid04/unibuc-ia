print("In progress...")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
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
X = vectorizer.fit_transform(train_text).toarray()
X_test = vectorizer.transform(test_text).toarray()

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(train_coords)

def intersection_kernel(X1, X2, chunk_size=1000):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(0, n1, chunk_size):
        end_i = min(i + chunk_size, n1)
        for j in range(0, n2, chunk_size):
            end_j = min(j + chunk_size, n2)
            block = np.minimum(X1[i:end_i, np.newaxis, :], X2[np.newaxis, j:end_j, :])
            K[i:end_i, j:end_j] = np.sum(block, axis=2)
    return K

X_train, X_val, y_train, y_val = train_test_split(X, y_scaled, test_size=0.1, random_state=1042)

K_train_train = intersection_kernel(X_train, X_train)
K_val_train = intersection_kernel(X_val, X_train)
K_test_train = intersection_kernel(X_test, X_train)

model_lat = SVR(kernel='precomputed')
model_lon = SVR(kernel='precomputed')

model_lat.fit(K_train_train, y_train[:, 0])
model_lon.fit(K_train_train, y_train[:, 1])

val_pred_lat = model_lat.predict(K_val_train)
val_pred_lon = model_lon.predict(K_val_train)
val_preds_scaled = np.stack([val_pred_lat, val_pred_lon], axis=1)

val_preds = scaler.inverse_transform(val_preds_scaled)
y_val_orig = scaler.inverse_transform(y_val)

mse = mean_squared_error(y_val_orig, val_preds)

final_pred_lat = model_lat.predict(K_test_train)
final_pred_lon = model_lon.predict(K_test_train)
final_preds_scaled = np.stack([final_pred_lat, final_pred_lon], axis=1)

final_predictions = scaler.inverse_transform(final_preds_scaled)
np.save('Irimia_David_subiect4_solutia1.npy', final_predictions)

print(f"MSE: {mse}")

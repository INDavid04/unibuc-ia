print("In progess...")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def build_vocabulary(texts, coords):
    coords[:, 1] /= 2
    
    min_lat, min_lon = coords.min(axis=0)
    max_lat, max_lon = coords.max(axis=0)
    
    lat_bins = np.linspace(min_lat, max_lat, 5)
    lon_bins = np.linspace(min_lon, max_lon, 41)
    
    region_texts = [[] for _ in range(4 * 40)]
    
    for i in range(len(texts)):
        lat_idx = np.searchsorted(lat_bins[1:-1], coords[i, 0])
        lon_idx = np.searchsorted(lon_bins[1:-1], coords[i, 1])
        region_texts[lat_idx * 40 + lon_idx].append(texts[i])
        
    super_docs = [" ".join(docs) for docs in region_texts if docs]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(super_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    vocab = set()
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(i).toarray().flatten()
        top_indices = row.argsort()[::-1][:1500]
        for idx in top_indices:
            if row[idx] > 0:
                vocab.add(feature_names[idx])
                
    sorted_vocab = sorted(list(vocab), key=lambda word: sum(doc.count(word) for doc in super_docs))
    return sorted_vocab

with open('../train_samples.txt', 'r', encoding='utf-8') as f:
    train_text = f.readlines()
train_coords = np.load('../train_coordinates.npy')

vocabulary = build_vocabulary(train_text, train_coords)

with open('Irimia_David_subiect2_solutia1.txt', 'w', encoding='utf-8') as f:
    for word in vocabulary:
        f.write(f"{word}\n")

print("Vezi Irimia_David_subiect2_solutia1.txt")

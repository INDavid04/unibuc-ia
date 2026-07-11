"""
Nume: Irimia David | Grupa: 241 | Colocviu IA, 6 iunie 2026, Varianta 1

Subiectul 2 (2.5p): Constructie vocabular TF-ISF (Term Frequency - Inverse
Spatial Frequency) pentru datele geo-etichetate.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tf_isf_vocabulary(texts, coords, n_lat_bins=4, n_lon_bins=40,
                             top_k=1500, analyzer='word', ngram_range=(1, 1)):
    """
    Construieste vocabularul TF-ISF descris in cerinta:

    1. Se determina limitele extreme ale spatiului geografic (lat/lon) al
       datelor de antrenare si se construieste o grila de n_lat_bins x
       n_lon_bins subregiuni de aceeasi suprafata.
    2. Fiecare text este atribuit subregiunii in care cade punctul lui.
    3. Textele dintr-o subregiune sunt concatenate intr-un "super-document".
    4. Se aplica TF-IDF la nivelul intregii harti (super-documentele
       formeaza corpusul): termenii comuni peste tot sunt penalizati, cei
       specifici unei zone primesc scoruri mari.
    5. Pentru fiecare subregiune se pastreaza cele mai importante top_k
       cuvinte (sau toate, daca subregiunea are mai putine).
    6. Vocabularul final = reuniunea (fara duplicate) a vocabularelor
       regionale.

    Parametri
    ---------
    texts : list[str] - textele de antrenare
    coords : array (N, 2) - (latitudine, longitudine) pentru fiecare text
    n_lat_bins, n_lon_bins : dimensiunile grilei (implicit 4 x 40 = 160 subregiuni)
    top_k : cate cuvinte se pastreaza per subregiune (implicit 1500)

    Returns
    -------
    vocab : list[str] - vocabularul final (cuvinte unice, sortate)
    grid_info : dict - limitele grilei folosite (utile pentru reproducere)
    """

    texts = list(texts)
    coords = np.asarray(coords, dtype=np.float64)
    lat, lon = coords[:, 0], coords[:, 1]

    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()

    lat_edges = np.linspace(lat_min, lat_max, n_lat_bins + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_lon_bins + 1)

    lat_idx = np.clip(np.digitize(lat, lat_edges[1:-1], right=False), 0, n_lat_bins - 1)
    lon_idx = np.clip(np.digitize(lon, lon_edges[1:-1], right=False), 0, n_lon_bins - 1)

    region_id = lat_idx * n_lon_bins + lon_idx
    n_regions = n_lat_bins * n_lon_bins

    super_docs = ["" for _ in range(n_regions)]
    region_has_text = np.zeros(n_regions, dtype=bool)
    for txt, rid in zip(texts, region_id):
        if super_docs[rid]:
            super_docs[rid] += " " + txt
        else:
            super_docs[rid] = txt
        region_has_text[rid] = True

    nonempty_idx = np.where(region_has_text)[0]
    nonempty_docs = [super_docs[i] for i in nonempty_idx]

    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                  lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    tfidf_matrix = vectorizer.fit_transform(nonempty_docs)
    feature_names = np.array(vectorizer.get_feature_names_out())

    final_vocab = set()
    for row in range(tfidf_matrix.shape[0]):
        row_vec = tfidf_matrix.getrow(row).toarray().ravel()
        nnz = np.nonzero(row_vec)[0]
        if len(nnz) == 0:
            continue
        k = min(top_k, len(nnz))
        top_idx = nnz[np.argsort(row_vec[nnz])[::-1][:k]]
        final_vocab.update(feature_names[top_idx].tolist())

    vocab = sorted(final_vocab)
    grid_info = dict(n_lat_bins=n_lat_bins, n_lon_bins=n_lon_bins,
                      lat_edges=lat_edges, lon_edges=lon_edges)
    return vocab, grid_info


if __name__ == "__main__":
    train_samples = open("../train_samples.txt", encoding="utf-8").read().splitlines()
    train_coords = np.load("../train_coordinates.npy")
    vocab, grid_info = build_tf_isf_vocabulary(train_samples, train_coords,
                                                n_lat_bins=4, n_lon_bins=40, top_k=1500)
    print("Dimensiune vocabular final:", len(vocab))
    print("Exemple cuvinte:", vocab[:20])

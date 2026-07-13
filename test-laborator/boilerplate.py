"""
BOILERPLATE — Colocviu/Test de laborator IA (Universitatea din Bucuresti)
Bazat pe subiectele rezolvate (2025: clasificare text/Naive Bayes; 2026:
regresie geo/MLP/Kernel Ridge/SVM cu kernel-uri custom) SI pe restul
materiei din curs (KNN pe MNIST, perceptron/Widrow-Hoff, clustering,
algoritmi genetici, cautare informata A*/IDA*) -- ca sa acopere cat mai
multe variante posibile ale unui viitor test de laborator.

Cum se foloseste: copiaza blocurile de care ai nevoie in fisierul final
{Nume}_{Prenume}_{Grupa}_subiect{i}_solutia_{j}.py -- fiecare submisie
trebuie sa fie un SINGUR fisier .py, autonom (fara importuri din alte
fisiere proprii). Acest fisier e doar REFERINTA -- majoritatea blocurilor
sunt comentate intentionat (nu se executa nimic daca il rulezi ca atare).

=============================================================================
CHECKLIST INAINTE SA INCEPI (citeste enuntul de 2 ori!)
=============================================================================
1. Noteaza EXACT constrangerile de arhitectura/hiperparametri (numar de
   straturi, numar de neuroni, functie de activare, gamma/C/alpha fixati).
   Daca cerinta zice "gamma=777" sau "cel mult 2 straturi", respecta LITERAL
   -- chiar daca rezultatul pare ciudat la prima vedere (vezi sectiunea
   "capcane" mai jos). Nerespectarea structurii => de multe ori 0 puncte,
   chiar daca MSE-ul e bun.
2. Verifica formatul exact de nume fisier/folder cerut in README
   (Nume_Prenume_Grupa_subiectN_solutia_M.py/.npy) si respecta-l STRICT.
3. Verifica daca fiecare submisie trebuie sa fie intr-un singur fisier .py
   -- daca da, nu importa din module proprii (inline tot codul).
4. Verifica pragurile de punctaj (de obicei MSE/acuratete <= prag_X => Y
   puncte) -- tine minte sa optimizezi spre cel mai bun prag, nu doar sa
   "mearga".
5. La final, ruleaza scriptul intr-un folder curat (doar cu datele + acel
   .py) ca sa te asiguri ca merge de sine statator.

=============================================================================
0. INCARCARE DATE (patternurile intalnite pana acum -- exemple, comentate)
=============================================================================
"""
import numpy as np
import random

# --- text + coordonate (regresie geo, tip 2026) ---
# with open("train_samples.txt", encoding="utf-8") as f:
#     train_samples = f.read().splitlines()
# train_coords = np.load("train_coordinates.npy")   # shape (N, 2) -> (lat, lon)
# with open("test_samples.txt", encoding="utf-8") as f:
#     test_samples = f.read().splitlines()

# --- text + etichete (clasificare text, tip 2025) ---
# with open("train_sentences.txt", encoding="utf-8") as f:
#     train_sentences = np.array([l.strip() for l in f if l.strip()])
# train_labels = np.load("train_labels.npy", allow_pickle=True)
# with open("test_sentences.txt", encoding="utf-8") as f:
#     test_sentences = np.array([l.strip() for l in f if l.strip()])
# # optional: mapping.txt / words.txt -- vocabular sau mapare eticheta<->nume

"""
=============================================================================
1. SPLIT INTERN DE VALIDARE (mereu util pentru raport / alegere hiperparametri)
=============================================================================
"""
from sklearn.model_selection import train_test_split
# X_tr, X_val, y_tr, y_val = train_test_split(train_samples, train_coords,
#                                              test_size=0.15, random_state=42)
# --> alege hiperparametrii pe (X_val, y_val), apoi REANTRENEAZA pe 100% din
#     date pentru submisia finala (.npy trimis).

"""
=============================================================================
2. VECTORIZARE TEXT
=============================================================================
"""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# --- TF-IDF generic (char n-grams merg surprinzator de bine pt limbaj colocvial/emoji) ---
# tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=10000,
#                          min_df=2, sublinear_tf=True)
# word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000,
#                               min_df=2, sublinear_tf=True)
# from scipy.sparse import hstack  -- combina cele doua feature seturi:
# from scipy.sparse import hstack
# X = hstack([tfidf.fit_transform(texts), word_tfidf.fit_transform(texts)]).tocsr()

# --- reducere dimensionalitate (utila pt retele custom / distante euclidiene) ---
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler
# svd = TruncatedSVD(n_components=600, random_state=42)
# X_reduced = svd.fit_transform(X)
# X_scaled = StandardScaler().fit_transform(X_reduced)

# --- vocabular custom (fixat de un exercitiu anterior) ---
# tfidf = TfidfVectorizer(vocabulary=vocab_dat, lowercase=True,
#                          token_pattern=r"(?u)\b\w\w+\b",
#                          sublinear_tf=True, smooth_idf=True, norm='l1')  # sau 'l2'
# ! norm='l1' functioneaza mult mai bine decat 'l2' pt kernel-uri liniare in
#   valorile vectorilor (ex: kernel de intersectie). 'l2' e ok pt RBF.

"""
=============================================================================
3. VOCABULAR TIP "TF-ISF" (Term Frequency - Inverse SPATIAL/GROUP Frequency)
   Util cand ai nevoie de vocabular specific unor regiuni/grupuri (geo,
   clase, etc.) in loc de vocabular global. Sunt functii reale, gata de
   folosit (nu comentate).
=============================================================================
"""


def build_group_specific_vocabulary(texts, group_ids, n_groups, top_k=1500):
    """
    1. Grupeaza textele dupa group_ids (ex: subregiuni geo, clase).
    2. Concateneaza textele fiecarui grup intr-un "super-document".
    3. TF-IDF la nivel de super-documente (corpus = super-documentele).
    4. Pastreaza top_k cuvinte per grup dupa scor TF-IDF.
    5. Reuneste (fara duplicate) vocabularele -> vocabular final.
    """
    super_docs = ["" for _ in range(n_groups)]
    has_text = np.zeros(n_groups, dtype=bool)
    for txt, gid in zip(texts, group_ids):
        super_docs[gid] = (super_docs[gid] + " " + txt) if super_docs[gid] else txt
        has_text[gid] = True
    nonempty_docs = [super_docs[i] for i in np.where(has_text)[0]]

    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    tfidf_matrix = vectorizer.fit_transform(nonempty_docs)
    feature_names = np.array(vectorizer.get_feature_names_out())

    vocab = set()
    for row in range(tfidf_matrix.shape[0]):
        row_vec = tfidf_matrix.getrow(row).toarray().ravel()
        nnz = np.nonzero(row_vec)[0]
        if len(nnz) == 0:
            continue
        k = min(top_k, len(nnz))
        top_idx = nnz[np.argsort(row_vec[nnz])[::-1][:k]]
        vocab.update(feature_names[top_idx].tolist())
    return sorted(vocab)


def geo_grid_group_ids(coords, n_lat_bins, n_lon_bins):
    """Imparte coordonate (lat, lon) intr-o grila n_lat_bins x n_lon_bins de
    subregiuni de aceeasi suprafata (utila pt build_group_specific_vocabulary)."""
    lat, lon = coords[:, 0], coords[:, 1]
    lat_edges = np.linspace(lat.min(), lat.max(), n_lat_bins + 1)
    lon_edges = np.linspace(lon.min(), lon.max(), n_lon_bins + 1)
    lat_idx = np.clip(np.digitize(lat, lat_edges[1:-1]), 0, n_lat_bins - 1)
    lon_idx = np.clip(np.digitize(lon, lon_edges[1:-1]), 0, n_lon_bins - 1)
    return lat_idx * n_lon_bins + lon_idx


"""
=============================================================================
4. MODELE UZUALE (exemple, comentate)
=============================================================================
"""
# --- clasificare text simpla (tip 2025) ---
# from sklearn.naive_bayes import MultinomialNB
# model = MultinomialNB()
# model.fit(X_train_counts, y_train)
# preds = model.predict(X_test_counts)

# --- regresie liniara/kernel ---
# from sklearn.kernel_ridge import KernelRidge
# model = KernelRidge(kernel='rbf', gamma=..., alpha=...)  # alpha = regularizare

# --- SVM/SVR, inclusiv kernel precomputat custom ---
# from sklearn.svm import SVC, SVR
# svr = SVR(kernel='precomputed', C=..., epsilon=...)
# svr.fit(K_train, y_train); pred = svr.predict(K_test_train)
# NOTA: SVR e single-output -> pentru tinte multi-dimensionale (ex: lat+lon),
# antreneaza cate un model separat per dimensiune.

# --- MLP standard (daca activarea ceruta e in {identity, logistic, tanh, relu}) ---
# from sklearn.neural_network import MLPRegressor, MLPClassifier
# model = MLPRegressor(hidden_layer_sizes=(128, 128), activation='relu', ...)
# ATENTIE: verifica activarea si nr de straturi cerute! sklearn NU are LeakyReLU.

"""
=============================================================================
5. RETEA NEURALA CUSTOM (cand activarea ceruta nu exista in sklearn, ex. LReLU)
   Clasa reala, gata de folosit -- adapteaz-o dupa nr straturi/neuroni cerute!
=============================================================================
"""


class LReLUMLPRegressor:
    """Retea feed-forward numpy, activare LeakyReLU, Adam, early stopping.
    Verifica in enunt nr maxim de straturi ascunse si de neuroni/strat si
    seteaza hidden_layer_sizes in consecinta (ex: (128, 128) pt 2 straturi
    de cate 128 neuroni, maxim permis intr-un caz uzual)."""

    def __init__(self, hidden_layer_sizes=(128, 128), alpha_leak=0.03,
                 l2=6e-5, lr=1.5e-3, epochs=200, batch_size=48,
                 random_state=42, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha_leak = alpha_leak
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def _leaky_relu(self, x):
        return np.where(x > 0, x, self.alpha_leak * x)

    def _leaky_relu_grad(self, x):
        return np.where(x > 0, 1.0, self.alpha_leak)

    def _init_params(self, n_in, n_out):
        rng = np.random.RandomState(self.random_state)
        sizes = [n_in] + list(self.hidden_layer_sizes) + [n_out]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            limit = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-limit, limit, size=(sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]

    def _forward(self, X):
        activations, zs, a = [X], [], X
        n_layers = len(self.W)
        for i in range(n_layers):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            a = self._leaky_relu(z) if i < n_layers - 1 else z
            activations.append(a)
        return activations, zs

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_in = X.shape
        n_out = y.shape[1]
        self._init_params(n_in, n_out)
        rng = np.random.RandomState(self.random_state)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        n_layers = len(self.W)
        best_val, best_state, patience, bad_epochs = np.inf, None, 15, 0

        for epoch in range(self.epochs):
            idx = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                bidx = idx[start:start + self.batch_size]
                Xb, yb = X[bidx], y[bidx]
                m = Xb.shape[0]
                activations, zs = self._forward(Xb)
                dA = (activations[-1] - yb) * (2.0 / m)
                gradsW, gradsb = [None] * n_layers, [None] * n_layers
                for i in reversed(range(n_layers)):
                    dZ = dA if i == n_layers - 1 else dA * self._leaky_relu_grad(zs[i])
                    gradsW[i] = activations[i].T @ dZ + self.l2 * self.W[i]
                    gradsb[i] = dZ.sum(axis=0)
                    dA = dZ @ self.W[i].T
                t += 1
                for i in range(n_layers):
                    self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gradsW[i]
                    self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (gradsW[i] ** 2)
                    self.W[i] -= self.lr * (self.mW[i] / (1 - beta1 ** t)) / (np.sqrt(self.vW[i] / (1 - beta2 ** t)) + eps)
                    self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gradsb[i]
                    self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (gradsb[i] ** 2)
                    self.b[i] -= self.lr * (self.mb[i] / (1 - beta1 ** t)) / (np.sqrt(self.vb[i] / (1 - beta2 ** t)) + eps)
            if X_val is not None:
                val_mse = np.mean((self.predict(X_val) - y_val) ** 2)
                if val_mse < best_val - 1e-6:
                    best_val = val_mse
                    bad_epochs = 0
                    best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                else:
                    bad_epochs += 1
                if self.verbose and epoch % 20 == 0:
                    print(f"epoch {epoch} val_mse {val_mse:.4f}")
                if bad_epochs >= patience:
                    break
        if best_state is not None:
            self.W, self.b = best_state
        return self

    def predict(self, X):
        activations, _ = self._forward(np.asarray(X, dtype=np.float64))
        return activations[-1]


"""
=============================================================================
6. KERNEL-URI CUSTOM (RBF cu gamma fixat, kernel de intersectie, etc.)
   Functie reala, gata de folosit.
=============================================================================
"""
from scipy.sparse import csc_matrix


def intersection_kernel_matrix(X, Y=None, gamma=1.0):
    """K(x,y) = gamma * sum_k min(x_k, y_k), calculat EFICIENT pe coloane
    (nu O(n1*n2*d) brut -- aia iti da MemoryError garantat pt vocabulare
    mari!). Functioneaza foarte bine cand X e rar (bag-of-words/TF-IDF)."""
    Xc = csc_matrix(X)
    Yc = Xc if Y is None else csc_matrix(Y)
    n1, n2 = Xc.shape[0], Yc.shape[0]
    K = np.zeros((n1, n2), dtype=np.float64)
    xi, xd, xp = Xc.indices, Xc.data, Xc.indptr
    yi, yd, yp = Yc.indices, Yc.data, Yc.indptr
    for col in range(Xc.shape[1]):
        xs, xe = xp[col], xp[col + 1]
        if xe == xs:
            continue
        ys, ye = yp[col], yp[col + 1]
        if ye == ys:
            continue
        xrows, xvals = xi[xs:xe], xd[xs:xe]
        yrows, yvals = yi[ys:ye], yd[ys:ye]
        K[np.ix_(xrows, yrows)] += np.minimum(xvals[:, None], yvals[None, :])
    return gamma * K


"""
=============================================================================
7. KNN CLASIFICATOR (generic, tip MNIST/imagini sau orice date numerice)
   Clasa reala, gata de folosit.
=============================================================================
"""


class KnnClassifier:
    """Clasificator k-NN simplu (fara sklearn), distanta L1 sau L2.
    Util pt date numerice/imagini aplatizate (ex: MNIST, shape (N, 784))."""

    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        if metric == 'l1':
            distances = np.sum(np.abs(self.train_images - test_image), axis=1)
        elif metric == 'l2':
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))
        else:
            raise ValueError("Metrica trebuie sa fie l1 sau l2")
        nearest_indices = np.argsort(distances)[:num_neighbors]
        nearest_labels = self.train_labels[nearest_indices]
        vote_counts = np.bincount(nearest_labels.astype(int))
        return np.argmax(vote_counts)

    def classify_all(self, test_images, num_neighbors=3, metric='l2'):
        return np.array([self.classify_image(test_images[i:i + 1], num_neighbors, metric)
                          for i in range(len(test_images))])


# echivalent rapid cu sklearn (de obicei mult mai rapid pt seturi mari):
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=3, metric='l2' -> 'minkowski', p=2)
# model.fit(train_images, train_labels); preds = model.predict(test_images)


"""
=============================================================================
8. PERCEPTRON / WIDROW-HOFF (clasificare liniara binara, +1/-1)
   Functie reala, gata de folosit.
=============================================================================
"""


def widrow_hoff(X, y, lr=0.1, epochs=70, random_state=42):
    """Antreneaza un perceptron cu regula delta (Widrow-Hoff, fara functie
    de activare in timpul antrenarii). y trebuie sa fie in {-1, +1}.
    Returneaza (ponderi, bias)."""
    rng = np.random.RandomState(random_state)
    n_features = X.shape[1]
    W = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        idx = np.arange(len(X))
        rng.shuffle(idx)
        for i in idx:
            pred_continua = np.dot(X[i], W) + b
            W = W - lr * (pred_continua - y[i]) * X[i]
            b = b - lr * (pred_continua - y[i])
    return W, b


def perceptron_predict(X, W, b):
    return np.sign(np.dot(X, W) + b)


"""
=============================================================================
9. CLUSTERING SI REDUCERE DIMENSIONALITATE GENERICA (daca cerinta e
   nesupervizata: grupare, vizualizare, compresie de features)
=============================================================================
"""
# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=5, random_state=42, n_init=10)
# cluster_ids = km.fit_predict(X)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, random_state=42)
# X_2d = pca.fit_transform(X)   # util si pt vizualizare rapida cu matplotlib

"""
=============================================================================
10. ALGORITM GENETIC GENERIC (daca cerinta e de optimizare / cautare
    euristica cu populatie, tip Mastermind)
    Functie reala, gata de adaptat (fitness + reprezentare cromozom).
=============================================================================
"""


def algoritm_genetic(fitness_fn, gene_len, gene_range, target_fitness=0,
                      pop_size=80, mutation_rate=0.2, n_generations=100,
                      random_state=None):
    """
    Schelet generic de algoritm genetic (minimizare fitness).
    fitness_fn(individ) -> numar (cu cat mai mic, cu atat mai bun).
    gene_len -- lungimea cromozomului (nr de gene).
    gene_range -- (min, max) valoare posibila per gena (int, inclusiv).
    target_fitness -- fitness la care oprim cautarea (solutie gasita).

    Returneaza (cel_mai_bun_individ, generatia_la_care_s-a_oprit).
    Adapteaza fitness_fn / crossover / mutatie in functie de cerinta reala!
    """
    rng = random.Random(random_state)
    lo, hi = gene_range

    def individ_random():
        return [rng.randint(lo, hi) for _ in range(gene_len)]

    populatie = [individ_random() for _ in range(pop_size)]

    for generatie in range(n_generations):
        populatie.sort(key=fitness_fn)
        cel_mai_bun = populatie[0]
        best_fitness = fitness_fn(cel_mai_bun)

        if best_fitness <= target_fitness:
            return cel_mai_bun, generatie

        parinti_buni = populatie[:pop_size // 2]
        generatie_noua = list(parinti_buni[:5])  # elitism

        while len(generatie_noua) < pop_size:
            tata, mama = rng.choice(parinti_buni), rng.choice(parinti_buni)
            punct_taiere = rng.randint(1, gene_len - 1) if gene_len > 1 else 1
            copil = tata[:punct_taiere] + mama[punct_taiere:]
            if rng.random() < mutation_rate:
                idx_mutat = rng.randint(0, gene_len - 1)
                copil[idx_mutat] = rng.randint(lo, hi)
            generatie_noua.append(copil)

        populatie = generatie_noua

    populatie.sort(key=fitness_fn)
    return populatie[0], n_generations


"""
=============================================================================
11. CAUTARE INFORMATA (A* / IDA*) -- mai putin probabil la un colocviu axat
    pe ML, dar util daca apare o problema de cautare in spatiul de stari
    (aceste teme apar de obicei la "Proiect RPC", nu la "test-laborator",
    dar mai bine sa ai schela pregatita).
=============================================================================
"""
import heapq


def a_star(start_state, is_goal_fn, successors_fn, heuristic_fn):
    """
    A* generic. start_state trebuie sa fie hashable (tuple, nu list!).
    successors_fn(stare) -> iterabil de stari succesoare (cost pas = 1).
    heuristic_fn(stare) -> estimare admisibila a costului ramas.
    Returneaza lista de stari [start, ..., goal] sau None daca nu exista solutie.
    """
    coada = [(heuristic_fn(start_state), 0, start_state, [start_state])]
    vizitate = set()
    while coada:
        _, cost_trecut, stare, path = heapq.heappop(coada)
        if is_goal_fn(stare):
            return path
        if stare in vizitate:
            continue
        vizitate.add(stare)
        for succ in successors_fn(stare):
            if succ not in vizitate:
                heapq.heappush(coada, (cost_trecut + 1 + heuristic_fn(succ),
                                        cost_trecut + 1, succ, path + [succ]))
    return None


"""
=============================================================================
12. EVALUARE GENERICA (clasificare si regresie)
=============================================================================
"""
# --- clasificare ---
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# acc = accuracy_score(y_val, preds)
# print(confusion_matrix(y_val, preds))
# print(classification_report(y_val, preds))

# --- regresie ---
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y_val, preds)   # atentie la multioutput='uniform_average' (implicit)

"""
=============================================================================
13. CAPCANE INTALNITE (citeste inainte sa pierzi timp cu debugging!)
=============================================================================

A) gamma prea mare pe kernel RBF + TF-IDF normalizat L2
   -> exp(-gamma * d^2) satureaza la 0 pt orice pereche de documente diferite
      (documente cvasi-ortogonale in spatii de mii de dimensiuni => d^2 ~ 1-2).
      Simptom: MSE gigantic (sute/mii), model practic prezice 0 peste tot.
   -> Solutie: daca gamma e FIXAT prin cerinta, scaleaza global vectorii de
      intrare (acelasi factor pe train/test) pana kernelul devine informativ.
      Cauta scala printr-un grid-search rapid.

B) Kernel de intersectie calculat brut, dens: np.minimum(X1[:,None,:], X2[None,:,:])
   -> exploda memoria (GiB/TiB) pt vocabulare mari (mii de coloane). Foloseste
      varianta pe coloane de mai sus (intersection_kernel_matrix), care e
      rapida daca datele sunt rare (bag-of-words tipic e >99% rar).

C) Normalizarea TF-IDF conteaza pt kernel-uri NELINIARE-in-scala:
   - RBF (patratic in distante): norm='l2' e ok, scalare globala ajuta la (A).
   - Intersectie / alte kernele liniare in valori: norm='l1' functioneaza
     mult mai bine decat 'l2' (evita ca documentele lungi sa domine kernelul).
   -> Daca esti aproape de un prag de punctaj, TESTEAZA ambele normalizari +
      sublinear_tf True/False -- diferenta poate fi decisiva (~0.03 MSE).

D) Respecta LITERAL constrangerile de arhitectura (nr straturi, nr neuroni,
   tip activare). MLPRegressor din sklearn NU are LeakyReLU -- daca cerinta
   o cere, scrie reteaua de mana (vezi sectiunea 5, LReLUMLPRegressor). Multe
   bareme dau 0 puncte pe partea de model daca arhitectura nu respecta
   cerinta, INDIFERENT de cat de bun e MSE-ul.

E) Timeout/memorie limitata in mediul de rulare: daca antrenarea dureaza
   mult (retele custom, kernel-uri mari), fixeaza un numar de epoci pe baza
   convergentei observate pe setul de validare (nu lasa early-stopping "la
   infinit" pe reantrenarea finala -- risti sa depasesti orice limita de timp).

F) Intotdeauna reantreneaza pe 100% din date pentru submisia FINALA, dupa ce
   ai ales hiperparametrii pe split-ul de validare. Val MSE != MSE-ul cu care
   esti notat, dar e cel mai bun proxy pe care il ai.

G) Verifica shape-ul si tipul predictiilor salvate (.npy) sa corespunda cu ce
   se asteapta (ex: (N, 2) pt regresie 2D, (N,) pt clasificare) -- o predictie
   cu shape gresit poate insemna 0 puncte automat la evaluare.

=============================================================================
14. PATTERN SUBMISIE FINALA (single-file, self-contained)
=============================================================================

Structura recomandata pentru fisierul final:

    if __name__ == "__main__":
        # 1. incarca datele (sectiunea 0)
        # 2. split intern train/val -> alege hiperparametri, raporteaza MSE/acuratete
        # 3. REANTRENEAZA pe 100% din train
        # 4. prezice pe test, salveaza:
        #    np.save("{Nume}_{Prenume}_{Grupa}_subiect{i}_solutia_{j}.npy", preds)
        # 5. printeaza un mesaj de confirmare + MSE-ul de validare (pt raport)
"""

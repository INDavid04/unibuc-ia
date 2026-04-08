# Sursa: Laboratorul 6.pdf

################
# Cerinta ex 3 #
################

# Antrenati un Perceptron cu algoritmul Widrow-Hoff pe multimea de antrenare X =[ [0, 0], [0, 1], [1, 0], [1, 1] ], y = [-1, 1, 1, -1]. Care este acuratetea pe multimea de antrenare? Apelati functia plot_decision_boundary la fiecare pas al algoritmului pentru a afisa dreapta de decizie.

##################
# Rezolvare ex 3 #
##################

import numpy as np

import matplotlib.pyplot as plt

def compute_y(x, W, bias):
    # Dreapta de decizie [x, y] * [W[0], W[1]] + b = 0
    return (-x * W[0] - bias) / (W[1] + 1e-10)

def plot_decision_boundary(X, y , W, b, current_x, current_y):
    x1 = -2.0
    y1 = compute_y(x1, W, b)
    x2 = 3.0
    y2 = compute_y(x2, W, b)

    # Sterge continutul ferestrei
    plt.clf()

    # Ploteaza multimea de antrenare
    color = 'r'
    if(current_y == -1):
        color = 'b'
    plt.ylim((-1, 2))
    plt.xlim((-1, 2))
    plt.plot(X[y == -1, 0], X[y == -1, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    
    # Ploteaza exemplul curent
    plt.plot(current_x[0], current_x[1], color+'s')
    
    # Afiseaza dreapta de decizie
    plt.plot([x1, x2] ,[y1, y2], 'black')
    plt.show(block=False)
    plt.pause(0.3)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, 1, 1, -1])

ponderi = np.zeros(2)
bias = 0.0
rata_de_invatare = 0.1
epoci = 70

for epoca in range(epoci):
    # Amesteca datelor la fiecare epoca
    indici = np.arange(len(X))
    np.random.shuffle(indici)
    
    for i in indici:
        current_x = X[i]
        current_y = y[i]
        
        # Predictia (fara functie de activare la Widrow-Hoff)
        predictie_continua = np.dot(current_x, ponderi) + bias
        
        # Actualizarea conform regulii delta
        ponderi = ponderi - rata_de_invatare * (predictie_continua - current_y) * current_x
        bias = bias - rata_de_invatare * (predictie_continua - current_y)

        plot_decision_boundary(X, y , ponderi, bias, current_x, current_y)
        
# Calculul acuratetii finale
predictii_finale = np.sign(np.dot(X, ponderi) + bias)
acuratete = np.mean(predictii_finale == y)
print(f"Acuratetea pe multimea de antrenare: {acuratete * 100}%")

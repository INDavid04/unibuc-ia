################
# Cerinta ex 1 #
################

# Creați clasa KnnClassifier, având constructorul următor:
# def __init__(self, train_images, train_labels):
#     self.train_images = train_images
#     self.train_labels = train_labels

##################
# Rezolvare ex 1 #
##################

# class KnnClassifier:
#     def __init__(self, train_images, train_labels):
#         self.train_images = train_images
#         self.train_labels = train_labels

################
# Cerinta ex 2 #
################

# Definiți metoda classify_image(self, test_image, num_neighbors = 3, metric = 'l2') care clasifică imaginea test_image cu metoda celor mai apropiați vecini, numărul vecinilor este stabilit de parametru num_neighbors, iar distanța poate fi L1 sau L2, în funcție de parametrul metric.

##################
# Rezolvare ex 2 #
##################

import numpy

class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors = 3, metric = 'l2'):

        # Calculeaza distantele
        
        if metric == 'l1':
            distances = numpy.sum(numpy.abs(self.train_images - test_image), axis = 1)
        elif metric == 'l2':
            distances = numpy.sqrt(numpy.sum((self.train_images - test_image)**2, axis = 1))
        else:
            raise ValueError("Metrica trebuie sa fie l1 sau l2")
        
        # Gaseste indecsii celor mai apropiati vecini

        nearest_indices = numpy.argsort(distances)[:num_neighbors] # sortam si luam primele trei distante

        # Extrage etichetele

        nearest_labels = self.train_labels[nearest_indices]

        # Gaseste clasa majoritara

        vote_counts = numpy.bincount(nearest_labels.astype(int)) # bincount calculeaza numarul de aparitii si functioneaza numai cu intregi, motiv pentru care am convertit nearest_labels la int cu astype(int)

        # Salveaza eticheta nu numar maxim de voturi

        predicted_label = numpy.argmax(vote_counts)

        # Returneaza eticheta

        return predicted_label
    
################
# Testare ex 2 #
################

if __name__ == "__main__":
    mock_train_images = numpy.array([
        [1, 1], 
        [2, 2],  
        [10, 10], 
        [11, 11]  
    ])
    
    mock_train_labels = numpy.array([0, 0, 1, 1])

    mock_test_image = numpy.array([[1.5, 1.5]])

    knn = KnnClassifier(mock_train_images, mock_train_labels)

    rezultat_predictie = knn.classify_image(mock_test_image, num_neighbors=3, metric='l2')

    print(f"Clasa prezisa pentru imaginea de test este: {rezultat_predictie}")
    
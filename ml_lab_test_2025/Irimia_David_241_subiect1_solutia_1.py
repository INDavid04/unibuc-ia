# IA | Colocviu 21 Iunie 2025 | Varianta 1 

#################
# Organizatoric #
#################

# Nume: Irimia David
# Grupa: 241
# Sursa subiect: https://github.com/INDavid04/unibuc-ia/blob/main/ml_lab_test_2025/c00-subiect-v1.pdf

##########
# Import #
##########

import numpy
from sklearn.naive_bayes import MultinomialNB

#########
# Class #
#########

class NumaraLitere:
    def __init__(self):
        self.vocabular = {}

    # Retine pozitia fiecarui caracter
    def strange_caractere(self, fraze):
        caractere = []
        for linie in fraze:
            for caracter in linie:
                if caracter not in self.vocabular:
                    caractere.append(caracter)
                    self.vocabular[caracter] = len(caractere) - 1
        return caractere
    
    def numara_aparitii(self, fraze):
        nr_linii = fraze.shape[0]
        nr_coloane = len(self.vocabular)
        aparitii = numpy.zeros((nr_linii, nr_coloane), dtype = numpy.float32)
        for i in range(nr_linii):
            for caracter in fraze[i]:
                # Ignora caracterele din afara vocabularului
                if caracter in self.vocabular:
                    aparitii[i][self.vocabular[caracter]] += 1
        return aparitii
    
########
# Main #
########

# Salveaza in lista fraze_antrenare toate liniile din fisierul train_sentences
with open("train_sentences.txt", "r", encoding="utf-8") as f:
    fraze_antrenare = numpy.array([linie.strip() for linie in f.readlines() if linie.strip()])

# Salveaza in lista fraze_testare toate liniile din fisierul test_sentences
with open("test_sentences.txt", "r", encoding="utf-8") as f:
    fraze_testare = numpy.array([linie.strip() for linie in f.readlines() if linie.strip()])

raspunsuri = numpy.load('train_labels.npy', allow_pickle=True)
analizator = NumaraLitere()
lista_vocabular = analizator.strange_caractere(fraze_antrenare)
aparitii_antrenare = analizator.numara_aparitii(fraze_antrenare)
aparitii_testare = analizator.numara_aparitii(fraze_testare)

# Antreneaza modelul bayes naiv
model_bayes_naiv = MultinomialNB()
model_bayes_naiv.fit(aparitii_antrenare, raspunsuri)

# Genereaza predictiile pentru datele de testare
predictii = model_bayes_naiv.predict(aparitii_testare)

numpy.save("Irimia_David_241_subiect1_solutia_1.npy", predictii)

print("Fisierul Irimia_David_241_subiect1_solutia_1.npy a fost generat cu succes!\n")

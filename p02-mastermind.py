# Proiect Rezolvarea Problemelor si Cautare - seria 24

#################
# Organizatoric #
#################

# Nume: Irimia David
# Grupa: 241
# Exercitiul: 3. Algoritmi Genetici > 3.1 Mastermind

###########
# Cerinta #
###########

# Mastermind este un joc in doi in care unul din jucatori creeaza un cod din 5 piese ordonate - piesele pot fi de 10 culori, sa zicem ca le numerotam de la 0 la 9 pentru simplitudine. Codul poate contine oricate piese de aceeasi culoare.
# Noi trebuie sa ghicim aceasta cheie intr-un numar minim posibil de incercari. La fiecare incercare, ni se spune cate piese sunt puse corect, si cate piese sunt de culoare corecta care sunt pe pozitie incorecta.
# Cerinta 1: Gasiti si implementati o functie de fitness cat mai potrivita pentru aceasta cerinta si justificati alegerea facuta (1p)
# Cerinta 2: Implementati, folosind algoritmi genetici, un joc complet de Mastermind. Codul adversarului va fi randomizat (2p)

import random

# Functia de fitness returneaza un numar intre 0 si 10, 0 cod identic, 10 cod total diferit
# Penalizare -2 daca cifra nu e buna, penalizare -1 daca cifra e buna dar pe o pozitie gresita si penalizare 0 daca cifra e buna si pe pozitia corecta
# De aceea se returneaza un numar intre 0 si 10, intrucat se pleaca de la presupusa eroare maxima, 2 * 5 = 10, 2 fiind penalizarea stabilita de noi si 5 fiind numarul de cifre al codului
# Astfel ca, fiecare cifra pozitionata bine aduce o scadere de 2 din eroare si fiecare cifra buna pozitionata gresit aduce o scadere de doar 1 din eroare
def calculeaza_eroare_fitness(candidat, cod_tinta):
    pozitii_bune = 0
    culori_bune = 0

    vizitat_tinta = [False] * 5
    vizitat_candidat = [False] * 5

    # Numara cifrele bune de pe pozitii bune
    for i in range(5):
        if candidat[i] == cod_tinta[i]:
            pozitii_bune += 1
            vizitat_tinta[i] = True
            vizitat_candidat[i] = True

    # Numara cifrele bune de pe pozitii gresite
    for i in range(5):
        if not vizitat_candidat[i]:
            for j in range(5):
                if not vizitat_tinta[j] and candidat[i] == cod_tinta[j]:
                    culori_bune += 1
                    vizitat_tinta[j] = True
                    break
    
    scor_un_fitness = 10 - (2 * pozitii_bune + 1 * culori_bune)

    return max(0, scor_un_fitness)

def porneste_joc(cod_secret):
    dimensiune_populatie = 80
    rata_mutatie = 0.2
    nr_generatii = 100

    populatie = []
    for _ in range(dimensiune_populatie):
        individ = [random.randint(0, 9) for _ in range(5)]
        populatie.append(individ)

    for generatie in range(nr_generatii):
        # Sorteaza crescator dupa eroare
        populatie.sort(key = lambda ind: calculeaza_eroare_fitness(ind, cod_secret))

        cel_mai_bun = populatie[0]
        eroare_minima = calculeaza_eroare_fitness(cel_mai_bun, cod_secret)

        print(f"Generatia #{generatie}: Cel mai bun candidat incercat {cel_mai_bun} are eroarea {eroare_minima}.")

        # Daca eroare minim a e zero inseamna ca s-a gasit codul secret
        if eroare_minima == 0:
            return cel_mai_bun, generatie
        
        # Selecteaza jumatatea superioara a populatiei cu erori mai mici
        parinti_buni = populatie[:dimensiune_populatie // 2]
        generatie_noua = []

        # Pastram cele mai apropiate coduri
        generatie_noua.extend(parinti_buni[:5])

        # Incrucisare si reproducere pentru restul populatiei
        while len(generatie_noua) < dimensiune_populatie:
            tata = random.choice(parinti_buni)
            mama = random.choice(parinti_buni)

            punct_taiere = random.randint(1, 4)
            copil = tata[:punct_taiere] + mama[punct_taiere:]

            if random.random() < rata_mutatie:
                index_mutat = random.randint(1, 4)
                copil[index_mutat] = random.randint(0,9)
            
            generatie_noua.append(copil)
        
        populatie = generatie_noua
    
    return populatie[0], nr_generatii

#############
# Test zone #
#############

cod_adversar = [random.randint(0, 9) for _ in range(5)]
solutie_gasita, runde = porneste_joc(cod_adversar)

print(f"Codul ghicit este {solutie_gasita}.")
print(f"Codul real era {cod_adversar}.")

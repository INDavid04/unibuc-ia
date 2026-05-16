# Proiect Rezolvarea Problemelor si Cautare - seria 24

#################
# Organizatoric #
#################

# Nume: Irimia David
# Grupa: 241
# Exercitiul: 1. Algoritmi de Cautare Informata > 1.1 Topspin

#######################
# Cerinta 1.1 Topspin #
#######################

# Se considera o banda circulara pe care sunt asezate n piese numerotate de la 1 la n si intr-un loc aleator se afla un turnichet pe care incap fix k piese. Putem roti banda cum vrem noi, fara niciun cost. Putem intoarce cele k piese de pe turnichet in orice moment.
# Vrem ca, in final, piesele sa fie ordonate crescator in ordinea acelor de ceasornic, intorcand turnichetul de cat mai putine ori.
# Cerinta 1: Gasiti o euristica admisibila si cat mai stransa pentru problema data. Justificati admisibilitatea euristicii (1p)
# Cerinta 2: Implementati algoritmul A* folosind euristica gasita. Daca euristica este inadmisibila, solutia nu se va puncta (1p)
# Cerinta 3: Implementati algoritmul IDA* folosind euristica gasita. Daca euristica este inadmisibila, solutia nu se va puncta (1p)

import heapq
import math

def calculeaza_conexiuni_gresite(banda_circulara):
    dimensiune_banda = len(banda_circulara)
    nr_greseli = 0

    for i in range(dimensiune_banda):
        piesa_curenta = banda_circulara[i]
        piesa_urmatoare = banda_circulara[(i + 1) % dimensiune_banda]
        if piesa_curenta == dimensiune_banda:
            piesa_estimata = 1
        else:
            piesa_estimata = piesa_curenta + 1
        if piesa_urmatoare != piesa_estimata:
            nr_greseli += 1

    return nr_greseli

def euristica(banda_circulara):
    # Euristica este egala cu raportul dintre numarul de greseli pe doi rotunjit prin adaos
    # Explicatie: Impartim la doi pentru ca o intoarcere de turnichet poate 'ordona' corect doar doua piese, capetele
    # Justificare: Este admisibila intrucat s-ar putea sa fie nevoie de mai multe mutari (piesele trebuiesc aduse in pozitii favorabile, unele mutari strica conexiuni temporar), deci costul real va fi mereu mai mare sau egal cu estimarea facuta
    return math.ceil(calculeaza_conexiuni_gresite(banda_circulara) / 2.0)

def calculeaza_succesori(banda_circulara, k):
    succesori = []
    dimensiune_banda = len(banda_circulara)

    # Genereaza toate starile posibile prin rotire si flip turnichet
    for index_curent in range(dimensiune_banda):
        banda_shiftata = banda_circulara[index_curent:] + banda_circulara[:index_curent]

        # Flip la primele k piese
        turnichet_intors = list(reversed(banda_shiftata[:k])) + list(banda_shiftata[k:])

        succesori.append(tuple(turnichet_intors))
    
    return set(succesori)

def e_ordonata(banda_circulara):
    dimensiune_banda = len(banda_circulara)
    index_prima_piesa = banda_circulara.index(1)
    banda_aliniata = banda_circulara[index_prima_piesa:] + banda_circulara[:index_prima_piesa]

    return banda_aliniata == tuple(range(1, dimensiune_banda + 1))

#################
# Algoritmul A* #
#################

def a_stelat(stare_inceput, k):
    stare_inceput = tuple(stare_inceput)
    coada = []
    heapq.heappush(coada, (euristica(stare_inceput), 0, stare_inceput, []))
    vizitate = set()

    while coada:
        cost_total_estimat, cost_trecut, piesa_curenta, path = heapq.heappop(coada)

        if e_ordonata(piesa_curenta):
            return path + [piesa_curenta]
        
        if piesa_curenta in vizitate:
            continue

        vizitate.add(piesa_curenta)

        for succ in calculeaza_succesori(piesa_curenta, k):
            if succ not in vizitate:
                heapq.heappush(coada, (cost_trecut + 1 + euristica(succ), cost_trecut + 1, succ, path + [piesa_curenta]))
    return None

###################
# Algoritmul IDA* #
###################

def ida_stelat(stare_inceput, k):
    stare_inceput = tuple(stare_inceput)

    def search(path, cost_trecut, bound):
        piesa_curenta = path[-1]
        cost_total_estimat = cost_trecut + euristica(piesa_curenta)

        if cost_total_estimat > bound:
            return cost_total_estimat
        
        if e_ordonata(piesa_curenta):
            return True
        
        min_bound = float('inf')

        for succ in calculeaza_succesori(piesa_curenta, k):
            if succ not in path:
                path.append(succ)
                t = search(path, cost_trecut + 1, bound)

                if t is True:
                    return True
                
                if t < min_bound:
                    min_bound = t

                path.pop()
        
        return min_bound
    
    bound = euristica(stare_inceput)
    path = [stare_inceput]

    while True:
        limita = search(path, 0, bound)
        
        if limita is True:
            return path
        
        if limita == float('inf'):
            return None
        
        bound = limita

#############
# Test zone #
#############

banda_initiala = [7, 3, 9, 2, 6, 1, 5, 4, 8]
k = 3

print("Algoritmul A*")
sol_a_stelat = a_stelat(banda_initiala, k)
print(f"Turnichetul a fost intors de {len(sol_a_stelat) - 1} ori.")

print("Algorimtul IDA*")
sol_ida_stelat = ida_stelat(banda_initiala, k)
print(f"Turnichetul a fost intors de {len(sol_ida_stelat) - 1} ori.")

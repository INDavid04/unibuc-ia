## **Inteligență Artificială Lucrare de laborator – 6 iunie 2026 Varianta 1** 

În această lucrare veți antrena  modele de regresie pentru determinarea coordonatelor geografice (latitudine și longitudine) asociate unor postări de pe rețelele de socializare, scrise în limba germană. 

Setul de date se află în directorul curent și este structurat în următoarele trei fișiere: 

- train_samples.txt - textele brute destinate etapei de antrenare a modelelor; 

- train_coordinates.npy - etichetele setului de antrenare (coordonatele geografice reale), folosite ca valori țintă; 

- test_samples.txt - setul de date de testare, necesar pentru crearea fișierelor de predicții. 

## _**Rezolvați următoarele cerințe:**_ 

1. **(3p)** Antrenați o rețea cu cel mult 22straturi ascunse și maxim 128 de neuroni pe un strat, folosiți funcția de activareLReLU pentru straturile ascunse. Pentru a obține punctajul acordat, trebuie să implementați corect modelul și să generați un fișier cu predicțiile pe datele de test (conform observațiilor de la final). 

**1p** – Mean Squared Error maxim pe datele de test = 1.30 **2p** – Mean Squared Error maxim pe datele de test = 1.10 **3p** – Mean Squared Error maxim pe datele de test = 0.90 

2. **(2.5p)** . Scrieți o funcție care construiește un vocabular format din cele mai relevante și specifice cuvinte pentru diferitele regiuni geografice din setul de antrenare, utilizând o abordare de tip TF-ISF (Term Frequency - Inverse Spatial Frequency), așa cum este descrisă mai jos. 

Funcția trebuie să partiționare spațiul geografic (definit de coordonatele datelor) întro grilă de 4x40subregiuni, toate având aceeași suprafață. Pentru a realiza acest lucru, veți determina mai întâi limitele extreme ale spațiului geografic pentru a genera granițele grilei. 

Ulterior, atribuiți fiecare text de antrenare subregiunii corespunzătoare pe baza coordonatelor sale. Concatenați toate textele dintr-o subregiune pentru a forma un singur "super-document" reprezentativ pentru acea zonă. Aplicați o transformare de tip TF-IDF la nivelul întregii hărți pe super-documentele rezultate. Astfel, cuvintele utilizate frecvent peste tot vor fi penalizate, iar termenii specifici unei anumite locații vor primi scoruri mari. 

Pentru fiecare subregiune, păstrați cele mai importante1500 de cuvinte pe baza scorurilor obținute (dacă o regiune are mai puține cuvinte le veți păstra pe toate). În 

final, reuniți vocabularele regionale obținute și păstrați doar cuvintele unice pentru a forma vocabularul general. 

3. **(2.5p)** Folosindu-vă de vocabularul calculat la exercițiul anterior, determinați reprezentarea TF-IDF a exemplelor și antrenați un model Kernel Ridge Regression folosind un kernel RBF. folosind gamma=777 

**1p** – Mean Squared Error maxim pe datele de test = 1.30 **2p** – Mean Squared Error maxim pe datele de test = 0.88 

**2.5p** – Mean Squared Error maxim pe datele de test = 0.80 

4. **(2.5p)** Antrenați un model SVM cu parametrul kernel setat la valoarea ‘precomputed’. Folosiți ca funcție kernel, pentru a crea matricele kernel de antrenare și test funcția kernel intersecție. Funcția kernel va fi aplicată pe vectorii rezultați de la exercițiul 3, mai precis reprezentarea TF-IDF. folosind gamma=777 

**1p** – Mean Squared Error maxim pe datele de test = 1.10 **2p** – Mean Squared Error maxim pe datele de test = 0.84 

**2.5p** – Mean Squared Error maxim pe datele de test = 0.80 

5. **(1.5p)** Creați un raport al experimentelor însoțit de evaluarea pe un set de validare a diferite combinații de hiperparametri pentru modelele de la punctele 1, 3 și 4. Raportul poate conține tabele sau grafice. 

## **1p - Oficiu** 

## _**Observații importante:**_ 

După implementarea cerințelor de mai sus, trebuie să trimiteți într-un folder denumit {Nume}_{Prenume}_{Grupa}_V{Varianta}: 

a) Cel mult 1 submisie pentru setul de testare cu metodele de la punctul 1; cel mult 3 submisii pentru setul de testare cu fiecare din metodele de la punctele 3 și 4. O submisie constă într-un fișier .npy denumit: 

_{Nume}_{Prenume}_{Grupa}_subiect{i}_solutia_{j}.npy_ 

unde i este numărul subiectului (1, 3 sau 4) și j este numărul submisiei (1, 2 sau 3), în care se află un tensor ce conține coordonatele pentru toate exemplele de test. 

b) Codul aferent pentru antrenarea modelelor și obținerea soluțiilor trimise. Pentru fiecare submisie, codul trebuie organizat într-un singur fișier .py denumit: 

_{Nume}_{Prenume}_{Grupa}_subiect{i}_solutia_{j}.py_ 

unde i este numărul subiectului (1, 2, 3 sau 4) și j este numărul submisiei (1, 2 sau 3). 

c) Raportul de la punctul 5. 

Folderul cu soluții și cod (fără date sau cerință) se va arhiva în format ZIP, sub denumirea: 

{Nume}_{Prenume}_{Grupa}_V{Varianta}.zip 

Exemplu: 

Denumire director: Popa_Marian_231_V1 Prima submisie pentru subiectul 3: Popa_Marian_subiect3_solutia1.txt Codul care a generat submisia de mai sus: Popa_Marian_subiect3_solutia1.py Denumire arhivă: Popa_Marian_231_V1.zip 

După finalizarea examenului, stații de lucru / laptop-ul va rămâne în sală, până la trimiterea soluției pe mail în prezența unui supraveghetor. 

Soluțiile se vor trimite la: **fmi.unibuc.ia+test@gmail.com** 


# Zadanie dla AI Developera


Mając zadany zbiór danych w postaci CSV (dane punktów dwuwymiarowych w postaci
współrzędnych x,y po jednym na każdą linię CSV) stworzyć sieć neuronową, która dla
zadanego stopnia wielomianu znajdzie współczynniki wielomianu, który najlepiej
aproksymuje zadany zbiór danych.

## Uruchamianie:

Trenowanie sieci: <br>
./polynomial train POLYNOMIAL_DEGREE PATH_TO_CSV

Estymacja wielomianem w punkcie (forward pass przez wytrenowaną sieć):<br>
./polynomial estimate X

Testy jednostkowe: <br>
cd folder_projektu <br>
python -m unittest discover


## Struktura projektu:
```
├── DATA
│   └── marie-knorps.csv
├── MODEL
│   ├── inoutnn_0.pcl
│   ├── inoutnn_1.pcl
│   ├── inoutnn_2.pcl
│   ├── inoutnn_3.pcl
│   ├── inoutnn_4.pcl
│   └── ...
├── polynomial
├── README.md
├── src
│   ├── data.png
│   ├── EDA.ipynb
│   ├── estimate.py
│   ├── inout_nn.py
│   ├── preprocessing.py
│   └── train.py
├── test
│   ├── test_inout_nn.py
│   └── test_preprocessing.py
└── zadanie.pdf

```
## Założenia:

W programie zastosowano jako model prostą sztuczną sieć neuronową. Sieć składa się z dwóch warstw - wejściowej, składającej się z $n+1$ elementów (n - stopień wielomianu aproksymującego) i wyjściowej składającej się z jednego neuronu z liniową funkcją aktywacji. Taka architektura jest równoważna regresji liniowej dla funkcji liniowej wielu zmiennych. <br>

### Przygotowanie danych:<br>
Dane są dzielone na część treningową i testową (możliwe również dodanie walidacyjnej). Test nie jest obecnie wypisywany. Zwraca wartości funkcji kosztu dla zbioru testowego i treningowego.<br>
Dodawane są do danych nowe kolumny, w których snajdują się kolejne potęgi wektora x: [1,x,x^2,..., x^n] <br>
Kolumny są normalizowane (x_i - mean)/std w celu lepszej zbieżności metody gradientu prostego użzytej do znalezienia wag sieci.

### Funkcja błędu:<br>
norma l2 z możliwością rozszerzenia o regularyzację 

### Metoda wyboru współczynników optymalnych:<br>
gradient descent z warunkiem stopu różnicą kolejnych wartości funkcji kosztu: <br>
'''convergence_condition = np.abs(cost_prev-cost)< tol''',
z dodatkowym zabezpieczeniem liczbą iteracji "itmax" i sprawdzeniem czy funkcja kosztu nie wzrasta. 


### Zwracanie współczynników: <br>
Dane wejściowe zostały przeskalowane, to oznacza, że wyliczone przez sieć współczynniki również powinny zostać przeskalowane.<br>
$\sum_{i=0}^{n}\theta_i* (x_i - mean_i)/std_i = \sum_{i=0}^{n}theta_i/std_i*x_i - \sum_{i=0}^{n}\theta_i* mean_i/std_i $

## Możliwości sieci:
Sieć została przetestowana dla n = 0,..,10 . Do ![](https://latex.codecogs.com/svg.latex?n=8) daje wyniki podobne do funkcji '''polyfit''' z biblioteki numpy, natomiast dla ![](https://latex.codecogs.com/svg.latex?n>8) przy standardowych ustawieniach metody gradientu prostego nie jest zbieżna. Należy wtedy zmienić szybkość uczenia metody gradientu.


## Zgodność z wymaganiami:
1. Program został napisany w Python3.6 z użyciem bibliotek Numpy (v. 1.14.2) on Ubuntu 17.10.
2. Nie używano zewnętrznych bibliotek do konstrukcji projektu (wyjątek - wizualizacja 
w Jupyter Notebooku: EDA.ipynb, która jest niezależna od programu)
3. Projekt umieszczony na githubie [HERE](https://github.com/mknorps/QuantumLab)
4. Commity pokazują rozwój aplikacji - zaczynam od szkieletu zgodnego z interfejsem, a następnie
uzupałniam klasy i funcje o wczytywanie, transformacje danych i sam model
5. Sieć jest zapisywana na dysk w Pythonowym formacie zapisu obiektów (pickle - object serialization)
6. Program korzysta z dynamicznego adresu Python3 i dodaje biblioteki z których korzysta do sys.path,
więc może być wywoływany z dowolnej lokacji
7. README.md obecne
8. Interfejs nie odbiega od zdefiniowanego w wymaganiach

## Możliwe rozszerzenia:
1. Optymalizacja kodu
2. Optymalizacja metody gradientu - dynamiczny dobór szybkości uczenia lub np grid test + paralelizacja
3. Sieć neuronowa z ukrytą warstwą i eksponencjalną(wystarczy $n$ wyrazów szeregu Taylora) funkcją aktywacji (wtedy zamiast gradientu prostego będzie wsteczna propagacja)
4. rozwinięcie interfejsu m.in. o opcję verbose


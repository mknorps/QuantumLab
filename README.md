# Zadanie dla AI Developera


Mając zadany zbiór danych w postaci CSV (dane punktów dwuwymiarowych w postaci
współrzędnych x,y po jednym na każdą linię CSV) stworzyć sieć neuronową, która dla
zadanego stopnia wielomianu znajdzie współczynniki wielomianu, który najlepiej
aproksymuje zadany zbiór danych.

## Uruchamianie:

W modzie treningowym:
./polynomial train POLYNOMIAL_DEGREE PATH_TO_CSV

Estymacja wielomianem w punkcie:
./polynomial estimate X

Testy jednostkowe:
cd folder_projektu
python -m unittest discover


## Struktura projektu:
```
  ├── DATA
  │   └── marie-knorps.csv
  ├── polynomial
  ├── README.md
  ├── src
  │   ├── EDA.ipynb
  │   ├── estimate.py
  │   ├── preprocessing.py
  │   ├── simple_nn.py
  │   └── train.py
  ├── test
  │   └── test_preprocessing.py
  └── zadanie.pdf
```
## Możliwości sieci:
W programime zastosowano jako model prostą sztuczną sieć neuronową. Sieć składa się z trzech warstw, w tym jednej ukrytej.<br>
Warstwa wejścia składa się z n+1 elementów, gdzie n jest rzędem wielomianu optymalizującego, a "+1" jest tzw. "bias unit" gwarantujacą przesunięcie o czynnik stały funkcji liniowych.<br>
Warstwa ukryta składa się z ? neuronów <br>
Warstwa wyjściowa składa się z jednego elementu.


## Założenia:
Funkcja błędu:<br>
norma l2

Metoda wyboru współczynników optymalnych:<br>
gradient descent




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


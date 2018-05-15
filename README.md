# Zadanie dla AI Developera


Mając zadany zbiór danych w postaci CSV (dane punktów dwuwymiarowych w postaci
współrzędnych x,y po jednym na każdą linię CSV) stworzyć sieć neuronową, która dla
zadanego stopnia wielomianu znajdzie współczynniki wielomianu, który najlepiej
aproksymuje zadany zbiór danych.

Program został napisany w Python3.6 z użyciem bibliotek Numpy (v. 1.14.2).

## Struktura projektu:
```
  ├── DATA
  │   └── marie-knorps.csv
  ├── polynomial
  ├── README.md
  ├── simple_nn.py
  ├── src
  │   ├── EDA.ipynb
  │   ├── estimate.py
  │   ├── __init__.py
  │   ├── preprocessing.py
  │   ├── __pycache__
  │   └── train.py
  ├── test
  │   ├── __init__.py
  │   ├── __pycache__
  │   └── test_preprocessing.py
  └── zadanie.pdf
```
## Możliwości sieci:
...

## Założenia:
...

## Uruchamianie:

W modzie traningowym:
./polynomial train POLYNOMIAL_DEGREE PATH_TO_CSV

Estymacja wielomianem w punkcie:
./polynomial estimate X

Testy jednostkowe:
cd folder_projektu
python -m unittest discover

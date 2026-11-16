# Ekstrakcja cech na podstawie minucji

## Temat
Projekt i implementacja systemu ekstrakcji cech na podstawie odcisku palca (linii papilarnych).

## Baza danych
**Sokoto Coventry Fingerprint Dataset (SOCOFing)**.
* **Opis bazy:** Zawiera oryginalne skany od 600 afrykańskich subiektów oraz syntetycznie zniekształcone wersje testowe (z uwzględnieniem zatarć, rotacji centralnej oraz przecięć typu Z-cut).
* **Link:** [Sokoto Coventry Fingerprint Dataset na Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing)

## Opis projektu

Aplikacja służy do automatycznego wykrywania minucji na podstawie obrazu odcisku palca.

Minucje to charakterystyczne punkty linii papilarnych:
- **Ending** – zakończenie linii papilarnej.
- **Bifurcation** – rozdwojenie linii papilarnej.

System wykorzystuje klasyczne przetwarzanie obrazu oraz sieć neuronową CNN opartą na architekturze ResNet-18.

Wyniki można przeglądać w aplikacji internetowej, zapisywać w bazie SQLite oraz eksportować do plików JSON i CSV.

## Etapy działania

1. Wczytanie obrazu i konwersja do skali szarości.
2. Preprocessing, binaryzacja i skeletonizacja.
3. Wyszukiwanie kandydatów metodą Crossing Number.
4. Klasyfikacja fragmentów 15 × 15 pikseli przez CNN.
5. Filtrowanie wyników i tworzenie listy minucji.
6. Generowanie wektora cech (x, y, typ).
7. Wizualizacja wyników, zapis w bazie i eksport danych.

## Wykorzystane technologie

- Python – główny język programowania.
- OpenCV – przetwarzanie obrazów.
- NumPy – operacje na danych obrazowych.
- scikit-image – skeletonizacja.
- PyTorch i torchvision – model ResNet-18.
- Flask – aplikacja internetowa.
- SQLite – baza danych.
- Matplotlib – wizualizacja wyników treningu.

## Struktura projektu

    app.py
    preprocessing.py
    model_cnn.py
    features.py
    database.py
    train.py
    tests.py
    templates/
        index.html
    model_minucje_v3.pth
    biometria.db

## Instalacja i uruchomienie

### 1. Utworzenie środowiska

    py -m venv .venv

### 2. Aktywacja środowiska (Windows)

    .venv\Scripts\activate

### 3. Instalacja bibliotek

    pip install flask numpy opencv-python scikit-image matplotlib torch torchvision

### 4. Uruchomienie aplikacji

    python app.py

Aplikacja będzie dostępna pod adresem:

http://127.0.0.1:5000

Przed uruchomieniem należy upewnić się, że wytrenowany model `model_minucje_v3.pth` znajduje się w głównym folderze projektu.

## Trenowanie modelu

Do treningu wykorzystano zbiór SOCOFing.

Obrazy należy umieścić w folderze:

    data/SOCOFing/Real

Następnie uruchomić:

    python train.py

Model jest trenowany przez 15 epok.

Wyniki zapisywane są w plikach:

- `model_minucje_v3.pth` – wytrenowane wagi modelu.
- `wykres_uczenia_v3.png` – wykres dokładności i funkcji straty.

## Wyniki

Podczas treningu uzyskano dokładność walidacyjną klasyfikacji fragmentów obrazu na poziomie około 88–89%.

Wartość ta dotyczy klasyfikacji lokalnych fragmentów odcisku, a nie skuteczności rozpoznawania tożsamości osób.

## Możliwości aplikacji

- Wczytywanie obrazów BMP, PNG i JPG.
- Automatyczne wykrywanie minucji.
- Wizualizacja zakończeń i rozdwojeń linii.
- Generowanie wektora cech.
- Zapisywanie wyników w SQLite.
- Eksport danych do JSON i CSV.




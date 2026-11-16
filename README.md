# Ekstrakcja cech na podstawie minucji

## Temat
Projekt i implementacja systemu ekstrakcji cech na podstawie odcisku palca (linii papilarnych).

## Baza danych
**Sokoto Coventry Fingerprint Dataset (SOCOFing)**.
* **Opis bazy:** Zawiera oryginalne skany od 600 afrykańskich subiektów oraz syntetycznie zniekształcone wersje testowe (z uwzględnieniem zatarć, rotacji centralnej oraz przecięć typu Z-cut).
* **Link:** [Sokoto Coventry Fingerprint Dataset na Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing)

## Opis projektu

Celem projektu jest stworzenie systemu umożliwiającego automatyczne wykrywanie i ekstrakcję cech biometrycznych na podstawie obrazu odcisku palca.

System analizuje przebieg linii papilarnych i wyszukuje charakterystyczne punkty, nazywane **minucjami**.

W projekcie wykrywane są dwa rodzaje minucji:
- **Ending** – zakończenie linii papilarnej.
- **Bifurcation** – rozdwojenie linii papilarnej.

Do analizy wykorzystano klasyczne metody przetwarzania obrazów oraz konwolucyjną sieć neuronową CNN opartą na architekturze ResNet-18.

Aplikacja umożliwia wyświetlenie wyników analizy, zapisanie ich w bazie SQLite oraz eksport do plików JSON i CSV.

## Schemat działania systemu

Proces ekstrakcji cech składa się z siedmiu etapów:

### 1. Wczytanie obrazu

Program przyjmuje obraz odcisku palca w formacie BMP, PNG lub JPG.

Biblioteka OpenCV odpowiada za odczyt obrazu i jego konwersję do skali szarości.

### 2. Preprocessing obrazu

Obraz jest przygotowywany do dalszej analizy.

Wykonywane operacje:
- Usunięcie ramki skanera.
- Zmiana rozmiaru obrazu do 260 × 280 pikseli.
- Wyznaczenie maski obszaru odcisku.
- Odszumianie za pomocą filtra Gaussa.
- Binaryzacja adaptacyjna.
- Usunięcie drobnych zakłóceń.
- Skeletonizacja linii papilarnych.

Skeletonizacja polega na zmniejszeniu grubości linii do jednego piksela, przy zachowaniu ich przebiegu.

### 3. Wykrywanie kandydatów na minucje

Do wykrywania potencjalnych minucji wykorzystano metodę **Crossing Number (CN)**.

Metoda analizuje otoczenie każdego piksela linii papilarnej w oknie 3 × 3.

Podstawowe wartości:
- **CN = 1** – zakończenie linii.
- **CN = 2** – zwykły fragment linii.
- **CN = 3** – rozdwojenie linii.

Następnie wykonywane jest filtrowanie kandydatów w celu ograniczenia liczby błędnych detekcji.

### 4. Klasyfikacja CNN

Kandydaci są analizowani przez konwolucyjną sieć neuronową CNN.

W projekcie zastosowano architekturę **ResNet-18**, dostosowaną do obrazów jednokanałowych.

Dla każdego kandydata pobierany jest fragment szkieletu o rozmiarze 15 × 15 pikseli.

Model rozróżnia trzy klasy:
- **0** – inny fragment lub fałszywy kandydat.
- **1** – zakończenie linii.
- **2** – rozdwojenie linii.

Na podstawie wyników klasyfikacji i przyjętych progów odrzucane są niepewne detekcje.

### 5. Tworzenie końcowej listy minucji

Po klasyfikacji program:
- Odrzuca punkty niespełniające wymaganych warunków.
- Usuwa blisko położone, powtarzające się detekcje.
- Zapisuje zaakceptowane zakończenia i rozdwojenia.

Wyniki są przedstawiane na obrazie:
- Zielony kolor – zakończenia.
- Czerwony kolor – rozdwojenia.

### 6. Tworzenie wektora cech

Każda wykryta minucja jest opisywana za pomocą trzech wartości:

- `x` – współrzędna pozioma.
- `y` – współrzędna pionowa.
- `type` – rodzaj minucji.

Przykładowa minucja:

```json
{
  "x": 92,
  "y": 160,
  "type": "bifurcation"
}
```

Program oblicza również statystyki:
- Całkowitą liczbę minucji.
- Liczbę zakończeń linii.
- Liczbę rozdwojeń linii.

Dodatkowo generowany jest skrót SHA-256 danych szablonu.

Skrót pozwala sprawdzić, czy dane są identyczne, ale nie służy do oceny podobieństwa dwóch odcisków.

### 7. Wyświetlanie i zapis wyników

Aplikacja internetowa została przygotowana przy użyciu frameworka Flask.

Użytkownik może:
- Wczytać obraz odcisku palca.
- Uruchomić analizę.
- Zobaczyć oryginalny obraz.
- Zobaczyć szkielet linii papilarnych.
- Zobaczyć wykryte minucje.
- Sprawdzić statystyki.
- Wyeksportować wyniki do JSON lub CSV.

Wyniki analizy są również zapisywane w lokalnej bazie danych SQLite.

## Wykorzystane technologie

| Technologia | Zastosowanie |
|---|---|
| Python | Główny język programowania |
| OpenCV | Wczytywanie i przetwarzanie obrazów |
| NumPy | Operacje na tablicach i współrzędnych |
| scikit-image | Skeletonizacja linii papilarnych |
| PyTorch | Trenowanie i uruchamianie modelu CNN |
| torchvision | Architektura ResNet-18 |
| Flask | Aplikacja internetowa i API |
| SQLite | Przechowywanie wyników |
| Matplotlib | Wizualizacja treningu i walidacji |
| HTML, CSS, JavaScript | Interfejs użytkownika |

## Struktura projektu

```text
ekstrakcja_cech_na_podstawie_minucji/
│
├── app.py
├── preprocessing.py
├── model_cnn.py
├── features.py
├── database.py
├── train.py
├── tests.py
│
├── model_minucje_v3.pth
├── biometria.db
├── wykres_uczenia_v3.png
│
├── templates/
│   └── index.html
│
└── data/
    └── SOCOFing/
        └── Real/
```

### Opis plików

**app.py**

Główny plik aplikacji Flask. Obsługuje przesyłanie obrazów, uruchamia proces analizy i zwraca wyniki do przeglądarki.

**preprocessing.py**

Zawiera funkcje przygotowujące obraz do analizy, w tym normalizację, wyznaczanie maski, binaryzację i skeletonizację.

**model_cnn.py**

Odpowiada za wyszukiwanie kandydatów metodą Crossing Number, klasyfikację przy użyciu ResNet-18 i końcowe filtrowanie minucji.

**features.py**

Tworzy wektor cech, oblicza statystyki i skrót SHA-256 oraz obsługuje eksport do JSON i CSV. Zawiera również eksperymentalną funkcję porównywania wektorów.

**database.py**

Obsługuje bazę SQLite, zapis wyników, odczyt historii oraz wczytywanie obrazów ze zbioru SOCOFing.

**train.py**

Odpowiada za przygotowanie danych treningowych, uczenie sieci neuronowej, walidację i zapis wytrenowanego modelu.

**tests.py**

Zawiera testy sprawdzające odrzucanie nieprawidłowych obrazów, zachowanie systemu przy zaszumionym obrazie oraz czas przetwarzania.

**templates/index.html**

Interfejs aplikacji internetowej. Umożliwia przesyłanie odcisków, przeglądanie obrazów i pobieranie wyników.

**model_minucje_v3.pth**

Plik zawierający wytrenowane wagi sieci ResNet-18.

**biometria.db**

Lokalna baza danych SQLite zawierająca wyniki przeprowadzonych analiz.

**wykres_uczenia_v3.png**

Wykres funkcji straty oraz dokładności modelu podczas treningu i walidacji.

## Instalacja

Do uruchomienia projektu wymagany jest Python oraz niezbędne biblioteki.

### 1. Pobranie projektu

Sklonuj repozytorium lub pobierz jego zawartość jako ZIP.

### 2. Utworzenie środowiska wirtualnego

W terminalu, w katalogu projektu, wykonaj:

```bash
python -m venv .venv
```

### 3. Aktywacja środowiska (Windows)

```bash
.venv\Scripts\activate
```

### 4. Instalacja bibliotek

```bash
pip install flask numpy opencv-python scikit-image torch torchvision matplotlib
```

### 5. Uruchomienie aplikacji

Upewnij się, że plik `model_minucje_v3.pth` znajduje się w głównym katalogu projektu.

Uruchom:

```bash
python app.py
```

Następnie otwórz w przeglądarce:

http://127.0.0.1:5000

## Obsługa aplikacji

1. Otwórz aplikację w przeglądarce.
2. Kliknij przycisk wyboru pliku.
3. Wybierz obraz odcisku palca.
4. Kliknij **Analizuj odcisk**.
5. Poczekaj na zakończenie analizy.
6. Sprawdź wykryte minucje i statystyki.
7. W razie potrzeby wyeksportuj dane do JSON lub CSV.

## Trenowanie modelu

Do treningu wykorzystano obrazy ze zbioru **SOCOFing**.

Obrazy należy umieścić w katalogu:

```text
data/SOCOFing/Real
```

Aby rozpocząć trening, uruchom:

```bash
python train.py
```

### Parametry treningu

| Parametr | Wartość |
|---|---|
| Model | ResNet-18 |
| Liczba klas | 3 |
| Rozmiar fragmentu | 15 × 15 pikseli |
| Liczba epok | 15 |
| Batch size | 128 |
| Optymalizator | AdamW |
| Learning rate | 0.0003 |
| Funkcja straty | CrossEntropyLoss |

Podczas przygotowywania danych stosowany jest podział treningowy i walidacyjny na podstawie identyfikatorów osób.

Wykorzystywana jest również augmentacja polegająca na obracaniu i odbijaniu fragmentów obrazów.

Po zakończeniu treningu zapisywane są:
- `model_minucje_v3.pth`
- `wykres_uczenia_v3.png`

## Wyniki treningu

Podczas treningu uzyskano dokładność walidacyjną klasyfikacji fragmentów obrazu na poziomie około **88–89%**.

Dokładność treningowa pod koniec uczenia osiągnęła około 90–91%.

Funkcja straty treningowej stopniowo malała, natomiast strata walidacyjna wykazywała wahania.

Wyniki te odnoszą się do klasyfikacji fragmentów odcisków według przygotowanych etykiet, a nie do skuteczności identyfikacji użytkowników.

## Testowanie

Do przeprowadzenia testów służy plik `tests.py`.

Uruchomienie:

```bash
python tests.py
```

Testowane elementy:
- Odrzucanie pustego obrazu.
- Działanie systemu na zaszumionym odcisku.
- Wydajność przetwarzania większej liczby obrazów.

Testy korzystające z obrazów SOCOFing wymagają obecności odpowiedniego katalogu z danymi.

## Eksport wyników

Program umożliwia zapis wyników do dwóch formatów.

### JSON

Zawiera informacje o analizowanym obrazie, statystyki, skrót SHA-256 i listę wykrytych minucji.

### CSV

Zawiera tabelę wykrytych minucji wraz z ich współrzędnymi i typami.

## Ograniczenia projektu

System ma charakter edukacyjny i badawczy.

Jego głównym zadaniem jest ekstrakcja cech biometrycznych, a nie pełna identyfikacja użytkownika.

Obecna implementacja nie stanowi kompletnego systemu uwierzytelniania.

Dokładność klasyfikatora CNN nie jest równoznaczna ze skutecznością porównywania odcisków palców.

Przed zastosowaniem systemu w praktyce konieczne byłyby dalsze testy jakości detekcji oraz niezawodności rozpoznawania.

## Autorzy

**Patryk Majewski**  
**Rafał Kruszewski**

Projekt zrealizowany w ramach przedmiotu **Podstawy biometrii**.

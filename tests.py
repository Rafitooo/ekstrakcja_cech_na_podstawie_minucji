import os
import time
import numpy as np

from preprocessing import przetworz_obraz
from model_cnn import wykryj_minucje
from features import utworz_wektor_cech, porownaj_wektory
from database import wczytaj_baze_paczkami

FOLDER = "data/SOCOFing/Real"


def test_pustego_obrazu():
    obraz = np.full((103, 96), 255, dtype=np.uint8)

    try:
        przetworz_obraz(obraz)
        print("[FAIL] Bialy obraz zostal zaakceptowany.")
    except ValueError:
        print("[PASS] Bialy obraz zostal odrzucony.")


def test_szumu():
    if not os.path.isdir(FOLDER):
        print("[SKIP] Brak folderu Real.")
        return

    paczka = next(wczytaj_baze_paczkami(FOLDER, 1), None)
    if not paczka:
        print("[SKIP] Brak obrazow.")
        return

    nazwa, obraz = paczka[0]

    los = np.random.default_rng(42)
    szum = los.normal(0, 15, obraz.shape)
    obraz_z_szumem = np.clip(
        obraz.astype(float) + szum, 0, 255
    ).astype(np.uint8)

    std1, szk1 = przetworz_obraz(obraz)
    std2, szk2 = przetworz_obraz(obraz_z_szumem)

    cechy1 = utworz_wektor_cech(
        nazwa, wykryj_minucje(szk1, std1)
    )
    cechy2 = utworz_wektor_cech(
        nazwa, wykryj_minucje(szk2, std2)
    )

    wynik = porownaj_wektory(cechy1, cechy2)
    print(f"[INFO] Podobienstwo z zaszumionym obrazem: {wynik}%")
    print("[INFO] Matching Score nie oznacza dokladnosci wykrywania minucji.")


def test_wydajnosci(limit=100):
    if not os.path.isdir(FOLDER):
        print("[SKIP] Brak folderu Real.")
        return

    start = time.time()
    licznik = 0

    for paczka in wczytaj_baze_paczkami(FOLDER, 25):
        for nazwa, obraz in paczka:
            try:
                std, szkielet = przetworz_obraz(obraz)
                wykryj_minucje(szkielet, std)
                licznik += 1
            except ValueError:
                continue

            if licznik >= limit:
                break

        if licznik >= limit:
            break

    if licznik:
        czas = time.time() - start
        print(f"[INFO] {licznik} obrazow w {czas:.2f} s")


if __name__ == "__main__":
    test_pustego_obrazu()
    test_szumu()
    test_wydajnosci()
import json
import csv
import hashlib
import numpy as np

def utworz_wektor_cech(nazwa_pliku, lista_minucji):
    """
    Tworzy jednokierunkowy wektor cech biometrycznych wraz ze statystykami.
    """
    liczba_zakonczen = sum(1 for m in lista_minucji if m["type"] == "ending")
    liczba_rozdwojen = sum(1 for m in lista_minucji if m["type"] == "bifurcation")

    # Wyliczenie skrotu kryptograficznego SHA-256 dla wektora
    tekst_wektora = json.dumps(lista_minucji, sort_keys=True)
    hash_sha256 = hashlib.sha256(tekst_wektora.encode("utf-8")).hexdigest()

    return {
        "plik": nazwa_pliku,
        "hash_szablonu": hash_sha256,
        "statystyki": {
            "liczba_minucji": len(lista_minucji),
            "zakonczenia_linii": liczba_zakonczen,
            "rozdwojenia_linii": liczba_rozdwojen
        },
        "minucje": lista_minucji
    }

def eksportuj_do_json(wektor_danych, sciezka="wektor_cech.json"):
    with open(sciezka, "w", encoding="utf-8") as plik:
        json.dump(wektor_danych, plik, indent=4, ensure_ascii=False)
    return sciezka

def eksportuj_do_csv(wektor_danych, sciezka="wektor_cech.csv"):
    with open(sciezka, "w", newline="", encoding="utf-8") as plik:
        writer = csv.writer(plik)
        writer.writerow(["ID", "Wspolrzedna_X", "Wspolrzedna_Y", "Typ_Minucji"])
        for indeks, m in enumerate(wektor_danych["minucje"], start=1):
            writer.writerow([indeks, m["x"], m["y"], m["type"]])
    return sciezka

def porownaj_wektory(wektor1, wektor2, prog_odleglosci=8.0):
    """
    Modul Matching (Could Have):
    Porownuje dwa wektory minucji i zwraca wskaznik podobienstwa (Matching Score w %).
    """
    punkty1 = wektor1.get("minucje", [])
    punkty2 = wektor2.get("minucje", [])

    if not punkty1 or not punkty2:
        return 0.0

    dopasowane = 0
    uzyte_indeksy = set()

    for m1 in punkty1:
        for idx, m2 in enumerate(punkty2):
            if idx in uzyte_indeksy:
                continue
            if m1["type"] == m2["type"]:
                dystans = np.hypot(m1["x"] - m2["x"], m1["y"] - m2["y"])
                if dystans <= prog_odleglosci:
                    dopasowane += 1
                    uzyte_indeksy.add(idx)
                    break

    srednia_liczba = (len(punkty1) + len(punkty2)) / 2.0
    wynik = (dopasowane / srednia_liczba) * 100.0
    return round(min(wynik, 100.0), 2)
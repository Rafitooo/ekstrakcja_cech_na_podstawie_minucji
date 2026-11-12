import os
import json
import sqlite3
from preprocessing import wczytaj_obraz

DB_NAME = "biometria.db"


def init_db():
    with sqlite3.connect(DB_NAME) as baza:
        baza.execute("""
            CREATE TABLE IF NOT EXISTS odciski (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nazwa_pliku TEXT NOT NULL,
                liczba_minucji INTEGER,
                zakonczenia INTEGER,
                rozdwojenia INTEGER,
                wektor_json TEXT NOT NULL,
                data_analizy TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def zapisz_do_bazy(wektor):
    init_db()
    statystyki = wektor["statystyki"]

    with sqlite3.connect(DB_NAME) as baza:
        baza.execute("""
            INSERT INTO odciski
            (nazwa_pliku, liczba_minucji, zakonczenia, rozdwojenia, wektor_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            wektor["plik"],
            statystyki["liczba_minucji"],
            statystyki["zakonczenia_linii"],
            statystyki["rozdwojenia_linii"],
            json.dumps(wektor["minucje"])
        ))


def pobierz_historie_z_bazy(limit=10):
    init_db()

    with sqlite3.connect(DB_NAME) as baza:
        return baza.execute("""
            SELECT id, nazwa_pliku, liczba_minucji,
                   zakonczenia, rozdwojenia, data_analizy
            FROM odciski
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()


def wczytaj_baze_paczkami(folder="data/SOCOFing/Real", rozmiar_paczki=10):
    if not os.path.isdir(folder):
        raise FileNotFoundError("Nie znaleziono folderu: " + folder)

    if rozmiar_paczki < 1:
        raise ValueError("Rozmiar paczki musi byc dodatni.")

    paczka = []

    for katalog, podkatalogi, pliki in os.walk(folder):
        podkatalogi.sort()

        for nazwa in sorted(pliki):
            if not nazwa.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
                continue

            sciezka = os.path.join(katalog, nazwa)

            try:
                with open(sciezka, "rb") as plik:
                    obraz = wczytaj_obraz(plik.read())
            except (ValueError, OSError):
                continue

            paczka.append((nazwa, obraz))

            if len(paczka) >= rozmiar_paczki:
                yield paczka
                paczka = []

    if paczka:
        yield paczka
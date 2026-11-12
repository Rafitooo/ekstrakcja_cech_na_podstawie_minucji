import random
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from database import wczytaj_baze_paczkami
from preprocessing import przetworz_obraz, maska_odcisku
from model_cnn import (
    utworz_model_resnet,
    znajdz_kandydatow,
    surowi_kandydaci,
    SCIEZKA_MODELU
)


def przygotuj_zbior_danych(liczba_obrazow=600, limit_na_klase=3500):
    los = random.Random(42)
    dane = {"train": [[], [], []], "val": [[], [], []]}
    licznik = 0

    print("[1/3] Przygotowywanie danych treningowych...")

    for paczka in wczytaj_baze_paczkami("data/SOCOFing/Real", rozmiar_paczki=25):
        for nazwa, obraz in paczka:
            identyfikator = nazwa.split("__")[0]

            if not identyfikator.isdigit():
                continue

            try:
                obraz_std, szkielet = przetworz_obraz(obraz)
            except ValueError:
                continue

            # Podzial wedlug osoby
            zbior = "val" if int(identyfikator) % 5 == 0 else "train"
            klasy = dane[zbior]

            binarny = (szkielet > 0).astype(np.float32)

            # Kandydaci po filtrowaniu
            kandydaci = znajdz_kandydatow(szkielet, obraz_std)
            los.shuffle(kandydaci)

            # Klasa 1 - zakonczenia, klasa 2 - rozdwojenia
            for typ, klasa in ((1, 1), (3, 2)):
                punkty = [p for p in kandydaci if p[2] == typ][:12]

                for x, y, _ in punkty:
                    wycinek = binarny[y-7:y+8, x-7:x+8]
                    klasy[klasa].append(wycinek)

            # Klasa 0 - zwykle linie i podejrzane punkty
            maska = maska_odcisku(obraz_std)

            wszyscy = surowi_kandydaci(
                szkielet,
                maska,
                filtr_galezi=False
            )

            niepewne = []

            for x, y, cn in wszyscy:
                daleko = all(
                    np.hypot(x - px, y - py) >= 8
                    for px, py, _ in kandydaci
                )

                if daleko:
                    niepewne.append((x, y))

            los.shuffle(niepewne)

            for x, y in niepewne[:4]:
                wycinek = binarny[y-7:y+8, x-7:x+8]
                klasy[0].append(wycinek)

            # Zwykle fragmenty linii papilarnych
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (19, 19)
            )
            maska = cv2.erode(maska, kernel)

            sasiedzi = cv2.filter2D(
                binarny.astype(np.uint8),
                -1,
                np.ones((3, 3), np.uint8)
            )

            ys, xs = np.where(
                (binarny > 0) & (maska > 0) & (sasiedzi == 3)
            )

            normalne = list(zip(xs.tolist(), ys.tolist()))
            los.shuffle(normalne)
            dodane = 0

            for x, y in normalne:
                if x < 8 or y < 8:
                    continue

                if x >= binarny.shape[1] - 8 or y >= binarny.shape[0] - 8:
                    continue

                blisko = any(
                    np.hypot(x - px, y - py) < 10
                    for px, py, _ in kandydaci
                )

                if blisko:
                    continue

                klasy[0].append(binarny[y-7:y+8, x-7:x+8])
                dodane += 1

                if dodane >= 8:
                    break

            licznik += 1

            if licznik % 50 == 0:
                print(f"Przygotowano {licznik}/{liczba_obrazow} obrazow")

            if licznik >= liczba_obrazow:
                break

        if licznik >= liczba_obrazow:
            break

    def zrob_tensor(klasy, nazwa):
        ile = min(limit_na_klase, *(len(k) for k in klasy))

        if ile < 10:
            raise ValueError(
                f"Za malo przykladow w zbiorze {nazwa}. Sprawdz baze SOCOFing."
            )

        probki = []
        etykiety = []

        for klasa, punkty in enumerate(klasy):
            los.shuffle(punkty)
            probki.extend(punkty[:ile])
            etykiety.extend([klasa] * ile)

        pary = list(zip(probki, etykiety))
        los.shuffle(pary)

        X, y = zip(*pary)

        print(f"{nazwa}: {ile} przykladow na klase")

        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.long)

        return X, y

    X_train, y_train = zrob_tensor(dane["train"], "Trening")
    X_val, y_val = zrob_tensor(dane["val"], "Walidacja")

    return X_train, y_train, X_val, y_val


def augmentuj(batch):
    # Obroty i odbicia obrazu
    batch = torch.rot90(batch, random.randint(0, 3), dims=(2, 3))

    if random.random() < 0.5:
        batch = torch.flip(batch, dims=(2,))

    if random.random() < 0.5:
        batch = torch.flip(batch, dims=(3,))

    return batch


def trenuj_siec():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    urzadzenie = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Urzadzenie: {urzadzenie}")

    X_train, y_train, X_val, y_val = przygotuj_zbior_danych()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=128,
        shuffle=False
    )

    # ResNet-18 z trzema klasami wyjsciowymi
    model = utworz_model_resnet().to(urzadzenie)

    kryterium = nn.CrossEntropyLoss()

    optymalizator = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01
    )

    najlepsza_strata = float("inf")
    najlepsze_wagi = None

    historia = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    epoki = 15

    print(f"\n[2/3] Rozpoczynam trening ResNet-18 ({epoki} epok)...")

    for epoka in range(epoki):
        model.train()

        suma = 0.0
        poprawne = 0
        wszystkie = 0

        for bx, by in train_loader:
            bx = augmentuj(bx).to(urzadzenie)
            by = by.to(urzadzenie)

            optymalizator.zero_grad()

            wyjscia = model(bx)
            strata = kryterium(wyjscia, by)

            strata.backward()
            optymalizator.step()

            suma += strata.item() * len(by)
            poprawne += (wyjscia.argmax(1) == by).sum().item()
            wszystkie += len(by)

        train_loss = suma / wszystkie
        train_acc = poprawne * 100 / wszystkie

        # Walidacja
        model.eval()

        suma = 0.0
        poprawne = 0
        wszystkie = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(urzadzenie)
                by = by.to(urzadzenie)

                wyjscia = model(bx)
                strata = kryterium(wyjscia, by)

                suma += strata.item() * len(by)
                poprawne += (wyjscia.argmax(1) == by).sum().item()
                wszystkie += len(by)

        val_loss = suma / wszystkie
        val_acc = poprawne * 100 / wszystkie

        historia["train_loss"].append(train_loss)
        historia["val_loss"].append(val_loss)
        historia["train_acc"].append(train_acc)
        historia["val_acc"].append(val_acc)

        if val_loss < najlepsza_strata:
            najlepsza_strata = val_loss
            najlepsze_wagi = {
                nazwa: waga.detach().cpu().clone()
                for nazwa, waga in model.state_dict().items()
            }

        print(
            f"Epoka {epoka + 1}/{epoki} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

    # Zapis najlepszego modelu
    torch.save(najlepsze_wagi, SCIEZKA_MODELU)

    print(f"\n[3/3] Zapisano model: {SCIEZKA_MODELU}")

    # Wykres przebiegu treningu
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(historia["train_loss"], label="Train Loss")
    plt.plot(historia["val_loss"], label="Val Loss")
    plt.title("Funkcja straty")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(historia["train_acc"], label="Train Acc")
    plt.plot(historia["val_acc"], label="Val Acc")
    plt.title("Dokladnosc klasyfikacji")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy [%]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("wykres_uczenia_v3.png", dpi=150)
    plt.close()

    print("Zapisano wykres uczenia: wykres_uczenia_v3.png")


if __name__ == "__main__":
    trenuj_siec()
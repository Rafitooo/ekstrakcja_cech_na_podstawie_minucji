import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from preprocessing import maska_odcisku, utworz_szkielet

SCIEZKA_MODELU = "model_minucje_v3.pth"
MODEL = None
ZAPISZ_PODGLAD = True


def utworz_model_resnet():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model


def wczytaj_wytrenowany_model():
    global MODEL

    if MODEL is None:
        if not os.path.exists(SCIEZKA_MODELU):
            raise FileNotFoundError("Brak model_minucje_v3.pth w folderze projektu.")

        MODEL = utworz_model_resnet()
        wagi = torch.load(SCIEZKA_MODELU, map_location="cpu", weights_only=True)
        MODEL.load_state_dict(wagi)
        MODEL.eval()

    return MODEL


def oblicz_crossing_number(okno):
    p = [
        okno[0, 1], okno[0, 2], okno[1, 2], okno[2, 2],
        okno[2, 1], okno[2, 0], okno[1, 0], okno[0, 0]
    ]

    p = [int(w > 0) for w in p]
    return sum(abs(p[i] - p[(i + 1) % 8]) for i in range(8)) // 2


def policz_odnogi(szkielet, x, y):
    # Usuwamy srodek punktu i sprawdzamy, ile linii od niego odchodzi
    r = 8
    fragment = szkielet[y-r:y+r+1, x-r:x+r+1].copy()

    if fragment.shape != (17, 17):
        return 0, 0

    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    dystans = np.sqrt(xx**2 + yy**2)

    fragment[dystans <= 1.5] = 0

    _, etykiety, pola, _ = cv2.connectedComponentsWithStats(
        fragment, connectivity=8
    )

    blisko = set(np.unique(
        etykiety[(dystans > 1.5) & (dystans <= 3.2)]
    )) - {0}

    dalej = set(np.unique(
        etykiety[(dystans >= 4) & (dystans <= 6)]
    )) - {0}

    najdalej = set(np.unique(
        etykiety[(dystans >= 6) & (dystans <= 8)]
    )) - {0}

    odnogi = sum(
        pola[i, cv2.CC_STAT_AREA] >= 3
        for i in blisko & dalej
    )

    dlugie = sum(
        pola[i, cv2.CC_STAT_AREA] >= 4
        for i in blisko & najdalej
    )

    return odnogi, dlugie


def surowi_kandydaci(szkielet, maska, filtr_galezi=True):
    obraz = (szkielet > 0).astype(np.uint8)
    h, w = obraz.shape

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    liczba_sasiadow = cv2.filter2D(obraz, cv2.CV_8U, kernel)

    kernel_maski = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (21, 21)
    )
    wewnatrz = cv2.erode(maska, kernel_maski)

    ys, xs = np.where((obraz > 0) & (wewnatrz > 0))
    punkty = []

    for y, x in zip(ys, xs):
        if x < 12 or y < 12 or x >= w - 12 or y >= h - 12:
            continue

        cn = oblicz_crossing_number(obraz[y-1:y+2, x-1:x+2])

        if not filtr_galezi:
            if cn in (1, 3):
                punkty.append((int(x), int(y), cn))
            continue

        if cn == 1:
            odnogi, dlugie = policz_odnogi(obraz, x, y)

            if odnogi == 1 and dlugie == 1:
                punkty.append((int(x), int(y), 1))

        elif cn >= 3 or (cn == 2 and liczba_sasiadow[y, x] >= 4):
            odnogi, dlugie = policz_odnogi(obraz, x, y)

            # Trzy odnogi, z czego przynajmniej dwie sa dluzsze
            if odnogi == 3 and dlugie >= 2:
                punkty.append((int(x), int(y), 3))

    return punkty


def _kandydaci_z_ocena(szkielet, obraz_std):
    maska = maska_odcisku(obraz_std)
    surowe = surowi_kandydaci(szkielet, maska)

    # Sprawdzamy, czy punkty wystepuja przy innych ustawieniach binaryzacji
    warianty = [
        (utworz_szkielet(obraz_std, maska, c) > 0).astype(np.uint8)
        for c in (3, 7)
    ]

    wynik = []
    obraz = (szkielet > 0).astype(np.uint8)

    for x, y, cn in surowe:
        _, dlugie = policz_odnogi(obraz, x, y)
        potwierdzenia = 0

        for wariant in warianty:
            znaleziono = False

            for yy in range(y-3, y+4):
                for xx in range(x-3, x+4):
                    if wariant[yy, xx] == 0:
                        continue

                    cn2 = oblicz_crossing_number(
                        wariant[yy-1:yy+2, xx-1:xx+2]
                    )

                    if cn == 1 and cn2 == 1:
                        znaleziono = True
                        break

                    if cn == 3 and cn2 >= 2:
                        odnogi2, dlugie2 = policz_odnogi(wariant, xx, yy)

                        if odnogi2 == 3 and dlugie2 >= 2:
                            znaleziono = True
                            break

                if znaleziono:
                    break

            if znaleziono:
                potwierdzenia += 1

        if cn == 1 and potwierdzenia == 2:
            wynik.append((x, y, cn, dlugie, potwierdzenia))

        elif cn == 3 and (dlugie == 3 or potwierdzenia == 2):
            wynik.append((x, y, cn, dlugie, potwierdzenia))

    return wynik


def znajdz_kandydatow(szkielet, obraz_std):
    # Zachowujemy format zgodny z train.py
    return [
        (x, y, cn)
        for x, y, cn, _, _ in _kandydaci_z_ocena(szkielet, obraz_std)
    ]


def zapisz_wizualizacje(szkielet, kandydaci, minucje):
    # Kandydaci na minucje przed klasyfikacja CNN
    obraz = cv2.cvtColor(szkielet, cv2.COLOR_GRAY2BGR)

    for x, y, *_ in kandydaci:
        cv2.circle(obraz, (x, y), 2, (0, 220, 220), 1)

    cv2.imwrite("kandydaci_na_szkielecie.png", obraz)

    # Koncowe minucje po klasyfikacji CNN
    obraz = cv2.cvtColor(szkielet, cv2.COLOR_GRAY2BGR)

    for m in minucje:
        x, y = m["x"], m["y"]

        if m["type"] == "ending":
            kolor = (0, 200, 0)
        else:
            kolor = (0, 0, 235)

        cv2.rectangle(
            obraz,
            (x-4, y-4),
            (x+4, y+4),
            kolor,
            1
        )

    cv2.imwrite("minucje_na_szkielecie.png", obraz)


def wykryj_minucje(szkielet, obraz_std, prog=0.90):
    kandydaci = _kandydaci_z_ocena(szkielet, obraz_std)

    if not kandydaci:
        print("[DETEKCJA] Brak kandydatow.")

        if ZAPISZ_PODGLAD:
            zapisz_wizualizacje(szkielet, [], [])

        return []

    # CNN otrzymuje wycinki bezposrednio ze szkieletu
    obraz = (szkielet > 0).astype(np.float32)

    wycinki = np.stack([
        obraz[y-7:y+8, x-7:x+8]
        for x, y, *_ in kandydaci
    ])

    tensor = torch.from_numpy(wycinki).unsqueeze(1)
    model = wczytaj_wytrenowany_model()

    with torch.no_grad():
        wyjscia = model(tensor)
        wynik_cnn = torch.softmax(wyjscia, dim=1).numpy()

    wybrane = []

    for (x, y, cn, dlugie, stabilnosc), p in zip(kandydaci, wynik_cnn):
        klasa = 1 if cn == 1 else 2
        pewnosc = float(p[klasa])

        # Rozdwojenia z krotsza odnoga wymagaja wiekszej pewnosci CNN
        if cn == 1:
            wymagany_prog = prog
        elif dlugie == 2:
            wymagany_prog = 0.96
        else:
            wymagany_prog = 0.93

        if int(np.argmax(p)) != klasa:
            continue

        if pewnosc < wymagany_prog:
            continue

        wybrane.append((x, y, cn, dlugie, stabilnosc, pewnosc))

    # Jedno rozgalezienie moze zajmowac kilka sasiednich pikseli
    wybrane.sort(key=lambda p: (p[3], p[4], p[5]), reverse=True)

    minucje = []

    for x, y, cn, *_ in wybrane:
        duplikat = any(
            np.hypot(x - m["x"], y - m["y"]) < 7
            for m in minucje
        )

        if duplikat:
            continue

        minucje.append({
            "x": x,
            "y": y,
            "type": "ending" if cn == 1 else "bifurcation"
        })

    zakonczenia = sum(m["type"] == "ending" for m in minucje)
    rozdwojenia = len(minucje) - zakonczenia

    print(f"[DETEKCJA] Kandydaci: {len(kandydaci)}")
    print(f"[DETEKCJA] Zakonczenia: {zakonczenia}")
    print(f"[DETEKCJA] Rozdwojenia: {rozdwojenia}")

    if ZAPISZ_PODGLAD:
        zapisz_wizualizacje(szkielet, kandydaci, minucje)

    return minucje
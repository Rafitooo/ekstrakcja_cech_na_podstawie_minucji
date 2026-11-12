import cv2
import numpy as np
from skimage.morphology import skeletonize

ROZMIAR = (260, 280)


def wczytaj_obraz(bajty):
    obraz = cv2.imdecode(np.frombuffer(bajty, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if obraz is None:
        raise ValueError("Nie mozna odczytac obrazu.")

    if obraz.ndim == 2:
        return obraz

    if obraz.shape[2] == 4:
        alfa = obraz[:, :, 3:4].astype(np.float32) / 255.0
        kolor = (obraz[:, :, :3] * alfa + 255 * (1 - alfa)).astype(np.uint8)
        return cv2.cvtColor(kolor, cv2.COLOR_BGR2GRAY)

    return cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)


def wyczysc_ramke_skanera(obraz):
    czysty = obraz.copy()
    czysty[:2, :] = 255
    czysty[-2:, :] = 255
    czysty[:, :2] = 255
    czysty[:, -2:] = 255
    return cv2.resize(czysty, ROZMIAR, interpolation=cv2.INTER_LINEAR)


def maska_odcisku(obraz):
    # Oceniamy lokalny kontrast i kierunek linii papilarnych
    obraz = obraz.astype(np.float32)
    srednia = cv2.GaussianBlur(obraz, (0, 0), 6)
    wariancja = cv2.GaussianBlur(obraz * obraz, (0, 0), 6) - srednia * srednia
    kontrast = np.sqrt(np.maximum(wariancja, 0))

    dx = cv2.Sobel(obraz, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(obraz, cv2.CV_32F, 0, 1, ksize=3)

    xx = cv2.GaussianBlur(dx * dx, (0, 0), 6)
    yy = cv2.GaussianBlur(dy * dy, (0, 0), 6)
    xy = cv2.GaussianBlur(dx * dy, (0, 0), 6)

    zgodnosc = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2) / (xx + yy + 1e-5)

    maska = ((kontrast > 12) & (zgodnosc > 0.5)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, kernel)

    ile, labels, pola, _ = cv2.connectedComponentsWithStats(maska, 8)
    if ile <= 1:
        return np.zeros_like(maska)

    najwiekszy = 1 + np.argmax(pola[1:, cv2.CC_STAT_AREA])
    maska = np.where(labels == najwiekszy, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    maska = cv2.erode(maska, kernel)
    return maska


def utworz_szkielet(obraz, maska, parametr_c=5):
    odszumiony = cv2.GaussianBlur(obraz, (3, 3), 0)

    binarny = cv2.adaptiveThreshold(
        odszumiony,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19,
        parametr_c
    )

    binarny[maska == 0] = 0

    # Usuwamy drobne elementy, ktore moga byc szumem
    ile, labels, pola, _ = cv2.connectedComponentsWithStats(binarny, 8)
    for nr in range(1, ile):
        if pola[nr, cv2.CC_STAT_AREA] < 12:
            binarny[labels == nr] = 0

    # Scienianie linii do szerokosci jednego piksela
    return skeletonize(binarny > 0).astype(np.uint8) * 255


def przetworz_obraz(obraz):
    if obraz is None or obraz.size == 0 or obraz.ndim != 2:
        raise ValueError("Nieprawidlowy obraz odcisku palca.")

    if obraz.std() < 10:
        raise ValueError("Obraz jest zbyt jednolity, aby wykryc linie.")

    obraz_std = wyczysc_ramke_skanera(obraz)
    maska = maska_odcisku(obraz_std)

    if cv2.countNonZero(maska) < 700:
        raise ValueError("Nie znaleziono obszaru odcisku palca.")

    szkielet = utworz_szkielet(obraz_std, maska)
    return obraz_std, szkielet
import base64
import logging
import cv2

from flask import Flask, request, jsonify, render_template, send_file
from preprocessing import przetworz_obraz, wczytaj_obraz
from model_cnn import wykryj_minucje
from features import utworz_wektor_cech, eksportuj_do_json, eksportuj_do_csv
from database import init_db, zapisz_do_bazy, pobierz_historie_z_bazy

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

OSTATNI_WYNIK = None
logging.basicConfig(level=logging.INFO)


def obraz_na_base64(obraz):
    sukces, bufor = cv2.imencode(".png", obraz)
    if not sukces:
        raise ValueError("Nie udalo sie zapisac obrazu.")
    return base64.b64encode(bufor).decode("utf-8")


def narysuj_minucje(oryginal, minucje):
    h, w = oryginal.shape
    kolorowy = cv2.cvtColor(oryginal, cv2.COLOR_GRAY2BGR)

    for punkt in minucje:
        x = round(punkt["x"] * w / 260)
        y = round(punkt["y"] * h / 280)

        kolor = (0, 200, 0) if punkt["type"] == "ending" else (0, 0, 235)
        cv2.rectangle(kolorowy, (x - 2, y - 2), (x + 2, y + 2), kolor, 1)

    return kolorowy


@app.route("/")
def strona_glowna():
    init_db()
    return render_template("index.html")


@app.route("/api/preview", methods=["POST"])
def api_preview():
    if "file" not in request.files:
        return jsonify({"error": "Nie wybrano pliku."}), 400

    try:
        obraz = wczytaj_obraz(request.files["file"].read())
        return jsonify({"oryginal_b64": obraz_na_base64(obraz)})
    except ValueError as blad:
        return jsonify({"error": str(blad)}), 400


@app.route("/api/extract", methods=["POST"])
def api_extract():
    global OSTATNI_WYNIK

    if "file" not in request.files:
        return jsonify({"error": "Nie wybrano pliku."}), 400

    plik = request.files["file"]

    try:
        obraz = wczytaj_obraz(plik.read())

        # 1. Preprocessing
        obraz_std, szkielet = przetworz_obraz(obraz)

        # 2. Wykrywanie minucji
        minucje = wykryj_minucje(szkielet, obraz_std)

        # 3. Wektor cech
        wynik = utworz_wektor_cech(plik.filename, minucje)

        # 4. Zapis do bazy
        zapisz_do_bazy(wynik)
        OSTATNI_WYNIK = wynik

        # 5. Wizualizacja na oryginalnym obrazie
        wizualizacja = narysuj_minucje(obraz, minucje)

        return jsonify({
            "status": "success",
            "data": wynik,
            "oryginal_b64": obraz_na_base64(obraz),
            "szkielet_b64": obraz_na_base64(szkielet),
            "wizualizacja_b64": obraz_na_base64(wizualizacja),
            "historia": pobierz_historie_z_bazy()
        })

    except ValueError as blad:
        return jsonify({"error": str(blad)}), 400

    except Exception:
        app.logger.exception("Blad analizy pliku %s", plik.filename)
        return jsonify({
            "error": "Blad analizy. Szczegoly znajduja sie w terminalu."
        }), 500


@app.route("/api/export/<format_pliku>")
def api_export(format_pliku):
    if OSTATNI_WYNIK is None:
        return jsonify({"error": "Najpierw przeanalizuj odcisk."}), 400

    if format_pliku == "json":
        return send_file(eksportuj_do_json(OSTATNI_WYNIK), as_attachment=True)

    if format_pliku == "csv":
        return send_file(eksportuj_do_csv(OSTATNI_WYNIK), as_attachment=True)

    return jsonify({"error": "Nieobslugiwany format."}), 400


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
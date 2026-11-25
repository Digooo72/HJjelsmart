from flask import Flask, render_template, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2 import sql
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)
CORS(app)

PORT = 6720

# PostgreSQL kapcsolat
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="jelsmart_db",
            user="jelsmart",
            password="ihne",
            host="192.168.0.18",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Adatbázis hiba: {e}")
        return None


# Tükör program inicializálása
model_path = "/home/ihne/jelsmart-server/programs/mirror/model.p"
if os.path.exists(model_path):
    try:
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        label_encoder = model_dict['label_encoder']
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(min_detection_confidence=0.3)
    except Exception as e:
        print(f"Modell betöltési hiba: {e}")
        model = None
        label_encoder = None
        hands = None
else:
    print(f"Modell fájl nem található: {model_path}")
    model = None
    label_encoder = None
    hands = None


# Főoldal
@app.route('/')
def index():
    return render_template('index.html')


# Leckék listázása
@app.route('/lecke')
def lessons():
    conn = get_db_connection()
    if conn is None:
        return render_template('lessons.html', lessons=[], error="Adatbázis nem elérhető")
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, title, description FROM jelapp_lesson")
        lessons = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('lessons.html', lessons=lessons)
    except Exception as e:
        return render_template('lessons.html', lessons=[], error=f"Hiba: {str(e)}")


# Tükör API (vázlat, később WebRTC)
@app.route('/api/tukor', methods=['GET'])
def tukor_api():
    if model is None or label_encoder is None:
        return jsonify({"error": "Modell nem elérhető"}), 500
    return jsonify({"message": "Tükör API kész, WebRTC integráció később"})


# Tükör oldal
@app.route('/tukor')
def tukor():
    return render_template('tukor.html')


# Demo mount
MOUNT_DEMO = os.environ.get("MOUNT_DEMO", "true").lower() == "true"
application = app

if MOUNT_DEMO:
    try:
        from werkzeug.middleware.dispatcher import DispatcherMiddleware
        from demo.endpoints import app as demo_app

        application = DispatcherMiddleware(app, {
            "/demo": demo_app,
        })
    except Exception as e:
        print(f"[app] demo mount kihagyva, hiba történt: {e}")
        application = app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
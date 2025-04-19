import os
import sqlite3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# === CONFIG ===
UPLOAD_FOLDER     = 'static/uploads'
ENREGISTRE_FOLDER = 'static/enregistres'
DB_PATH           = 'chat_database.db'
MODEL_PATH        = 'chat_recognition_model.h5'
ALLOWED_EXT       = {'png','jpg','jpeg'}

# Création des dossiers si nécessaire
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENREGISTRE_FOLDER, exist_ok=True)

# === FLASK APP ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER']     = UPLOAD_FOLDER
app.config['ENREGISTRE_FOLDER'] = ENREGISTRE_FOLDER

# === CHARGEMENT DU MODÈLE ===
model = load_model(MODEL_PATH)

# === BASE DE DONNÉES ===
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute('''
      CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        image_path TEXT,
        created_at DATETIME
      )
    ''')
    db.commit()
    db.close()

init_db()

# === UTILITAIRES ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def send_email(to_email, chat_name):
    FROM = 'alexandre@amconsulting-formation.com'
    PWD  = 'gnoutos01'
    msg = MIMEMultipart()
    msg['From']    = FROM
    msg['To']      = to_email
    msg['Subject'] = "🐱 Votre chat a été détecté !"
    body = f"Bonjour,\n\nNous avons détecté votre chat nommé '{chat_name}'."
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP_SSL('ssl0.ovh.net', 465)
        server.login(FROM, PWD)
        server.sendmail(FROM, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("Erreur SMTP :", e)
        return False

def predict_is_cat(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    score = model.predict(arr)[0][0]
    return score > 0.5, float(score)

# === ROUTES ===

# Page d'accueil (formulaire upload + lien vers enregistrement)
@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

# Traitement du formulaire d'upload
@app.route('/upload', methods=['POST'])
def upload():
    file  = request.files.get('image')
    name  = request.form.get('name')
    email = request.form.get('email')
    if not file or not allowed_file(file.filename):
        return redirect(url_for('index'))

    # Sauvegarde dans uploads
    filename = secure_filename(f"{datetime.utcnow().timestamp()}_{file.filename}")
    path_up  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path_up)

    # Détection
    is_cat, score = predict_is_cat(path_up)
    message = "Chat détecté !" if is_cat else "Aucun chat détecté"

    # Si chat, on enregistre + envoi mail + copie
    if is_cat:
        # copier dans enregistres
        path_reg = os.path.join(app.config['ENREGISTRE_FOLDER'], filename)
        Image.open(path_up).save(path_reg)

        # stocker en base
        db = get_db()
        db.execute(
            "INSERT INTO chats(name,email,image_path,created_at) VALUES (?,?,?,?)",
            (name, email, path_reg, datetime.now())
        )
        db.commit()
        db.close()

        # envoi d'email
        send_email(email, name)

    # Affichage du résultat
    return render_template(
        'resultat.html',
        chat_detecte=is_cat,
        chat_name=name,
        image_url=url_for('static', filename='uploads/'+filename),
        message=message,
        confidence=f"{score:.2f}"
    )

# Page dédiée pour enregistrer un chat existant (optionnel)
@app.route('/enregistrer', methods=['GET','POST'])
def enregistrer():
    if request.method == 'POST':
        file  = request.files.get('image')
        name  = request.form.get('chat_name')
        email = request.form.get('owner_email')
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{datetime.utcnow().timestamp()}_{file.filename}")
            path_reg = os.path.join(app.config['ENREGISTRE_FOLDER'], filename)
            file.save(path_reg)
            db = get_db()
            db.execute(
                "INSERT INTO chats(name,email,image_path,created_at) VALUES (?,?,?,?)",
                (name, email, path_reg, datetime.now())
            )
            db.commit()
            db.close()
            return redirect(url_for('index'))
    return render_template('enregistrer.html')

# Liste des chats enregistrés
@app.route('/chats', methods=['GET'])
def liste_chats():
    db = get_db()
    cur = db.execute("SELECT name, email, image_path FROM chats ORDER BY created_at DESC")
    chats = cur.fetchall()
    db.close()
    return render_template('liste.html', chats=chats)

if __name__ == '__main__':
    app.run(debug=True)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
import json
from datetime import datetime
from functools import wraps

from flask import (
    Flask, request, render_template, Response,
    jsonify, session, redirect, url_for, flash
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from models.cnn_model import PneumoniaCNN
from utils import preprocess_image, generate_gradcam, train_data
import pdfkit

# NEW: import all DB helpers (auth still uses JSON)
from database import (
    init_db, migrate_from_json,
    add_patient, get_all_patients, get_patient,
    update_patient_feedback, update_patient_report,
    delete_patient as db_delete_patient,
    get_model_stats,
    add_chat_message, get_all_chat_messages,
)

# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)
app.secret_key = "federated_pneumonia_super_secret"

UPLOAD_FOLDER   = "static/uploads"
HEATMAP_FOLDER  = "static/heatmaps"
REPORTS_FOLDER  = "static/reports"
PROFILES_FOLDER = "static/profiles"
DOCTORS_DB_FILE = "doctors_auth.json"

for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER, REPORTS_FOLDER, PROFILES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# =========================================================
# DOCTOR AUTH  (stays JSON — as requested)
# =========================================================

DOCTOR_ACCOUNTS = {
    "amayavk118@gmail.com":              "passwordamaya",
    "nandhanakmurali123@gmail.com":      "passwordnandhu",
    "adhithyavs3@gmail.com":             "passwordadhi",
    "aidaelizabathvarghese2003@gmail.com":"passwordaida"
}


def load_json_db(file_path, default):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                return default
    return default


def save_json_db(file_path, db):
    with open(file_path, 'w') as f:
        json.dump(db, f, indent=4)


def initialize_doctors():
    docs    = load_json_db(DOCTORS_DB_FILE, {})
    changed = False
    for em, initial_pw in DOCTOR_ACCOUNTS.items():
        if em not in docs:
            docs[em] = {
                "email":       em,
                "username":    em.split("@")[0],
                "password":    generate_password_hash(initial_pw.strip()),
                "profile_pic": ""
            }
            changed = True
    if changed:
        save_json_db(DOCTORS_DB_FILE, docs)


initialize_doctors()

# =========================================================
# ONE-TIME JSON → SQLite MIGRATION
# =========================================================
migrate_from_json()

# =========================================================
# MODEL LOAD
# =========================================================

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_data.classes) if hasattr(train_data, 'classes') else 3
model       = PneumoniaCNN(num_classes=num_classes).to(device)

try:
    model.load_state_dict(
        torch.load("models/best_model.pth", map_location=device, weights_only=True)
    )
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️  Could not load model weights: {e}")

model.eval()

# =========================================================
# AUTH HELPERS
# =========================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.context_processor
def inject_doctor():
    if 'logged_in' in session:
        email = session.get('doctor_email')
        docs  = load_json_db(DOCTORS_DB_FILE, {})
        return dict(current_doctor=docs.get(email))
    return dict(current_doctor=None)

# =========================================================
# AUTH ROUTES
# =========================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        identifier = request.form.get('email', '').strip()
        password   = request.form.get('password', '')
        docs       = load_json_db(DOCTORS_DB_FILE, {})
        logged_in_email = None

        if identifier in docs:
            if check_password_hash(docs[identifier]['password'], password):
                logged_in_email = identifier
        else:
            for email, profile in docs.items():
                if profile.get('username') == identifier:
                    if check_password_hash(profile['password'], password):
                        logged_in_email = email
                    break

        if logged_in_email:
            session['logged_in']     = True
            session['doctor_email']  = logged_in_email
            return redirect(url_for('index'))
        else:
            error = "Invalid credentials. Only pre-authorized medical staff are permitted."

    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    error   = None
    success = None
    email   = session.get('doctor_email')
    docs    = load_json_db(DOCTORS_DB_FILE, {})
    profile = docs.get(email)

    if request.method == 'POST':
        if 'update_password' in request.form:
            current_pw = request.form.get('current_password')
            new_pw     = request.form.get('new_password')
            if check_password_hash(profile['password'], current_pw):
                profile['password'] = generate_password_hash(new_pw)
                success = "Password updated successfully."
            else:
                error = "Incorrect current password."

        elif 'update_username' in request.form:
            new_user = request.form.get('new_username', '').strip()
            taken    = any(
                v.get('username') == new_user and k != email
                for k, v in docs.items()
            )
            if taken:
                error = "Username already taken."
            else:
                profile['username'] = new_user
                success = "Username successfully updated."

        elif 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file.filename:
                filename = secure_filename(f"{profile['username']}_{file.filename}")
                filepath = os.path.join(PROFILES_FOLDER, filename)
                file.save(filepath)
                profile['profile_pic'] = f"profiles/{filename}"
                success = "Profile picture updated."

        docs[email] = profile
        save_json_db(DOCTORS_DB_FILE, docs)

    return render_template('account.html', error=error, success=success, profile=profile)

# =========================================================
# MAIN / SCAN UPLOAD
# =========================================================

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        file     = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        patient_name = request.form.get("patient_name", "Unknown")
        age          = request.form.get("age", "")
        gender       = request.form.get("gender", "")
        phone        = request.form.get("phone", "")
        height       = request.form.get("height", "")
        weight       = request.form.get("weight", "")

        image_tensor = preprocess_image(filepath).to(device)
        classes      = getattr(train_data, 'classes', ["Pneumonia", "COVID-19", "Tuberculosis"])

        with torch.no_grad():
            outputs = model(image_tensor)
            probs   = torch.sigmoid(outputs)[0]

        detected        = []
        max_prob        = 0.0
        predicted_class = torch.argmax(probs).item()

        for i, prob in enumerate(probs):
            if prob > 0.5:
                detected.append(classes[i])
            if prob.item() > max_prob:
                max_prob = prob.item()

        prediction = ", ".join(detected) if detected else "NORMAL"
        confidence = max_prob

        heatmap_path = os.path.join(HEATMAP_FOLDER, f"heatmap_{filename}")
        generate_gradcam(model, image_tensor, predicted_class, filepath, heatmap_path)

        email    = session.get('doctor_email')
        docs     = load_json_db(DOCTORS_DB_FILE, {})
        consultant_name = docs.get(email, {}).get('username', 'Unknown')
        current_date    = datetime.now().strftime("%d %b %Y, %H:%M")

        patient_data = {
            "id":         str(uuid.uuid4()),
            "name":       patient_name,
            "age":        age,
            "gender":     gender,
            "phone":      phone,
            "height":     height,
            "weight":     weight,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "filename":   filename,
            "heatmap":    f"heatmaps/heatmap_{filename}",
            "report_url": None,
            "feedback":   "Pending",
            "consultant": consultant_name,
            "date":       current_date,
        }

        add_patient(patient_data)   # ← SQLite

        return render_template("result.html", patient=patient_data)

    return render_template("index.html")

# =========================================================
# PATIENT API
# =========================================================

@app.route("/api/patients")
@login_required
def api_patients():
    return jsonify(get_all_patients())


@app.route("/api/patients/<patient_id>", methods=["DELETE"])
@login_required
def delete_patient_route(patient_id):
    patient = db_delete_patient(patient_id)   # removes from DB + returns record
    if not patient:
        return jsonify({"success": False, "error": "Patient not found"}), 404

    # Clean up associated files
    for path in [
        os.path.join(UPLOAD_FOLDER,  patient.get("filename", "")),
        os.path.join("static",       patient.get("heatmap",  "")),
    ]:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"File cleanup error: {e}")

    report_url = patient.get("report_url")
    if report_url:
        report_path = os.path.join("static", report_url.replace("/static/", ""))
        try:
            if os.path.exists(report_path):
                os.remove(report_path)
        except Exception as e:
            print(f"Report cleanup error: {e}")

    return jsonify({"success": True})

# =========================================================
# REPORT DOWNLOAD
# =========================================================

@app.route("/download/<patient_id>")
@login_required
def download_report(patient_id):
    patient = get_patient(patient_id)
    if not patient:
        return "Patient not found", 404

    filename    = patient["filename"]
    upload_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
    heatmap_path = os.path.abspath(
        os.path.join(HEATMAP_FOLDER, f"heatmap_{filename}")
    )

    upload_url  = "file:///" + upload_path.replace("\\", "/")
    heatmap_url = "file:///" + heatmap_path.replace("\\", "/")

    rendered = render_template(
        "report_template.html",
        patient=patient,
        upload_path=upload_url,
        heatmap_path=heatmap_url,
    )

    config  = pdfkit.configuration(
        wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    )
    options = {"enable-local-file-access": None}
    pdf     = pdfkit.from_string(rendered, False, configuration=config, options=options)

    safe_name       = patient["name"].replace(" ", "_").lower()
    report_filename = f"{safe_name}_report_{patient_id[:8]}.pdf"
    report_path     = os.path.join(REPORTS_FOLDER, report_filename)

    with open(report_path, "wb") as f:
        f.write(pdf)

    update_patient_report(patient_id, f"/static/reports/{report_filename}")  # ← SQLite

    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment;filename={report_filename}"}
    )

# =========================================================
# FEEDBACK API
# =========================================================

@app.route("/api/feedback/<patient_id>", methods=["POST"])
@login_required
def submit_feedback(patient_id):
    is_correct = request.json.get('is_correct')
    feedback   = "Correct" if is_correct else "Incorrect"
    update_patient_feedback(patient_id, feedback)   # ← SQLite
    return jsonify({"success": True, "message": "Feedback recorded successfully"})

# =========================================================
# MODEL STATS API
# =========================================================

@app.route("/api/model_stats")
@login_required
def model_stats_route():
    return jsonify(get_model_stats())   # ← SQLite

# =========================================================
# CHAT API
# =========================================================

@app.route("/api/chat", methods=["GET"])
@login_required
def get_chat():
    return jsonify(get_all_chat_messages())   # ← SQLite


@app.route("/api/chat", methods=["POST"])
@login_required
def post_chat():
    text = request.json.get('text', '').strip()
    if not text:
        return jsonify({"error": "Message cannot be empty"}), 400

    email = session.get('doctor_email')
    docs  = load_json_db(DOCTORS_DB_FILE, {})
    sender_name = docs.get(email, {}).get('username', 'Unknown')

    msg = {
        "id":           str(uuid.uuid4()),
        "sender_email": email,
        "sender_name":  sender_name,
        "text":         text,
        "timestamp":    datetime.now().strftime("%d %b %Y, %H:%M"),
    }

    add_chat_message(msg)   # ← SQLite
    return jsonify({"success": True, "message": msg})

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
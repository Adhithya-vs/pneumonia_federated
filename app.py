import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template, Response, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from models.cnn_model import PneumoniaCNN
from utils import preprocess_image, generate_gradcam
import pdfkit
import uuid
import json
from functools import wraps
from datetime import datetime

app = Flask(__name__)
app.secret_key = "federated_pneumonia_super_secret"

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
REPORTS_FOLDER = "static/reports"
PATIENTS_DB_FILE = "patients_db.json"
DOCTORS_DB_FILE = "doctors_auth.json"
CHAT_DB_FILE = "chat_db.json"
PROFILES_FOLDER = "static/profiles"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(PROFILES_FOLDER, exist_ok=True)

# Add new doctors here by providing their email and an initial connection password
DOCTOR_ACCOUNTS = {
    "amayavk118@gmail.com": "passwordamaya",
    "nandhanakmurali123@gmail.com": "passwordnandhu",
    "adhithyavs3@gmail.com": "passwordadhi",
    "aidaelizabathvarghese2003@gmail.com": "passwordaida"
}

def load_json_db(file_path, default):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except:
                return default
    return default

def save_json_db(file_path, db):
    with open(file_path, 'w') as f:
        json.dump(db, f, indent=4)

# Initialize authorized doctors
def initialize_doctors():
    docs = load_json_db(DOCTORS_DB_FILE, {})
    changed = False
    for em, initial_pw in DOCTOR_ACCOUNTS.items():
        if em not in docs:
            docs[em] = {
                "email": em,
                "username": em.split("@")[0],
                "password": generate_password_hash(initial_pw.strip()), 
                "profile_pic": ""
            }
            changed = True
    if changed:
        save_json_db(DOCTORS_DB_FILE, docs)

initialize_doctors()

from utils import train_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_classes = len(train_data.classes) if hasattr(train_data, 'classes') else 3
model = PneumoniaCNN(num_classes=num_classes).to(device)
try:
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
except:
    pass
model.eval()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def inject_doctor():
    if 'logged_in' in session:
        email = session.get('doctor_email')
        docs = load_json_db(DOCTORS_DB_FILE, {})
        return dict(current_doctor=docs.get(email))
    return dict(current_doctor=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        identifier = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        docs = load_json_db(DOCTORS_DB_FILE, {})
        logged_in_email = None

        if identifier in docs:
            match = check_password_hash(docs[identifier]['password'], password)
            if match:
                logged_in_email = identifier
        else:
            for email, profile in docs.items():
                if profile.get('username') == identifier:
                    match = check_password_hash(profile['password'], password)
                    if match:
                        logged_in_email = email
                    break

        if logged_in_email:
            session['logged_in'] = True
            session['doctor_email'] = logged_in_email
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
    error = None
    success = None
    email = session.get('doctor_email')
    docs = load_json_db(DOCTORS_DB_FILE, {})
    profile = docs.get(email)

    if request.method == 'POST':
        if 'update_password' in request.form:
            current_pw = request.form.get('current_password')
            new_pw = request.form.get('new_password')
            if check_password_hash(profile['password'], current_pw):
                profile['password'] = generate_password_hash(new_pw)
                success = "Password updated successfully."
            else:
                error = "Incorrect current password."
                
        elif 'update_username' in request.form:
            new_user = request.form.get('new_username').strip()
            taken = False
            for k, v in docs.items():
                if v.get('username') == new_user and k != email:
                    taken = True
            if taken:
                error = "Username already taken."
            else:
                profile['username'] = new_user
                success = "Username successfully updated."

        elif 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file.filename != '':
                filename = secure_filename(f"{profile['username']}_{file.filename}")
                filepath = os.path.join(PROFILES_FOLDER, filename)
                file.save(filepath)
                profile['profile_pic'] = f"profiles/{filename}"
                success = "Profile picture updated."

        docs[email] = profile
        save_json_db(DOCTORS_DB_FILE, docs)

    return render_template('account.html', error=error, success=success, profile=profile)

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        patient_name = request.form.get("patient_name", "Unknown")
        age = request.form.get("age", "")
        gender = request.form.get("gender", "")
        phone = request.form.get("phone", "")
        height = request.form.get("height", "")
        weight = request.form.get("weight", "")

        image_tensor = preprocess_image(filepath).to(device)

        classes = getattr(train_data, 'classes', ["Pneumonia", "COVID-19", "Tuberculosis"])

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.sigmoid(outputs)[0]
            
        detected = []
        max_prob = 0
        predicted_class = torch.argmax(probs).item()  # For GradCAM target
        
        for i, prob in enumerate(probs):
            if prob > 0.5:
                detected.append(classes[i])
            if prob > max_prob:
                max_prob = prob.item()

        prediction = ", ".join(detected) if detected else "NORMAL"
        confidence = max_prob

        heatmap_path = os.path.join(HEATMAP_FOLDER, f"heatmap_{filename}")
        generate_gradcam(model, image_tensor, predicted_class, filepath, heatmap_path)

        patient_id = str(uuid.uuid4())
        
        email = session.get('doctor_email')
        docs = load_json_db(DOCTORS_DB_FILE, {})
        consultant_name = docs.get(email, {}).get('username', 'Unknown')
        current_date = datetime.now().strftime("%d %b %Y, %H:%M")

        patient_data = {
            "id": patient_id,
            "name": patient_name,
            "age": age,
            "gender": gender,
            "phone": phone,
            "height": height,
            "weight": weight,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "filename": filename,
            "heatmap": f"heatmaps/heatmap_{filename}",
            "report_url": None,
            "feedback": "Pending",
            "consultant": consultant_name,
            "date": current_date
        }
        
        db = load_json_db(PATIENTS_DB_FILE, [])
        db.append(patient_data)
        save_json_db(PATIENTS_DB_FILE, db)

        return render_template(
            "result.html",
            patient=patient_data
        )

    return render_template("index.html")

@app.route("/api/patients")
@login_required
def api_patients():
    return jsonify(load_json_db(PATIENTS_DB_FILE, []))

@app.route("/download/<patient_id>")
@login_required
def download_report(patient_id):
    db = load_json_db(PATIENTS_DB_FILE, [])
    patient = next((p for p in db if p["id"] == patient_id), None)
    
    if not patient:
        return "Patient not found", 404

    filename = patient["filename"]

    upload_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
    heatmap_path = os.path.abspath(os.path.join(HEATMAP_FOLDER, f"heatmap_{filename}"))

    upload_url = "file:///" + upload_path.replace("\\", "/")
    heatmap_url = "file:///" + heatmap_path.replace("\\", "/")

    rendered = render_template(
        "report_template.html",
        patient=patient,
        upload_path=upload_url,
        heatmap_path=heatmap_url
    )

    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    options = {
        "enable-local-file-access": None
    }

    pdf = pdfkit.from_string(rendered, False, configuration=config, options=options)

    safe_name = patient["name"].replace(" ", "_").lower()
    report_filename = f"{safe_name}_report_{patient_id[:8]}.pdf"
    report_path = os.path.join(REPORTS_FOLDER, report_filename)
    
    with open(report_path, "wb") as f:
        f.write(pdf)
        
    patient["report_url"] = f"/static/reports/{report_filename}"
    save_json_db(PATIENTS_DB_FILE, db)

    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment;filename={report_filename}"})

@app.route("/api/feedback/<patient_id>", methods=["POST"])
@login_required
def submit_feedback(patient_id):
    data = request.json
    is_correct = data.get('is_correct')
    
    db = load_json_db(PATIENTS_DB_FILE, [])
    updated = False
    for p in db:
        if p["id"] == patient_id:
            p["feedback"] = "Correct" if is_correct else "Incorrect"
            updated = True
            break
            
    if updated:
        save_json_db(PATIENTS_DB_FILE, db)
        return jsonify({"success": True, "message": "Feedback recorded successfully"})
    else:
        return jsonify({"success": False, "error": "Patient not found"}), 404

@app.route("/api/model_stats")
@login_required
def get_model_stats():
    db = load_json_db(PATIENTS_DB_FILE, [])
    total = len(db)
    correct = sum(1 for p in db if p.get("feedback") == "Correct")
    incorrect = sum(1 for p in db if p.get("feedback") == "Incorrect")
    pending = total - (correct + incorrect)
    
    accuracy = 0
    if (correct + incorrect) > 0:
        accuracy = round((correct / (correct + incorrect)) * 100, 2)
        
    return jsonify({
        "total_predictions": total,
        "correct": correct,
        "incorrect": incorrect,
        "pending": pending,
        "accuracy": accuracy
    })

@app.route("/api/patients/<patient_id>", methods=["DELETE"])
@login_required
def delete_patient(patient_id):
    db = load_json_db(PATIENTS_DB_FILE, [])
    new_db = []
    deleted = False
    
    for p in db:
        if p["id"] == patient_id:
            deleted = True
            try:
                if p.get("filename"):
                    file_path = os.path.join(UPLOAD_FOLDER, p["filename"])
                    if os.path.exists(file_path):
                        os.remove(file_path)
                if p.get("heatmap"):
                    heatmap_path = os.path.join("static", p["heatmap"])
                    if os.path.exists(heatmap_path):
                        os.remove(heatmap_path)
                if p.get("report_url"):
                    report_path = os.path.join("static", p["report_url"].replace("/static/", ""))
                    if os.path.exists(report_path):
                        os.remove(report_path)
            except Exception as e:
                print(f"Error deleting files: {e}")
        else:
            new_db.append(p)
            
    if deleted:
        save_json_db(PATIENTS_DB_FILE, new_db)
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Patient not found"}), 404

@app.route("/api/chat", methods=["GET"])
@login_required
def get_chat_messages():
    chat_db = load_json_db(CHAT_DB_FILE, [])
    return jsonify(chat_db)

@app.route("/api/chat", methods=["POST"])
@login_required
def post_chat_message():
    data = request.json
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({"error": "Message cannot be empty"}), 400

    email = session.get('doctor_email')
    docs = load_json_db(DOCTORS_DB_FILE, {})
    consultant_name = docs.get(email, {}).get('username', 'Unknown')
    current_time = datetime.now().strftime("%d %b %Y, %H:%M")

    new_msg = {
        "id": str(uuid.uuid4()),
        "sender_email": email,
        "sender_name": consultant_name,
        "text": text,
        "timestamp": current_time
    }

    chat_db = load_json_db(CHAT_DB_FILE, [])
    chat_db.append(new_msg)
    save_json_db(CHAT_DB_FILE, chat_db)

    return jsonify({"success": True, "message": new_msg})

if __name__ == "__main__":
    app.run(debug=True)
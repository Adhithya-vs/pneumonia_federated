import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template, Response
from werkzeug.utils import secure_filename
from models.cnn_model import PneumoniaCNN
from utils import preprocess_image, generate_gradcam
import pdfkit

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

def severity_from_confidence(confidence):
    if confidence < 0.6:
        return "Mild"
    elif confidence < 0.8:
        return "Moderate"
    else:
        return "Severe"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        image_tensor = preprocess_image(filepath).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

        classes = ["NORMAL", "PNEUMONIA"]
        prediction = classes[predicted_class]
        severity = severity_from_confidence(confidence)

        heatmap_path = os.path.join(HEATMAP_FOLDER, f"heatmap_{filename}")
        generate_gradcam(model, image_tensor, predicted_class, filepath, heatmap_path)

        return render_template(
            "result.html",
            filename=filename,
            prediction=prediction,
            confidence=round(confidence * 100, 2),
            severity=severity,
            heatmap=f"heatmaps/heatmap_{filename}"
        )

    return render_template("index.html")

@app.route("/download/<filename>/<prediction>/<confidence>/<severity>")
def download_report(filename, prediction, confidence, severity):
    # Absolute paths for images
    upload_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
    heatmap_path = os.path.abspath(os.path.join(HEATMAP_FOLDER, f"heatmap_{filename}"))

    # Convert to file:// URLs
    upload_url = "file:///" + upload_path.replace("\\", "/")
    heatmap_url = "file:///" + heatmap_path.replace("\\", "/")

    rendered = render_template(
        "report_template.html",
        filename=filename,
        prediction=prediction,
        confidence=confidence,
        severity=severity,
        upload_path=upload_url,
        heatmap_path=heatmap_url
    )

    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    options = {
        "enable-local-file-access": None  # allow local file paths
    }

    pdf = pdfkit.from_string(rendered, False, configuration=config, options=options)

    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment;filename=report_{filename}.pdf"})

if __name__ == "__main__":
    app.run(debug=True)
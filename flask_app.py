import argparse
import io
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from flask import Flask, render_template, request, flash, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # For flashing messages

# Define upload folder and ensure it exists
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure images folder exists
IMAGES_FOLDER = os.path.join("static", "images")
os.makedirs(IMAGES_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def predict():
    result_image_url = None

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.")
            return render_template("index.html")
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.")
            return render_template("index.html")
        
        # Save the uploaded file to disk
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        # Load image with PIL (convert to RGB for consistency)
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            flash(f"Error opening image: {e}")
            return render_template("index.html")
        
        # ------------------------------
        # YOLOv5 Inference & Manual Drawing
        # ------------------------------
        results = model(img, size=640)
        # Each box: [x1, y1, x2, y2, conf, class_idx]
        boxes = results.xyxy[0].cpu().numpy()
        names = results.names  # class names

        # Create a copy for drawing
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        # Try loading a larger TrueType font (size 40); fallback to default if unavailable
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except Exception as e:
            font = ImageFont.load_default()

        for *xyxy, conf, cls_idx in boxes:
            x1, y1, x2, y2 = xyxy
            label = f"{names[int(cls_idx)]} {conf:.2f}"

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Use textbbox to measure label size
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # Position label above the box if there's space
            y_text = y1 - text_h if (y1 - text_h) > 0 else y1

            # Draw a filled rectangle behind the text
            draw.rectangle([x1, y_text, x1 + text_w, y_text + text_h], fill="red")
            # Draw label text
            draw.text((x1, y_text), label, fill="white", font=font)

        # ------------------------------
        # Enlarge the annotated image
        # ------------------------------
        scale_factor = 2
        enlarged_width = draw_img.width * scale_factor
        enlarged_height = draw_img.height * scale_factor
        enlarged_img = draw_img.resize((enlarged_width, enlarged_height), resample=Image.Resampling.LANCZOS)

        # Save the enlarged annotated image
        annotated_image_path = os.path.join(IMAGES_FOLDER, "annotated_image.jpg")
        enlarged_img.save(annotated_image_path)

        # Provide the annotated image URL to display
        result_image_url = url_for("static", filename="images/annotated_image.jpg")
    
    return render_template("index.html", result_image=result_image_url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    # Load YOLOv5 model from torch hub
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.eval()

    app.run(host="0.0.0.0", port=args.port, debug=True)
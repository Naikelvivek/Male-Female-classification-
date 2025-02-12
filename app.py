import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Load the trained model
model = tf.keras.models.load_model("gender_detection.keras")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Index page
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process the image
            processed_path, male_count, female_count = process_image(file_path)
            
            return render_template("result.html", result_image=processed_path, male=male_count, female=female_count)

    return render_template("index.html")

# Image processing function
def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    male_count, female_count = 0, 0

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = np.array(face, dtype="float32") / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)
        label = "Male" if np.argmax(prediction) == 0 else "Female"

        if label == "Male":
            male_count += 1
            color = (255, 0, 0)  # Blue for Male
        else:
            female_count += 1
            color = (255, 0, 255)  # Pink for Female

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    result_path = os.path.join(app.config["RESULT_FOLDER"], os.path.basename(image_path))
    cv2.imwrite(result_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return result_path, male_count, female_count

# Serve processed images
@app.route("/static/results/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)

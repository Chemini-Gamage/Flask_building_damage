from flask import Flask, request, render_template_string, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
import requests

app = Flask(__name__)

# Constants
MODEL_PATH = "model.h5"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/0es9tt5806lp14q05hjyr/model.h5?rlkey=vfh7xoo61o1weq8phh9nr1mku&st=14iph1ox&dl=1"
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Class labels
class_names = [
    "algae",
    "major_crack",
    "minor_crack",
    "peeling",
    "plain",
    "spalling",
    "stain"
]

# Lazy-load model placeholder
model = None

# Download model from Dropbox if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📦 Downloading model from Dropbox...")
        try:
            response = requests.get(DROPBOX_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print(f"✅ Model downloaded: {MODEL_PATH}")
        except Exception as e:
            print("❌ Model download failed:", e)
            exit(1)
    else:
        print("✅ Model already exists.")

# Call download at startup
download_model()
print("🚀 Flask app is initializing...")

# HTML form template
HTML_TEMPLATE = '''
<!doctype html>
<title>Building Damage Classifier</title>
<h1>Upload Image for Prediction</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file required>
  <input type=submit value=Predict>
</form>
{% if filename %}
  <h2>Uploaded Image:</h2>
  <img src="{{ url_for('static', filename=filename) }}" width="300">
  <h2>Prediction:</h2>
  <p><strong>Class:</strong> {{ predicted_class_name }}</p>
  <p><strong>Confidence:</strong> {{ confidence }}</p>
{% endif %}
'''

# Route
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    global model

    # Load model only when first needed
    if model is None:
        print("🔄 Loading model into memory...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded.")

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template_string(HTML_TEMPLATE)

        file = request.files['file']
        os.makedirs('static', exist_ok=True)
        unique_name = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
        filepath = os.path.join('static', unique_name)
        file.save(filepath)

        # Preprocess
        img = load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = round(float(np.max(prediction)), 4)

        return render_template_string(
            HTML_TEMPLATE,
            filename=unique_name,
            predicted_class_name=predicted_class_name,
            confidence=confidence
        )

    return render_template_string(HTML_TEMPLATE)

# Entry point for local dev or Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

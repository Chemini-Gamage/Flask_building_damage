from flask import Flask, request, render_template_string, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
import gdown

app = Flask(__name__)

MODEL_PATH = "model.h5"

GOOGLE_DRIVE_FILE_ID = "16_nTplqDTOnnHlVUrjNS5jBvpXaKT7Q4"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        try:
            gdown.download(
    url="https://drive.google.com/uc?export=download&id=16_nTplqDTOnnHlVUrjNS5jBvpXaKT7Q4",
    output=MODEL_PATH,
    quiet=False,
    fuzzy=True,
    use_cookies=False
)

            if os.path.exists(MODEL_PATH):
                print("Model downloaded successfully.")
            else:
                print("Download attempted but model.h5 still not found.")
                exit(1)
        except Exception as e:
            print("Download failed:", e)
            exit(1)
    else:
        print("Model already exists.")


# Download model before app starts
download_model()

# Load model
model = load_model(MODEL_PATH)

# Class names (in training order)
class_names = [
    "algae",
    "major_crack",
    "minor_crack",
    "peeling",
    "plain",
    "spalling",
    "stain"
]

IMG_HEIGHT, IMG_WIDTH = 224, 224

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

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template_string(HTML_TEMPLATE)

        file = request.files['file']
        os.makedirs('static', exist_ok=True)
        unique_name = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
        filepath = os.path.join('static', unique_name)
        file.save(filepath)

        # Load and preprocess image
        img = load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

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

if __name__ == '__main__':
 port = int(os.environ.get("PORT", 5000))
 app.run(host="0.0.0.0", port=port)

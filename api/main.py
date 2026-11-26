from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pathlib import Path
from keras.layers import TFSMLayer

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

# MODEL = tf.keras.models.load_model("../model/2")

#MODEL.export(BASE_DIR / "model" /"1")

# MODEL = TFSMLayer(
#     BASE_DIR / "model" / "1",
#     call_endpoint="serving_default"
# )

MODEL = tf.keras.models.load_model(
    BASE_DIR / "model" / "image_classifier2.keras"
)

#endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"


CLASS_NAME = ["Early Blight", "Late Blight", "Healthy"]

def get_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data))) # converting raw bytes to 2d numpy array
    return image

@app.get("/health")
async def health_check():
    return "The server is running fine"

@app.post("/predict")
async def make_prediction(file:UploadFile = File(...)): # (...) make the file upload mandatory.
    image = get_image(await file.read())
    print(image.shape)

    img_batch = np.expand_dims(image, 0) # Expanding out dimension from (256, 256, 3) to (1, 256, 256, 3)
    predictions = MODEL.predict(img_batch)
    print(predictions)

    predicted_class = CLASS_NAME[int(np.argmax(predictions[0]))]
    confidence = round(float(np.max(predictions[0])) * 100, 2)

    return { "class" : predicted_class,
                "confidence" : confidence}


# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8080)
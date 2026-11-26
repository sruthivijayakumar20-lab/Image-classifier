from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]

)

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAME = ["Early Blight", "Late Blight", "Healthy"]

def get_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data))) # converting raw bytes to 2d numpy array
    return image

@app.get("/health")
async def health_check():
    return "The server is running fine"

@app.post("/predict")
async def make_predictions(file:UploadFile = File(...)): # (...) make the file upload mandatory.
    image = get_image(await file.read())

    img_batch = np.expand_dims(image, 0) # Expanding out dimension from (256, 256, 3) to (1, 256, 256, 3)

    json_data = {
        "instances" : img_batch.tolist()
    }

    result = requests.post(endpoint, json = json_data)

    predictions = result.json()['predictions'][0]

    predicted_class = CLASS_NAME[np.argmax(predictions)]
    confidence = round(100 * (np.max(predictions[0])),2)

    return { "class" : predicted_class,
            "confidence" : confidence}

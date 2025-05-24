import cv2
import numpy as np
import os
import face_recognition
import collections
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import cosine
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()

# تحميل الموديل
model = tf.keras.models.load_model("Adv_face_recognition_cnn_model.h5")

# المتغيرات لتخزين الترميزات (encoding) للـ ID والـ Reference
id_encoding = None
reference_encoding = None

# رفع صورة ID
# رفع صورة ID
@app.post("/upload-id/")
async def upload_id_image(ID_image: UploadFile = File(..., alias="ID_image")):
    global id_encoding
    id_img = await ID_image.read()
    id_img = np.frombuffer(id_img, np.uint8)
    id_img = cv2.imdecode(id_img, cv2.IMREAD_COLOR)

    if id_img is None:
        return JSONResponse(content={"message": "Failed to load ID image."}, status_code=400)

    id_img = cv2.cvtColor(id_img, cv2.COLOR_BGR2RGB)
    id_img = cv2.resize(id_img, (100, 100))
    id_img = np.expand_dims(id_img, axis=0) / 255.0
    id_encoding = model.predict(id_img, verbose=0)

    return {"message": "ID image uploaded and encoding computed successfully"}
# رفع صورة Reference
@app.post("/upload-reference/")
async def upload_reference_image(reference_image: UploadFile = File(..., alias="reference_image")):
    global reference_encoding
    ref_img = await reference_image.read()
    ref_img = np.frombuffer(ref_img, np.uint8)
    ref_img = cv2.imdecode(ref_img, cv2.IMREAD_COLOR)

    if ref_img is None:
        return JSONResponse(content={"message": "Failed to load reference image."}, status_code=400)

    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.resize(ref_img, (100, 100))
    ref_img = np.expand_dims(ref_img, axis=0) / 255.0
    reference_encoding = model.predict(ref_img, verbose=0)

    return {"message": "Reference image uploaded and encoding computed successfully"}

# التحقق من صورة الاختبار
@app.post("/verify/")
async def verify_image(test_image: UploadFile = File(..., alias="test_image")):
    global id_encoding, reference_encoding

    if id_encoding is None or reference_encoding is None:
        return JSONResponse(content={"message": "ID or reference image not uploaded."}, status_code=400)

    test_img = await test_image.read()
    test_img = np.frombuffer(test_img, np.uint8)
    test_img = cv2.imdecode(test_img, cv2.IMREAD_COLOR)

    if test_img is None:
        return JSONResponse(content={"message": "Failed to load test image."}, status_code=400)

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, (100, 100))
    test_img = np.expand_dims(test_img, axis=0) / 255.0
    test_encoding = model.predict(test_img, verbose=0)

    threshold = 0.7
    is_verified = False

    for ref_encoding in [id_encoding, reference_encoding]:
        distance = np.linalg.norm(ref_encoding - test_encoding)
        if distance < threshold:
            is_verified = True
            break

    return {"is_verified": is_verified}
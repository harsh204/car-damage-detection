from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException, status
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import torch
import asyncio
from functions import predictOnBatch, validateFilenames
from Constants import *
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

torch.no_grad()

model = YOLO(MODEL_PATH)
app = FastAPI()
client = AsyncIOMotorClient(CONNECTION_STRING)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


@app.post("/cardetails")
async def detect_imperfections(files: list[UploadFile], background_tasks: BackgroundTasks, confScore: float = 0.25, save: bool = True, xyxyn: bool = False):
    filenames = [file.filename for file in files]

    isValid = validateFilenames(filenames)

    if not isValid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Files with only {' '.join(IMAGE_EXTENSIONS)} are allowed.")

    contents = await asyncio.gather(*[i.read() for i in files])

    image_bytes_list = [BytesIO(content) for content in contents]

    response = []

    for i in range(0, len(filenames), BATCH_SIZE):
        images = [Image.open(image_bytes) for image_bytes in image_bytes_list[i:i+BATCH_SIZE]]

        response = response + await predictOnBatch(filenames[i:i+BATCH_SIZE], images, model, confScore, save, xyxyn, collection, background_tasks)

    torch.cuda.empty_cache()

    return response

@app.get("/cardetails")
async def getDetails(car_id: str):
    car_detail: dict | None = await collection.find_one({"_id": ObjectId(car_id)})
    car_detail.update({"_id": str(car_detail["_id"])})
    return car_detail

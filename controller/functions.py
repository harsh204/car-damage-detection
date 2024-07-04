from PIL import Image
import numpy as np
from fastapi import BackgroundTasks
import os
from Constants import *
import ultralytics  

async def predictOnBatch(filenames: list[str], 
                         images: list[Image.Image], 
                         model: ultralytics.YOLO, 
                         confScore: float, 
                         save: bool, 
                         xyxyn: bool, 
                         collection, 
                         background_tasks: BackgroundTasks):
    size = len(filenames)
    
    results_gpu = model.predict(images, conf=confScore, verbose=False)
    
    results = [results_gpu[i].cpu().numpy() for i in range(size)]
    response = [getResponseFromPredict(results[i]) for i in range(size)]
    if save:
        im_array = [results_gpu[i].plot() for i in range(size)]
        image_paths = saveImg(im_array, size, filenames, background_tasks)
        addImagePathToResponse(response, image_paths, size)
    else:
        [response[i].update({"image_path": None}) for i in range(size)]

    if xyxyn:
        updateResult(response, "xyxyn", size, results)
    else:
        [response[i].update({"xyxyn": None}) for i in range(size)]

    await storeInDatabase(response, size, collection)

    del results_gpu
    return response

def getResponseFromPredict(result):
    result = result.numpy()
    classes = np.array(result.boxes.cls, dtype=int).tolist()

    response = {"numDetections": len(classes),
                "xywhn": result.boxes.xywhn.tolist(),
                "classIds":  classes,
                "labels": [IDS_TO_CLASS[i] for i in classes],
                "original_shape": result.orig_shape,
                "conf": result.boxes.conf.tolist()}
    
    return response


def validateFilenames(filenames: list[str]):
    for name in filenames:
        _, ext = os.path.splitext(name.lower())
        
        if ext not in IMAGE_EXTENSIONS:
            print(name)
            return False

    return True

def saveImg(im_array, size: int, filenames: list[str], background_tasks: BackgroundTasks):
    im = [Image.fromarray(im_array[i][...,::-1]) for i in range(size)]
    image_paths = [f"{RESULT_DIR}/{filename}" for filename in filenames]
    [background_tasks.add_task(im[i].save, image_paths[i]) for i in range(size)]
    return image_paths

def updateResult(response: list[dict], key: str, size: int, results):
    for i in range(size):

        attr = getattr(results[i].boxes, key)
        response[i].update({key: attr.tolist()})
    

def addImagePathToResponse(response: list[dict], image_paths: list[str], size: int):
    for i in range(size):
        response[i].update({"image_path": "file://"+os.path.abspath(image_paths[i])})

async def storeInDatabase(response: list[dict], size:int, collection):
    res = await collection.insert_many(response)
    ids = list(res.inserted_ids)

    [response[i].update({"_id": str(ids[i])}) for i in range(size)]
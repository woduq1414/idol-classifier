from typing import Optional
import face_recognition as frc
import uvicorn
from fastapi import FastAPI, Request
import time
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os.path
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import face_recognition as frc
import numpy as np
from jjy.framework.functions import *
from jjy.framework.network import MultiLayerNet
from os import path
import uuid

from tmp1 import process_img

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model_url = "./model/idol_train_weight_2021-06-06 090301_8907_np.npz"
net = MultiLayerNet()
# net.load_model("./model/idol_train_weight_2021-04-20 054113_8381_np.npz")
try:
    net.load_model(model_url)
except:
    net = MultiLayerNet()
    net.load_model("./model/idol_train_weight_2021-04-20 054113_8381_np.npz")


@app.get("/")
async def read_root():
    # time.sleep(3)
    return {"Hello": "World"}


@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/report-result")
async def report_result(request: Request):
    eng_idol_list = ['iu', 'irene', 'arin']

    body = await request.json()
    image_base64 = body["image"].split(",")[1]
    predicted_idol = body["predicted"]
    real_idol = body["real"]

    if predicted_idol not in eng_idol_list or real_idol not in eng_idol_list:
        return

    img = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
    if not os.path.exists('./assets/report_image'):
        os.makedirs('./assets/report_image')
    save_fname = f"{real_idol}_{predicted_idol}_{uuid.uuid4()}.jpg"
    img.save(f"./assets/report_image/{save_fname}")

    return {
        "message": "success!"
    }


@app.post("/upload-image")
async def upload_image(request: Request):
    idol_list = ["아이유", "아이린", "아린"]

    # req_body = request.body()
    body = await request.json()
    image_base64 = body["image"].split(",")[1]

    img = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')

    img_array = np.asarray(img)
    plt.imshow(img)
    plt.show()
    # print(np.asarray(img).shape)
    faces = frc.face_locations(img_array)
    faces = sorted(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    if len(faces) >= 1:
        face_location = faces[0]
        top, right, bottom, left = face_location
        face_image = img_array[top:bottom, left:right]

        cropped_img = Image.fromarray(face_image)
    else:
        cropped_img = img

    resized_img = cropped_img.resize((128, 128)).convert("L")

    # plt.imshow(cropped_img)
    # plt.show()
    # print(np.asarray(resized_img).shape)

    cropped_array = np.asarray(resized_img).reshape(1, 128, 128)
    predict = net.predict(np.array([cropped_array]) / 255, train_flg=False)
    predict_index = np.argmax(predict, axis=1)[0]
    predict_confidence = max(min(predict[0][predict_index], 8), 0) * 12.5

    print(predict)

    predict = predict / 2.5
    predict = softmax(predict)

    predict_idol = ["아이유", "아이린", "아린"][predict_index]
    print(predict_idol)

    process_img(img, predict_index)

    return {
        "result": {
            "idol": predict_idol,
            "percentage": sort_dict({idol_list[i]: predict[0][i] * 100 for i in range(len(idol_list))}, reverse=True),
            "confidence": predict_confidence
        }
    }


def sort_dict(dic, reverse=False):
    return dict(sorted(dic.items(), key=lambda item: item[1], reverse=reverse))


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=7777, reload=True, )

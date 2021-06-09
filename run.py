import math
from typing import Optional
import face_recognition as frc
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, Cookie
import time
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, File, UploadFile, WebSocket, Response
import os.path
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import face_recognition as frc
import json
import numpy as np
from jjy.framework.functions import *
from jjy.framework.network import MultiLayerNet
from os import path
import uuid
from typing import List

from starlette.middleware.cors import CORSMiddleware

from tmp1 import process_img
import asyncio
import string
import random
from datetime import datetime


from time import time

app = FastAPI()


class ConnectionManager:
    def __init__(self):
        self.connections = {}

    async def connect(self, websocket: WebSocket, client_id):
        await websocket.accept()
        self.connections[client_id] = websocket
        print(self.connections)


    async def disconnect(self, client_id):
        del self.connections[client_id]

    async def broadcast(self, data: str):
        for connection in self.connections.values():
            await connection.send_text(data)

    async def get_connection_by_id(self, client_id):
        connection = self.connections[client_id]
        return connection

    async def send(self, data, client_id):
        print(data, client_id)
        conn = await self.get_connection_by_id(client_id)
        await conn.send_text(json.dumps(data))


multi_queue = {

}

manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket, client_id)
    while True:
        try:
            data = await websocket.receive_text()
        except:
            await manager.disconnect(client_id)
            return
        await manager.broadcast(f"Client {client_id}: {data}")



app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model_url = "./model/idol_train_weight_2021-06-06 090301_8907_np.npz"
net = MultiLayerNet()
net.load_model("./model/idol_train_weight_2021-06-06 090301_8907_np.npz")

idol_list = ["아이유", "아이린", "아린"]
eng_idol_list = ["iu", "irene", "arin"]


def id_generator(size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))


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


async def process_multi(files, client_id=None):
    hash = id_generator(8)

    print(files)
    result_list = []

    folder_path = f'./static/export/{hash}'
    os.makedirs(folder_path)
    for idol in idol_list:
        os.makedirs(f"{folder_path}/output/{idol}")

    img_list = []
    cropped_list = []

    async def process(file, idx):
        # print(file.filename)
        # await asyncio.sleep(3)
        bin = await file.read()
        print(file.filename)
        img = Image.open(BytesIO(bin)).convert('RGB')
        img_list.append(img)
        cropped_list.append(get_cropped_img_array(img))
        print("SEND")
        await manager.send({
            "message": f"사진 자르는 중.. ({idx + 1}/{len(files)})",
            "status": "crop",
            "time" : str(datetime.now())
        }, client_id)
        # print("Crop", file.filename)
        return True

    async def main():

        # futures = [asyncio.ensure_future(process(file, idx)) for idx, file in enumerate(files)]
        # 태스크(퓨처) 객체를 리스트로 만듦
        # crop_result = await asyncio.gather(*futures)  # 결과를 한꺼번에 가져옴

        # for idx, img in enumerate(img_list):
        #
        #     cropped_list.append(get_cropped_img_array(img))
        #     await manager.send({
        #         "message": f"사진 자르는 중.. ({idx + 1}/{len(files)})",
        #         "status": "crop"
        #     }, client_id)
        for idx, file in enumerate(files):
            await process(file, idx)
        print("Crop Finish")

        max_batch_size = 10

        predict_list = []
        for i in range(math.ceil(len(cropped_list) / max_batch_size)):
            crop_batch = cropped_list[max_batch_size * i: min(max_batch_size * (i + 1), len(cropped_list))]
            t = net.predict(np.array(crop_batch) / 255, train_flg=False).tolist()
            predict_list.extend(t)

            await manager.send({
                "message": f"예측하는 중.. ({min(len(cropped_list), (i + 1) * max_batch_size)}/{len(cropped_list)})",
                "status": "predict"
            }, client_id)



        predict_list = np.array(predict_list)
        print(predict_list.shape)
        # print(predict.shape)

        # print(predict_index)
        # predict_confidence = max(min(predict[0][predict_index], 8), 0) * 12.5

        predict_list = predict_list / 2.5

        for predict in predict_list:
            predict = softmax(predict)

            predict_index = np.argmax(predict, axis=0)
            predict_idol = ["아이유", "아이린", "아린"][predict_index]
            percentage = sort_dict({idol_list[i]: predict[i] * 100 for i in range(len(idol_list))}, reverse=True)
            predict_confidence = max(min(predict[predict_index], 8), 0) * 12.5
            print(predict_idol)
            result_list.append({
                "idol": predict_idol,
                "percentage": percentage,
                "confidence": predict_confidence
            })

        for idx, (img, file, result) in enumerate(zip(img_list, files, result_list)):
            print("Image saved", f"{folder_path}/output/{result['idol']}/{file.filename}")
            img.save(f"{folder_path}/output/{result['idol']}/{file.filename}")


        import zipfile
        f = zipfile.ZipFile(f'{folder_path}/output.zip', 'w', zipfile.ZIP_DEFLATED)
        # startdir = f"{folder_path}/output"
        owd = os.getcwd()
        os.chdir(f"{folder_path}/output")
        for dirpath, dirnames, filenames in os.walk("./"):
            for filename in filenames:
                f.write(os.path.join(dirpath, filename))
        f.close()

        os.chdir(owd)

        await manager.send({
            "message": f'{folder_path}/output.zip',
            "result": result_list,
            "status": "finish"
        }, client_id)

        return result_list

    await main()

    # print(cropped_list)
    # print('실행 시간: {0:.3f}초'.format(end - begin))

    return result_list


def delete_old_files():
    from datetime import datetime
    import shutil
    path_target = f'./static/export'
    print("delete start")
    """path_target:삭제할 파일이 있는 디렉토리, days_elapsed:경과일수"""
    # print(os.listdir(path_target))
    for f in os.listdir(path_target): # 디렉토리를 조회한다
        f = os.path.join(path_target, f)
        if True or os.path.isfile(f): # 파일이면
            timestamp_now = datetime.now().timestamp() # 타임스탬프(단위:초)
            # st_mtime(마지막으로 수정된 시간)기준 X일 경과 여부
            is_old = os.stat(f).st_mtime < timestamp_now - (3 * 60 * 60)
            if is_old: # X일 경과했다면
                try:

                    shutil.rmtree(f) # 파일을 지운다
                    print(f, 'is deleted') # 삭제완료 로깅
                except OSError: # Device or resource busy (다른 프로세스가 사용 중)등의 이유
                    pass
                    # print(f, 'can not delete') # 삭제불가 로깅
            else:
                pass


@app.post("/upload-multi")
async def upload_multi(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...),
                       websocket_cookie: Optional[str] = Cookie(None), ):
    if websocket_cookie is None:
        raise HTTPException(status_code=401, detail="WebSocket Id Err.")

    for file in files:
        if file.content_type[:5] != "image":
            raise HTTPException(status_code=400, detail="Please upload image.")
    background_tasks.add_task(process_multi, files, client_id=int(websocket_cookie))
    background_tasks.add_task(delete_old_files)
    return {
        "message": "Pending"
    }


@app.get("/multi", response_class=HTMLResponse)
async def upload_multi_form(request: Request):
    return templates.TemplateResponse("multi.html", {"request": request})


@app.post("/upload-image")
async def upload_image(request: Request):
    # req_body = request.body()
    body = await request.json()
    image_base64 = body["image"].split(",")[1]

    img = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')

    return {
        "result": predict_label(img)
    }

    # plt.imshow(img)
    # plt.show()
    # print(np.asarray(img).shape)


def get_cropped_img_array(img):
    img_array = np.asarray(img)
    faces = frc.face_locations(img_array)
    faces = sorted(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    # print(len(faces))
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
    return cropped_array


def predict_label(img, is_cropped=False):
    if is_cropped:
        cropped_array = img
    else:
        cropped_array = get_cropped_img_array(img)

    predict = net.predict(np.array([cropped_array]) / 255, train_flg=False)

    predict_index = np.argmax(predict, axis=1)[0]

    predict_confidence = max(min(predict[0][predict_index], 8), 0) * 12.5

    predict = predict / 2.5
    predict = softmax(predict)

    predict_idol = ["아이유", "아이린", "아린"][predict_index]
    print(predict_idol)

    process_img(img, predict_index)

    return {
        "idol": predict_idol,
        "percentage": sort_dict({idol_list[i]: predict[0][i] * 100 for i in range(len(idol_list))}, reverse=True),
        "confidence": predict_confidence
    }


def sort_dict(dic, reverse=False):
    return dict(sorted(dic.items(), key=lambda item: item[1], reverse=reverse))


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=7777, reload=True, )

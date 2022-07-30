from fastapi import FastAPI, Form
import uvicorn
import base64
import numpy as np
from PIL import Image
import io
import cv2

app = FastAPI()

@app.post("/get_blur/")
async def ret_blur(image: str=Form(), bbox: list=Form()):
    bbox = [int(i) for i in bbox ]
    print("bbox",bbox)
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(image))))
    
    blur = cv2.GaussianBlur(image, (25, 25), 0)
    portion = image[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    blur[bbox[1]:bbox[3],bbox[0]:bbox[2]] = portion

    _, imagebytes = cv2.imencode(".jpg", blur)
    temp_b64 = base64.b64encode(imagebytes).decode("utf8")

    return temp_b64

if __name__ == "__main__":
    print("Server Running....")
    uvicorn.run("api_v1:app", port=8000, host="0.0.0.0",reload=True)
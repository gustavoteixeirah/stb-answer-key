import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from marked_circles_detection.service import detect_filled_circles

app = FastAPI()

@app.post("/detect_marks")
async def detect_marks_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = detect_filled_circles(img)
    return result

def main():
    uvicorn.run(app=app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

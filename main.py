import uuid
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

from src.shared.log_utils import initialize_trace
from src.marked_circles_detection.service import MarkedCirclesDetector

print("Carregando o modelo de detecção...")
detector = MarkedCirclesDetector(model_path="src/marked_circles_detection/best.pt")
print("Modelo carregado com sucesso.")

app = FastAPI(
    title="StudyBuddy Answer Key API",
    description="API para detectar círculos marcados em gabaritos.",
    version="1.0.0"
)

@app.post("/detect_marks", summary="Detecta círculos marcados em uma imagem")
async def detect_marks_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    trace_id = str(uuid.uuid4())
    initialize_trace(trace_id)
    result_circles = detector.detect(img, trace_id=trace_id)

    return {
        "trace_id": trace_id,
        "detected_circles": result_circles
    }

def main():
    print("Iniciando o servidor FastAPI em http://0.0.0.0:8000")
    uvicorn.run(app=app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
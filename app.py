from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import torch, base64

app = FastAPI(title="Skin Models API")

# 🔹 Load all models once when the server starts
MODELS = {
    "acne": YOLO("weights/acne_best.pt"),
    "wrinkle": YOLO("weights/wrinkle_best.pt"),
    "blackhead": YOLO("weights/blackhead_best.pt"),
    "darkcircle": YOLO("weights/darkcircle_best.pt"),
    "pigmentation": YOLO("weights/pigmentation_best.pt"),
    "pore_redness": YOLO("weights/pore_redness_best.pt"),
}

device = 0 if torch.cuda.is_available() else "cpu"


@app.post("/receive-image")
async def receive_image(image: UploadFile = File(...)):
    """
    Receives an uploaded image from WordPress,
    runs all YOLO models locally,
    and returns detections + annotated images (base64) in JSON.
    """
    try:
        # ✅ Read uploaded image into memory
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        problems = []
        annotated_images = []

        # 🔁 Run through all models
        for name, model in MODELS.items():
            results = model.predict(img, conf=0.25, iou=0.45, imgsz=640, device=device, verbose=False)

            # Get detections
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Pick the top confidence detection
                top_conf = float(boxes.conf.max().item())
                problems.append({"name": name, "confidence": round(top_conf * 100, 2)})

                # Create annotated image (YOLO draws boxes automatically)
                annotated = results[0].plot()  # returns a NumPy array with annotations
                im_pil = Image.fromarray(annotated)

                # Convert to base64 string instead of saving file
                buffer = BytesIO()
                im_pil.save(buffer, format="JPEG")
                encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                proxied_url = f"data:image/jpeg;base64,{encoded_img}"

                annotated_images.append({"label": name, "proxied_url": proxied_url})

        # Return in same format as your current API
        data = {
            "problems": problems,
            "annotated_images": annotated_images,
            "recommendations": [],  # (you can fill later with your GPT logic)
        }

        return JSONResponse(content={"success": True, "data": data})

    except Exception as e:
        return JSONResponse(
            content={"success": False, "data": {"message": str(e)}},
            status_code=500,
        )


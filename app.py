from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from openai import OpenAI
import torch, base64, os, json

# ============================================================
# 1️⃣  CONFIGURATION
# ============================================================

app = FastAPI(title="Glow AI Recommender (Local YOLO + OpenAI)")

# --- Load all your trained YOLOv8 models from /weights ---
MODELS = {
    "acne": YOLO("weights/acne_best.pt"),
    "wrinkle": YOLO("weights/wrinkle_best.pt"),
    "blackhead": YOLO("weights/blackhead_best.pt"),
    "darkcircle": YOLO("weights/darkcircle_best.pt"),
    "pigmentation": YOLO("weights/pigmentation_best.pt"),
    "pore_redness": YOLO("weights/pore_redness_best.pt"),
}

device = 0 if torch.cuda.is_available() else "cpu"

# --- OpenAI API setup ---
# Make sure you set this as an environment variable in Railway or locally:
#    export OPENAI_API_KEY="sk-xxxx"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# 2️⃣  HELPER: convert PIL image → base64 for JSON
# ============================================================
def pil_to_base64(im_pil):
    buffer = BytesIO()
    im_pil.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# 3️⃣  MAIN ENDPOINT
# ============================================================
@app.post("/receive-image")
async def receive_image(image: UploadFile = File(...)):
    """
    Receives uploaded image from WordPress,
    runs local YOLO models to detect skin concerns,
    and returns detections + annotated base64 images + GPT recommendations.
    """
    try:
        # ---------------------------------------------
        # Step 1. Read uploaded image into memory
        # ---------------------------------------------
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        problems = []
        annotated_images = []

        # ---------------------------------------------
        # Step 2. Run all YOLO models locally
        # ---------------------------------------------
        for name, model in MODELS.items():
            results = model.predict(
                img, conf=0.25, iou=0.45, imgsz=640, device=device, verbose=False
            )

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                conf = float(boxes.conf.max().item())
                problems.append({"name": name, "confidence": round(conf * 100, 2)})

                annotated = results[0].plot()
                im_pil = Image.fromarray(annotated)
                annotated_images.append(
                    {"label": name, "proxied_url": pil_to_base64(im_pil)}
                )

        # ---------------------------------------------
        # Step 3. Generate recommendations with OpenAI
        # ---------------------------------------------
        recommendations = []
        if problems:
            try:
                # Create a user-friendly prompt for GPT
                prompt = (
                    "You are a skincare specialist. Based on these detected skin issues, "
                    "suggest suitable skincare products or treatments. "
                    "Return results as a JSON array with title, reason, and product_url fields.\n\n"
                    f"Detected issues: {json.dumps(problems, indent=2)}"
                )

                completion = client.chat.completions.create(
                    model="gpt-4o-mini",  # or "gpt-4-turbo"
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )

                # Parse OpenAI’s structured response safely
                content = completion.choices[0].message.content
                data_json = json.loads(content)
                recommendations = data_json.get("recommendations", [])
            except Exception as gpt_err:
                print("⚠️ OpenAI recommendation error:", gpt_err)
                recommendations = []

        # ---------------------------------------------
        # Step 4. Build API response for WordPress
        # ---------------------------------------------
        data = {
            "problems": problems,
            "annotated_images": annotated_images,
            "recommendations": recommendations,
        }

        return JSONResponse(content={"success": True, "data": data})

    except Exception as e:
        print("❌ Error:", e)
        return JSONResponse(
            content={"success": False, "data": {"message": str(e)}}, status_code=500
        )

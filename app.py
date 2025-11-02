from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from openai import OpenAI
import torch, base64, os, json

# ============================================================
# 1️⃣ CONFIGURATION
# ============================================================

app = FastAPI(title="Glow AI Recommender – Local YOLO + OpenAI")

device = 0 if torch.cuda.is_available() else "cpu"

# --- OpenAI setup (read from environment variables on Render) ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ============================================================
# 2️⃣ LOAD YOLO MODELS (read directly from your repo /weights)
# ============================================================

print("🧠 Loading YOLO models from local /weights folder...")

MODELS = {
    "acne": YOLO("weights/acne_best.pt"),
    "wrinkle": YOLO("weights/wrinkle_best.pt"),
    "blackhead": YOLO("weights/blackhead_best.pt"),
    "darkcircle": YOLO("weights/darkcircle_best.pt"),
    "pigmentation": YOLO("weights/pigmentation_best.pt"),
    "pore_redness": YOLO("weights/pore_redness_best.pt"),
}

print("✅ All YOLO models loaded successfully.")

# ============================================================
# 3️⃣ HELPER – Convert PIL image to base64 (for JSON response)
# ============================================================
def pil_to_base64(im_pil):
    buffer = BytesIO()
    im_pil.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# 4️⃣ MAIN ENDPOINT
# ============================================================
@app.post("/receive-image")
async def receive_image(image: UploadFile = File(...)):
    """
    Receives uploaded image from WordPress,
    runs YOLO models locally,
    and returns detections + annotated base64 images + GPT recommendations.
    """
    try:
        # Step 1️⃣ Read uploaded image
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        problems = []
        annotated_images = []

        # Step 2️⃣ Run detections using each model
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

        # Step 3️⃣ Generate recommendations using OpenAI
        recommendations = []
        if problems:
            try:
                prompt = (
                    "You are a skincare expert. Based on these detected skin issues, "
                    "suggest suitable skincare products or treatments. "
                    "Return a JSON object with a 'recommendations' array, "
                    "where each item has 'title', 'reason', and 'product_url'.\n\n"
                    f"Detected issues: {json.dumps(problems, indent=2)}"
                )

                completion = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )

                content = completion.choices[0].message.content
                data_json = json.loads(content)
                recommendations = data_json.get("recommendations", [])
            except Exception as gpt_err:
                print("⚠️ OpenAI recommendation error:", gpt_err)

        # Step 4️⃣ Return combined result to WordPress
        data = {
            "problems": problems,
            "annotated_images": annotated_images,
            "recommendations": recommendations,
        }

        return JSONResponse(content={"success": True, "data": data})

    except Exception as e:
        print("❌ Error:", e)
        return JSONResponse(
            content={"success": False, "data": {"message": str(e)}},
            status_code=500,
        )


# ============================================================
# 5️⃣ ROOT ENDPOINT (optional check)
# ============================================================
@app.get("/")
def root():
    return {"message": "Glow AI Recommender backend is running successfully!"}

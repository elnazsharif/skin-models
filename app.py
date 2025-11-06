from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from openai import OpenAI
import torch, base64, os, json, requests

# ============================================================
# 1️⃣ CONFIGURATION
# ============================================================

app = FastAPI(title="Glow AI Recommender – Local YOLO + OpenAI")

device = 0 if torch.cuda.is_available() else "cpu"

# --- OpenAI setup (read from environment variables on Render) ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  


# ============================================================
# 2️⃣ DOWNLOAD MODEL WEIGHTS (only first run)
# ============================================================

def download_if_missing(url, dest_path):
    """Download YOLO model weights if not already present"""
    if not os.path.exists(dest_path):
        print(f"⬇️ Downloading model: {os.path.basename(dest_path)}")
        r = requests.get(url)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded {dest_path}")
    else:
        print(f"✅ Found existing model: {dest_path}")

os.makedirs("weights", exist_ok=True)

# 🔹 Replace these with your actual Google Drive direct download links
model_links = {
        "darkcircle_best.pt": "https://drive.google.com/uc?export=download&id=1o9i07SIm1lXCOc_C7Hk_rREM6YaX7aWH",
    "pigmentation_best.pt": "https://drive.google.com/uc?export=download&id=1hzkesH6aF0FSKgfX-BpmaJ61X8pFDNpT",
     "wrinkle_best.pt": "https://drive.google.com/uc?export=download&id=1n-Yz3s0PGwmFSHG_Hu9yQ1gMDNFfkg8n",
    "blackhead_best.pt": "https://drive.google.com/uc?export=download&id=1pfwCADIuEPOki5nKriETUUqQ46JqJEv7",



   
}

for filename, url in model_links.items():
    download_if_missing(url, f"weights/{filename}")

# ============================================================
# 3️⃣ LOAD MODELS
# ============================================================
print("🧠 Loading YOLO models into memory...")
MODELS = {
        "darkcircle": YOLO("weights/darkcircle_best.pt"),
        "pigmentation": YOLO("weights/pigmentation_best.pt"),
    "wrinkle": YOLO("weights/wrinkle_best.pt"),
    "blackhead": YOLO("weights/blackhead_best.pt"),


    
}
print("✅ All models loaded successfully.")

# ============================================================
# 4️⃣ HELPER – Convert PIL image to base64
# ============================================================
def pil_to_base64(im_pil):
    buffer = BytesIO()
    im_pil.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

# ============================================================
# 5️⃣ MAIN ENDPOINT
# ============================================================
@app.post("/receive-image")
async def receive_image(image: UploadFile = File(...)):
    """
    Receives uploaded image from WordPress,
    runs YOLO models locally,
    and returns detections + annotated base64 images + GPT recommendations.
    """
    try:
        # Step 1. Read uploaded image
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        problems = []
        annotated_images = []

        # Step 2. Run detections
        for name, model in MODELS.items():
            results = model.predict(
                img, conf=0.01, iou=0.45, imgsz=640, device=device, verbose=False
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

        # Step 3. Get OpenAI recommendations
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

        # Step 4. Return final data to WordPress
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

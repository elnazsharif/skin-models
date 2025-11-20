from fastapi import FastAPI, UploadFile, File, Form
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

# --- OpenAI setup (read from environment variables on Railway) ---
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

model_links = {
    "darkcircle_best.pt":   "https://drive.google.com/uc?export=download&id=1o9i07SIm1lXCOc_C7Hk_rREM6YaX7aWH",
    "pigmentation_best.pt": "https://drive.google.com/uc?export=download&id=1hzkesH6aF0FSKgfX-BpmaJ61X8pFDNpT",
    "wrinkle_best.pt":      "https://drive.google.com/uc?export=download&id=1n-Yz3s0PGwmFSHG_Hu9yQ1gMDNFfkg8n",
    "blackhead_best.pt":    "https://drive.google.com/uc?export=download&id=1pfwCADIuEPOki5nKriETUUqQ46JqJEv7",
    # acne6-best.pt LEFT HERE BUT NOT USED IN MODELS (acne temporarily disabled)
    "acne6-best.pt":        "https://drive.google.com/uc?export=download&id=1cIi4wYajDJAMonhk7l_lfiS2rMK_fJ2-",
    "pore_redness_best.pt": "https://drive.google.com/uc?export=download&id=1tjrtrIuuE5cA987CzMnekb6lcIRAFC4y",
}

for filename, url in model_links.items():
    download_if_missing(url, f"weights/{filename}")

# ============================================================
# 3️⃣ LOAD MODELS
# ============================================================
print("🧠 Loading YOLO models into memory...")

# ❗ Acne model is intentionally NOT loaded (temporarily disabled)
MODELS = {
    "darkcircle":   YOLO("weights/darkcircle_best.pt"),
    "pigmentation": YOLO("weights/pigmentation_best.pt"),
    "wrinkle":      YOLO("weights/wrinkle_best.pt"),
    "blackhead":    YOLO("weights/blackhead_best.pt"),
    "pore_redness": YOLO("weights/pore_redness_best.pt"),
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
async def receive_image(
    image: UploadFile = File(...),

    # We KEEP conf_acne in the signature so WordPress can still send it,
    # but we do NOT use the acne model for now.
    conf_acne: float = Form(0.10),

    conf_wrinkle: float = Form(0.10),
    conf_eyebag: float = Form(0.10),         # darkcircle model
    conf_blackhead: float = Form(0.10),
    conf_pore_red: float = Form(0.10),
    conf_pigmentation: float = Form(0.10),

    overlap: int = Form(30),
):
    try:
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        problems = []
        annotated_images = []

        # Match exact YOLO model keys in MODELS {}
        model_thresholds = {
            # "acne":        conf_acne,   # <- DISABLED
            "wrinkle":      conf_wrinkle,
            "darkcircle":   conf_eyebag,
            "pigmentation": conf_pigmentation,
            "pore_redness": conf_pore_red,
            "blackhead":    conf_blackhead,
        }

        for name, model in MODELS.items():
            th = model_thresholds.get(name, 0.10)

            results = model.predict(
                img,
                conf=th,
                iou=overlap / 100.0,
                imgsz=640,
                device=device,
                verbose=False,
            )

            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                conf = float(boxes.conf.max().item())

                problems.append({
                    "name": name,
                    "confidence": round(conf * 100.0, 2),
                })

                annotated = results[0].plot()
                im_pil = Image.fromarray(annotated)

                annotated_images.append({
                    "label": name,
                    "proxied_url": pil_to_base64(im_pil),
                })

        # OpenAI recommendations
        recommendations = []
        if problems:
            prompt = (
                "You are a skincare expert. Based on these detected skin issues, "
                "suggest suitable skincare products or treatments. "
                "Return JSON with 'recommendations'.\n\n"
                f"Detected: {json.dumps(problems, indent=2)}"
            )

            try:
                completion = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                data_json = json.loads(completion.choices[0].message.content)
                recommendations = data_json.get("recommendations", [])
            except Exception as gpt_err:
                print("GPT error:", gpt_err)

        return JSONResponse(content={
            "success": True,
            "data": {
                "problems": problems,
                "annotated_images": annotated_images,
                "recommendations": recommendations,
            },
        })

    except Exception as e:
        print("❌ Error:", e)
        return JSONResponse(
            content={"success": False, "data": {"message": str(e)}},
            status_code=500,
        )

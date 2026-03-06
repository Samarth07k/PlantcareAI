# PlantCare AI – Flask Backend

AI-powered plant disease detection using MobileNetV2 transfer learning.

---

## Project Structure

```
plantcare_backend/
├── app.py                   # Flask application (routes + API)
├── model.py                 # ML model loading, prediction, recommendations
├── train_model.py           # MobileNetV2 training script
├── requirements.txt         # Python dependencies
├── plant_disease_model.h5   # ← Place your trained model here
├── static/
│   └── uploads/             # Uploaded images are stored here
└── templates/               # Jinja2 HTML templates
    ├── home.html
    ├── about.html
    ├── upload.html
    ├── result.html
    ├── healthyimage.html
    └── unhealthyimage.html
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# If you have a trained model, also install TensorFlow:
pip install tensorflow>=2.13.0
```

### 2. Add your trained model

Copy your trained model file to the project root:

```bash
cp /path/to/your/model.h5 plant_disease_model.h5
```

> **Note:** Without the model file, the app runs in **demo/stub mode** —  
> it still works and returns plausible predictions based on the filename hash,  
> which is useful for frontend testing.

### 3. Run the app

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## Training Your Own Model

### Dataset

Download the **New Plant Diseases Dataset** from Kaggle:  
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Expected structure:
```
New_Plant_Diseases_Dataset/
    train/
        Apple___Apple_scab/
        Apple___Black_rot/
        ... (38 classes)
    valid/
        Apple___Apple_scab/
        ...
```

### Train

```bash
python train_model.py --data_dir /path/to/New_Plant_Diseases_Dataset
```

Training runs in two phases:
1. **Phase 1** – Top layers trained, MobileNetV2 base frozen (10 epochs)
2. **Phase 2** – Fine-tuning the last 30 layers of MobileNetV2 (10 epochs)

The best model is saved automatically as `plant_disease_model.h5`.

---

## API Reference

### `POST /api/predict`

Accepts a multipart form with a `file` field (JPG, JPEG, PNG, WebP, max 10MB).

**Response:**
```json
{
  "success": true,
  "plant": "Tomato",
  "condition": "Early blight",
  "confidence": 94,
  "is_healthy": false,
  "raw_label": "Tomato___Early_blight",
  "recommendations": [
    "Remove affected leaves and dispose of them",
    "Apply copper-based or chlorothalonil fungicide",
    "Ensure proper air circulation around plants",
    "Avoid overhead watering – use drip irrigation"
  ],
  "image_url": "http://localhost:5000/static/uploads/abc123.jpg"
}
```

---

## Supported Plants & Diseases (38 classes)

| Plant | Diseases |
|-------|----------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Team

**Smart Bridge Hyderabad**  
- Samarth Kulkarni (Team Lead)
- Isha Raundal
- Tanishk Shrivastava
- Payal Wadile

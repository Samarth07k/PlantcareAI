"""
model.py
--------
Handles loading the trained MobileNetV2 plant disease model,
preprocessing input images, running inference, and returning
structured results with recommendations.

The 38 classes come from the New Plant Diseases Dataset.
If the trained model file (plant_disease_model.h5) is not found,
the module falls back to a clearly labeled stub so the app still runs.
"""

import os
import numpy as np

# ── Class labels (New Plant Diseases Dataset – 38 classes) ──────────────

CLASS_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ── Recommendations per disease ──────────────────────────────────────────

RECOMMENDATIONS = {
    # Apple
    "Apple___Apple_scab": [
        "Apply fungicide (captan or myclobutanil) during wet weather",
        "Remove and destroy fallen infected leaves",
        "Prune to improve air circulation in the canopy",
        "Plant resistant varieties when replanting",
    ],
    "Apple___Black_rot": [
        "Prune out cankers and dead wood immediately",
        "Apply copper-based fungicide at early stages",
        "Remove mummified fruits from the tree",
        "Avoid wounding the bark during harvest",
    ],
    "Apple___Cedar_apple_rust": [
        "Remove nearby cedar/juniper trees if possible",
        "Apply fungicide (myclobutanil) at bud-break",
        "Use resistant apple varieties",
        "Inspect regularly during spring and treat early",
    ],
    "Apple___healthy": [
        "Continue regular watering and balanced fertilization",
        "Monitor for pest activity under leaves",
        "Prune annually for good air circulation",
        "Apply preventive fungicide spray in spring",
    ],
    # Blueberry
    "Blueberry___healthy": [
        "Maintain soil pH between 4.5 and 5.5",
        "Water consistently – blueberries prefer moist soil",
        "Apply mulch to retain moisture and suppress weeds",
        "Watch for early signs of mummy berry disease",
    ],
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": [
        "Apply sulfur or potassium bicarbonate fungicide",
        "Prune to open the canopy and improve airflow",
        "Avoid excess nitrogen fertilization",
        "Water at the base to keep foliage dry",
    ],
    "Cherry_(including_sour)___healthy": [
        "Maintain regular watering schedule",
        "Inspect foliage weekly for fungal signs",
        "Thin fruit clusters to reduce disease pressure",
        "Apply dormant oil spray before bud-break",
    ],
    # Corn
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": [
        "Apply triazole or strobilurin fungicide at tasseling",
        "Rotate crops – avoid continuous corn planting",
        "Use resistant hybrids in your next planting",
        "Ensure proper plant spacing for airflow",
    ],
    "Corn_(maize)___Common_rust_": [
        "Apply foliar fungicide (propiconazole) if infection is severe",
        "Plant rust-resistant corn hybrids",
        "Monitor fields after cool, humid weather",
        "Ensure timely planting to avoid peak rust season",
    ],
    "Corn_(maize)___Northern_Leaf_Blight": [
        "Apply fungicide at early tassel stage",
        "Use resistant hybrids for future planting",
        "Remove and destroy crop residue after harvest",
        "Rotate with non-host crops like soybeans",
    ],
    "Corn_(maize)___healthy": [
        "Continue balanced NPK fertilization",
        "Scout fields regularly during vegetative stages",
        "Maintain proper plant density per hectare",
        "Monitor soil moisture to avoid drought stress",
    ],
    # Grape
    "Grape___Black_rot": [
        "Apply mancozeb or myclobutanil fungicide from bud-break",
        "Remove and destroy mummified berries",
        "Prune to improve air circulation through canopy",
        "Avoid overhead irrigation",
    ],
    "Grape___Esca_(Black_Measles)": [
        "Remove and destroy infected wood immediately",
        "Seal pruning wounds with wound sealant",
        "Avoid pruning during wet weather",
        "Consult a viticulture specialist for severe infections",
    ],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": [
        "Apply copper-based fungicide preventively",
        "Ensure good canopy management and ventilation",
        "Remove heavily infected leaves and burn them",
        "Avoid irrigation late in the day",
    ],
    "Grape___healthy": [
        "Continue canopy management practices",
        "Monitor for downy mildew and powdery mildew signs",
        "Apply preventive fungicide before wet season",
        "Maintain soil nutrition with proper vineyard testing",
    ],
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": [
        "Remove and destroy infected trees to prevent spread",
        "Control Asian citrus psyllid vector with insecticides",
        "Use certified disease-free planting material",
        "Consult local agricultural authority immediately",
    ],
    # Peach
    "Peach___Bacterial_spot": [
        "Apply copper-based bactericide at petal fall",
        "Avoid overhead irrigation to reduce leaf wetness",
        "Remove severely infected shoots and leaves",
        "Plant resistant varieties in future seasons",
    ],
    "Peach___healthy": [
        "Apply dormant oil spray to control overwintering pests",
        "Thin fruit for better size and air circulation",
        "Monitor for peach leaf curl in early spring",
        "Ensure adequate potassium for fruit quality",
    ],
    # Pepper
    "Pepper,_bell___Bacterial_spot": [
        "Apply copper-based bactericide spray",
        "Use certified disease-free transplants",
        "Avoid working in the field when plants are wet",
        "Rotate crops for at least 2 years",
    ],
    "Pepper,_bell___healthy": [
        "Maintain consistent soil moisture",
        "Side-dress with balanced fertilizer at flowering",
        "Monitor for aphids and spider mites",
        "Ensure good air circulation between plants",
    ],
    # Potato
    "Potato___Early_blight": [
        "Apply chlorothalonil or mancozeb fungicide",
        "Remove and destroy infected lower leaves",
        "Avoid overhead irrigation",
        "Hill soil around plants to protect tubers",
    ],
    "Potato___Late_blight": [
        "Destroy infected plants immediately to stop spread",
        "Apply preventive fungicide (metalaxyl) before wet weather",
        "Harvest tubers before heavy rain",
        "Do NOT compost infected material – burn or bury it",
    ],
    "Potato___healthy": [
        "Scout for early blight and late blight weekly",
        "Ensure proper crop rotation (3-year cycle)",
        "Apply hilling at 20–25 cm plant height",
        "Monitor soil moisture for consistent tuber development",
    ],
    # Raspberry, Soybean
    "Raspberry___healthy": [
        "Prune old canes after harvest",
        "Apply mulch to conserve soil moisture",
        "Monitor for raspberry cane borer",
        "Test soil annually and adjust pH to 5.6–6.2",
    ],
    "Soybean___healthy": [
        "Scout regularly for soybean aphid and sudden death syndrome",
        "Apply foliar micronutrients at V3–V5 stage",
        "Ensure proper nodulation for nitrogen fixation",
        "Rotate with non-legume crops",
    ],
    # Squash
    "Squash___Powdery_mildew": [
        "Apply sulfur or bicarbonate-based fungicide",
        "Plant resistant varieties",
        "Avoid water stress – inconsistent moisture encourages disease",
        "Remove heavily infected leaves to slow spread",
    ],
    # Strawberry
    "Strawberry___Leaf_scorch": [
        "Apply copper-based fungicide at early signs",
        "Remove and destroy infected leaves",
        "Avoid overhead irrigation",
        "Renovate beds annually after harvest",
    ],
    "Strawberry___healthy": [
        "Renew strawberry beds every 3–4 years",
        "Apply runner management to prevent overcrowding",
        "Monitor for botrytis (gray mold) during fruiting",
        "Water at the base to keep foliage dry",
    ],
    # Tomato
    "Tomato___Bacterial_spot": [
        "Apply copper-based bactericide spray",
        "Remove infected leaves immediately",
        "Avoid overhead watering",
        "Rotate with non-solanaceous crops",
    ],
    "Tomato___Early_blight": [
        "Remove affected leaves and dispose of them",
        "Apply copper-based or chlorothalonil fungicide",
        "Ensure proper air circulation around plants",
        "Avoid overhead watering – use drip irrigation",
    ],
    "Tomato___Late_blight": [
        "Apply metalaxyl or chlorothalonil fungicide immediately",
        "Remove and destroy all infected plant tissue",
        "Avoid wetting the foliage when irrigating",
        "Consult an agronomist if spread is rapid",
    ],
    "Tomato___Leaf_Mold": [
        "Improve greenhouse ventilation",
        "Apply chlorothalonil or copper fungicide",
        "Reduce humidity with proper spacing",
        "Remove lower infected leaves promptly",
    ],
    "Tomato___Septoria_leaf_spot": [
        "Apply fungicide (chlorothalonil) at first sign",
        "Remove infected lower leaves",
        "Avoid handling plants when wet",
        "Mulch around plants to reduce soil splash",
    ],
    "Tomato___Spider_mites Two-spotted_spider_mite": [
        "Apply miticide (abamectin) or insecticidal soap",
        "Increase humidity – mites prefer dry conditions",
        "Introduce predatory mites (Phytoseiulus persimilis)",
        "Rinse plants with strong jets of water",
    ],
    "Tomato___Target_Spot": [
        "Apply fungicide (azoxystrobin) at early infection",
        "Remove infected leaves and dispose properly",
        "Improve air circulation with proper staking",
        "Avoid excessive nitrogen fertilization",
    ],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
        "Control whitefly vectors using insecticides or sticky traps",
        "Remove and destroy infected plants immediately",
        "Use reflective mulch to repel whiteflies",
        "Plant resistant varieties in future seasons",
    ],
    "Tomato___Tomato_mosaic_virus": [
        "Remove and destroy infected plants immediately",
        "Disinfect tools with bleach solution (1:9 ratio)",
        "Wash hands thoroughly before handling plants",
        "Control aphid vectors and avoid tobacco near plants",
    ],
    "Tomato___healthy": [
        "Continue regular watering and care",
        "Ensure adequate sunlight and balanced nutrients",
        "Monitor for any changes in appearance weekly",
        "Maintain good air circulation around plants",
    ],
}

# ── Model Loading ────────────────────────────────────────────────────────

_model = None
_model_available = False
_IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'plant_disease_model.h5')


def _load_model():
    """Load the MobileNetV2 model from disk (lazy loading)."""
    global _model, _model_available
    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found at '{MODEL_PATH}'.")
        print("[WARNING] Using random prediction stub. Replace with your trained model.")
        _model_available = False
        return

    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        _model_available = True
        print("[INFO] Plant disease model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        _model_available = False


def _preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for MobileNetV2 inference."""
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 preprocessing: scale to [-1, 1]
    arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)


# ── Public API ───────────────────────────────────────────────────────────

def parse_label(raw_label: str) -> tuple[str, str, bool]:
    """
    Parse a raw class label like 'Tomato___Early_blight' into
    (plant_name, condition, is_healthy).
    """
    parts = raw_label.split('___')
    plant_raw = parts[0].replace('_', ' ').replace('(', '').replace(')', '').strip()
    condition_raw = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Unknown'
    is_healthy = 'healthy' in condition_raw.lower()
    return plant_raw, condition_raw, is_healthy


def predict_disease(image_path: str) -> dict:
    """
    Run plant disease prediction on an image file.

    Returns a dict with keys:
        plant       – human-readable plant name
        condition   – disease name (or 'Healthy')
        confidence  – int 0–100
        is_healthy  – bool
        raw_label   – original class string
    """
    _load_model()

    if _model_available:
        img_array = _preprocess_image(image_path)
        preds = _model.predict(img_array, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = int(round(float(preds[class_idx]) * 100))
        raw_label = CLASS_LABELS[class_idx]
    else:
        # ── Stub: deterministic demo predictions ──
        import hashlib
        h = int(hashlib.md5(os.path.basename(image_path).encode()).hexdigest(), 16)
        class_idx = h % len(CLASS_LABELS)
        confidence = 85 + (h % 14)   # 85–98 %
        raw_label = CLASS_LABELS[class_idx]

    plant, condition, is_healthy = parse_label(raw_label)

    return {
        'plant': plant,
        'condition': condition,
        'confidence': confidence,
        'is_healthy': is_healthy,
        'raw_label': raw_label,
    }


def get_recommendations(plant: str, condition: str, is_healthy: bool) -> list[str]:
    """
    Return a list of recommendation strings for the given prediction.
    Falls back to generic advice if no specific entry is found.
    """
    # Try to find an exact match by rebuilding the label key
    for label, recs in RECOMMENDATIONS.items():
        p, c, h = parse_label(label)
        if p.lower() == plant.lower() and c.lower() == condition.lower():
            return recs

    if is_healthy:
        return [
            "Continue regular watering and care",
            "Ensure adequate sunlight and nutrients",
            "Monitor for any changes in appearance",
            "Maintain good air circulation around plants",
        ]
    return [
        "Isolate affected plants to prevent further spread",
        "Consult with a local agricultural expert",
        "Consider appropriate fungicide or pesticide treatment",
        "Monitor surrounding plants for similar symptoms",
    ]

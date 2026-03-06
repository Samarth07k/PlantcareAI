# PlantCare AI – Flask Backend

This project is a Flask backend for a plant disease detection system.
It uses a MobileNetV2-based deep learning model to identify plant diseases from leaf images and provide treatment suggestions.

The backend handles image uploads, runs predictions using the trained model, and returns recommendations for the detected disease.

---

## Project Structure

```
plantcare_backend/

├── app.py                   # Main Flask application (routes + API)
├── model.py                 # Loads the trained model and handles predictions
├── train_model.py           # Script used to train the MobileNetV2 model
├── requirements.txt         # Python dependencies
├── plant_disease_model.h5   # Place the trained model file here
├── static/
│   └── uploads/             # Uploaded images are stored here
└── templates/               # HTML templates
    ├── home.html
    ├── about.html
    ├── upload.html
    ├── result.html
    ├── healthyimage.html
    └── unhealthyimage.html
```

---

## Quick Start

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you are running the model locally, make sure TensorFlow is installed:

```bash
pip install tensorflow>=2.13.0
```

---

### 2. Add the Trained Model

Copy your trained `.h5` model file into the project root directory and name it:

```
plant_disease_model.h5
```

Example:

```bash
cp /path/to/your/model.h5 plant_disease_model.h5
```

If the model file is not present, the application will still run in a **demo mode**.
In this mode, predictions are generated using a simple placeholder method so the frontend can still be tested.

---

### 3. Run the Application

Start the Flask server:

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

---

## Training Your Own Model

### Dataset

The model can be trained using the **New Plant Diseases Dataset** from Kaggle:

https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

After downloading and extracting the dataset, it should have a structure similar to this:

```
New_Plant_Diseases_Dataset/

    train/
        Apple___Apple_scab/
        Apple___Black_rot/
        ...

    valid/
        Apple___Apple_scab/
        ...
```

---

### Training the Model

Run the training script and specify the dataset directory:

```bash
python train_model.py --data_dir /path/to/New_Plant_Diseases_Dataset
```

Training happens in two stages:

**Stage 1**

* Train the classification layers
* MobileNetV2 base remains frozen

**Stage 2**

* Fine-tune the last few layers of MobileNetV2
* Helps improve accuracy on plant disease images

The best performing model will be saved automatically as:

```
plant_disease_model.h5
```

---

## API Reference

### POST `/api/predict`

This endpoint accepts an image file and returns the predicted plant disease.

**Request**

* Method: POST
* Content-Type: multipart/form-data
* Field name: `file`
* Supported formats: JPG, JPEG, PNG, WebP
* Maximum file size: 10MB

---

### Example Response

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

## Supported Plants and Diseases

The model supports **38 different classes** covering multiple plants and their common diseases.

| Plant      | Diseases                                                                                                                                           |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Apple      | Apple Scab, Black Rot, Cedar Apple Rust, Healthy                                                                                                   |
| Blueberry  | Mummy Berry, Botrytis Blight, Healthy                                                                                                              |
| Cherry     | Powdery Mildew, Healthy                                                                                                                            |
| Corn       | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy                                                                                         |
| Grape      | Black Rot, Esca (Black Measles), Leaf Blight, Healthy                                                                                              |
| Orange     | Huanglongbing (Citrus Greening)                                                                                                                    |
| Peach      | Bacterial Spot, Healthy                                                                                                                            |
| Pepper     | Bacterial Spot, Healthy                                                                                                                            |
| Potato     | Early Blight, Late Blight, Healthy                                                                                                                 |
| Raspberry  | Gray Mold, Anthracnose, Cane Blight, Healthy                                                                                                       |
| Soybean    | Sudden Death Syndrome, Frogeye Leaf Spot, Healthy                                                                                                  |
| Squash     | Powdery Mildew                                                                                                                                     |
| Strawberry | Leaf Scorch, Healthy                                                                                                                               |
| Tomato     | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Notes

* The backend is designed to work with a Flask frontend using HTML templates.
* Uploaded images are stored temporarily in the `static/uploads` directory.
* Predictions are returned through both the web interface and the API endpoint.

This project can be extended by:

* Adding more plant datasets
* Improving the model architecture
* Deploying the API using Docker or cloud services.

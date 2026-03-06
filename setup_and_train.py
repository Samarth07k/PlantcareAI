"""
setup_and_train.py
------------------
One-command script to:
  1. Install all required packages
  2. Download the New Plant Diseases Dataset from Kaggle
  3. Verify dataset structure
  4. Train the MobileNetV2 model (2 phases)
  5. Save plant_disease_model.h5 ready for app.py

HOW TO GET YOUR KAGGLE API TOKEN:
    1. Go to https://www.kaggle.com/settings
    2. Scroll to the "API" section
    3. Click "Create New Token"  →  kaggle.json downloads
    4. Place kaggle.json here:
         Windows : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
         macOS   : ~/.kaggle/kaggle.json
         Linux   : ~/.kaggle/kaggle.json
    5. On Linux/macOS: chmod 600 ~/.kaggle/kaggle.json

THEN JUST RUN:
    python setup_and_train.py
"""

import os
import sys
import subprocess
import zipfile
import shutil

# ── Config ────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "vipoooool/new-plant-diseases-dataset"
DOWNLOAD_DIR   = "dataset_download"
DATASET_DIR    = "New_Plant_Diseases_Dataset"   # final extracted folder name
OUTPUT_MODEL   = "plant_disease_model.h5"

IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
EPOCHS_FROZEN  = 10
EPOCHS_FINETUNE= 10
LEARNING_RATE  = 1e-3
FINETUNE_LR    = 1e-5


# ── Step 1: Install dependencies ──────────────────────────────────────────
def install_packages():
    print("\n" + "="*60)
    print("STEP 1: Installing dependencies")
    print("="*60)
    packages = [
        "flask>=3.0.0",
        "werkzeug>=3.0.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "tensorflow>=2.13.0",
        "kaggle>=1.6.0",
        "tqdm",
    ]
    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("  ✓ All packages installed.")


# ── Step 2: Verify Kaggle credentials ─────────────────────────────────────
def check_kaggle_credentials():
    print("\n" + "="*60)
    print("STEP 2: Checking Kaggle credentials")
    print("="*60)

    kaggle_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_path):
        print("\n  ✗ kaggle.json NOT found at:", kaggle_path)
        print("\n  Please follow these steps:")
        print("    1. Go to https://www.kaggle.com/settings")
        print("    2. Click 'Create New Token' under the API section")
        print("    3. Move the downloaded kaggle.json to:", kaggle_path)
        print("    4. On Linux/macOS run: chmod 600", kaggle_path)
        print("    5. Re-run this script")
        sys.exit(1)

    # Set permissions on Linux/macOS
    if sys.platform != "win32":
        os.chmod(kaggle_path, 0o600)

    print("  ✓ kaggle.json found at:", kaggle_path)


# ── Step 3: Download dataset ───────────────────────────────────────────────
def download_dataset():
    print("\n" + "="*60)
    print("STEP 3: Downloading New Plant Diseases Dataset (~2.6 GB)")
    print("="*60)

    # Skip if already extracted
    if os.path.isdir(DATASET_DIR):
        train_dir = os.path.join(DATASET_DIR, "train")
        valid_dir = os.path.join(DATASET_DIR, "valid")
        if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
            print(f"  ✓ Dataset already extracted at '{DATASET_DIR}'. Skipping download.")
            return

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print(f"  Downloading from Kaggle: {KAGGLE_DATASET}")
    print("  This may take several minutes depending on your internet speed...")

    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=DOWNLOAD_DIR,
        unzip=False,
        quiet=False
    )
    print("  ✓ Download complete.")

    # Find the zip file
    zip_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".zip")]
    if not zip_files:
        print("  ✗ No zip file found in download directory.")
        sys.exit(1)

    zip_path = os.path.join(DOWNLOAD_DIR, zip_files[0])
    print(f"\n  Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(".")

    # Cleanup download folder
    shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)

    # Find the extracted folder
    _fix_dataset_structure()
    print("  ✓ Extraction complete.")


def _fix_dataset_structure():
    """
    The Kaggle zip sometimes extracts with a nested folder.
    This normalises it so we always have:
        New_Plant_Diseases_Dataset/train/  and
        New_Plant_Diseases_Dataset/valid/
    """
    global DATASET_DIR

    # Common extraction patterns
    candidates = [
        "New Plant Diseases Dataset",
        "New_Plant_Diseases_Dataset",
        os.path.join("New Plant Diseases Dataset", "New Plant Diseases Dataset"),
        os.path.join("New_Plant_Diseases_Dataset", "New_Plant_Diseases_Dataset"),
    ]

    for candidate in candidates:
        train_check = os.path.join(candidate, "train")
        valid_check = os.path.join(candidate, "valid")
        if os.path.isdir(train_check) and os.path.isdir(valid_check):
            # Rename to standard name if needed
            if candidate != DATASET_DIR:
                if os.path.exists(DATASET_DIR):
                    shutil.rmtree(DATASET_DIR)
                os.rename(candidate, DATASET_DIR)
            return

    # Fallback: search for a folder containing train/ and valid/
    for root, dirs, _ in os.walk("."):
        if "train" in dirs and "valid" in dirs:
            if root != "." and root != DATASET_DIR:
                if os.path.exists(DATASET_DIR):
                    shutil.rmtree(DATASET_DIR)
                os.rename(root, DATASET_DIR)
                return


# ── Step 4: Verify dataset structure ──────────────────────────────────────
def verify_dataset():
    print("\n" + "="*60)
    print("STEP 4: Verifying dataset structure")
    print("="*60)

    train_dir = os.path.join(DATASET_DIR, "train")
    valid_dir = os.path.join(DATASET_DIR, "valid")

    if not os.path.isdir(train_dir):
        print(f"  ✗ train/ folder not found inside '{DATASET_DIR}'")
        sys.exit(1)
    if not os.path.isdir(valid_dir):
        print(f"  ✗ valid/ folder not found inside '{DATASET_DIR}'")
        sys.exit(1)

    train_classes = sorted(os.listdir(train_dir))
    valid_classes = sorted(os.listdir(valid_dir))

    print(f"  ✓ train/ — {len(train_classes)} classes found")
    print(f"  ✓ valid/ — {len(valid_classes)} classes found")

    if len(train_classes) != 38:
        print(f"  ⚠ Expected 38 classes, found {len(train_classes)}. Proceeding anyway.")

    # Count total images
    total_train = sum(
        len(os.listdir(os.path.join(train_dir, c)))
        for c in train_classes
        if os.path.isdir(os.path.join(train_dir, c))
    )
    total_valid = sum(
        len(os.listdir(os.path.join(valid_dir, c)))
        for c in valid_classes
        if os.path.isdir(os.path.join(valid_dir, c))
    )
    print(f"  ✓ Total training images : {total_train:,}")
    print(f"  ✓ Total validation images: {total_valid:,}")

    return train_dir, valid_dir, len(train_classes)


# ── Step 5: Train model ────────────────────────────────────────────────────
def train_model(train_dir, valid_dir, num_classes):
    print("\n" + "="*60)
    print("STEP 5: Training MobileNetV2 model")
    print(f"        Classes : {num_classes}")
    print(f"        Image size: {IMG_SIZE}")
    print(f"        Batch size: {BATCH_SIZE}")
    print("="*60)

    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    print(f"\n  TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  GPU detected       : {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("  GPU detected       : None (using CPU — training will be slower)")

    # ── Data generators ───────────────────────────────────────────────────
    def mobilenet_preprocess(x):
        return x / 127.5 - 1.0   # scale to [-1, 1]

    train_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    valid_datagen = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)

    print("\n  Loading training data...")
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    print("  Loading validation data...")
    valid_gen = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Save class index mapping alongside model
    class_indices_path = "class_indices.txt"
    with open(class_indices_path, "w") as f:
        for cls, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
            f.write(f"{idx},{cls}\n")
    print(f"  ✓ Class indices saved to '{class_indices_path}'")

    # ── Build model ───────────────────────────────────────────────────────
    print("\n  Building MobileNetV2 transfer learning model...")
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    print(f"  ✓ Model built — {model.count_params():,} total parameters")

    callbacks = [
        ModelCheckpoint(
            OUTPUT_MODEL, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.3,
            patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    # ── Phase 1: Train top layers (base frozen) ───────────────────────────
    print("\n" + "-"*50)
    print("  Phase 1: Training custom top layers (base frozen)")
    print(f"  Epochs: {EPOCHS_FROZEN}  |  LR: {LEARNING_RATE}")
    print("-"*50)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks,
        verbose=1
    )

    # ── Phase 2: Fine-tune top MobileNetV2 layers ─────────────────────────
    print("\n" + "-"*50)
    print("  Phase 2: Fine-tuning last 30 MobileNetV2 layers")
    print(f"  Epochs: {EPOCHS_FINETUNE}  |  LR: {FINETUNE_LR}")
    print("-"*50)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks,
        verbose=1
    )

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n" + "-"*50)
    print("  Final evaluation on validation set...")
    loss, acc = model.evaluate(valid_gen, verbose=1)
    print(f"\n  ✓ Final Validation Accuracy : {acc*100:.2f}%")
    print(f"  ✓ Final Validation Loss     : {loss:.4f}")
    print(f"  ✓ Model saved to            : '{OUTPUT_MODEL}'")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  PlantCare AI — Setup & Training Script")
    print("  Dataset: New Plant Diseases Dataset (Kaggle)")
    print("  Model  : MobileNetV2 (Transfer Learning)")
    print("="*60)

    install_packages()
    check_kaggle_credentials()
    download_dataset()
    train_dir, valid_dir, num_classes = verify_dataset()
    train_model(train_dir, valid_dir, num_classes)

    print("\n" + "="*60)
    print("  ✓ SETUP COMPLETE!")
    print(f"  Your trained model is ready: {OUTPUT_MODEL}")
    print("  Now run:  python app.py")
    print("  Open:     http://localhost:5000")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

"""
train_model.py
--------------
Transfer learning training script using MobileNetV2.

Usage:
    python train_model.py --data_dir /path/to/New_Plant_Diseases_Dataset

Expected dataset structure (New Plant Diseases Dataset):
    data_dir/
        train/
            Apple___Apple_scab/
            Apple___Black_rot/
            ...
        valid/
            Apple___Apple_scab/
            ...

After training, the model is saved as 'plant_disease_model.h5'
in the same directory as app.py.
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Config ───────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 38
EPOCHS_FROZEN  = 10   # epochs with MobileNetV2 base frozen
EPOCHS_FINETUNE = 10  # epochs with top layers unfrozen
LEARNING_RATE  = 1e-3
FINETUNE_LR    = 1e-5
OUTPUT_MODEL   = 'plant_disease_model.h5'


def build_model(num_classes: int) -> Model:
    """Build MobileNetV2 transfer learning model."""
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def get_data_generators(data_dir: str):
    """Create train and validation data generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1./127.5,
        preprocessing_function=lambda x: x - 1.0,  # MobileNetV2 [-1, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(
        rescale=1./127.5,
        preprocessing_function=lambda x: x - 1.0,
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    valid_gen = valid_datagen.flow_from_directory(
        os.path.join(data_dir, 'valid'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, valid_gen


def train(data_dir: str):
    print(f"[INFO] Building model for {NUM_CLASSES} classes...")
    model, base_model = build_model(NUM_CLASSES)

    callbacks = [
        ModelCheckpoint(OUTPUT_MODEL, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    ]

    train_gen, valid_gen = get_data_generators(data_dir)

    # ── Phase 1: Train only the top layers ──────────────────────────────
    print("\n[INFO] Phase 1: Training custom top layers (base frozen)...")
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

    # ── Phase 2: Fine-tune the top layers of MobileNetV2 ────────────────
    print("\n[INFO] Phase 2: Fine-tuning top MobileNetV2 layers...")
    # Unfreeze the last 30 layers of the base model
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

    print(f"\n[INFO] Training complete. Model saved to '{OUTPUT_MODEL}'")

    # Print class index mapping
    print("\n[INFO] Class indices:")
    for cls, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: {cls}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train plant disease model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset folder containing train/ and valid/ subdirectories')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Dataset directory not found: {args.data_dir}")
        exit(1)

    train(args.data_dir)

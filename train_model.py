from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "dataset" / "PlantVillage"
MODEL_DIR = BASE_DIR / "model"
IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_splits(dataset_dir: Path, seed: int):
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Create dataset\\PlantVillage\\<class name>\\ and put that class's images inside."
        )

    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    samples: list[tuple[Path, int]] = []
    class_names: list[str] = []
    rng = random.Random(seed)
    train_samples, validation_samples, test_samples = [], [], []

    for class_dir in class_dirs:
        paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        if len(paths) < 3:
            print(f"Skipping {class_dir.name}: at least 3 images are needed for train/validation/test.")
            continue
        class_index = len(class_names)
        class_names.append(class_dir.name)
        rng.shuffle(paths)
        test_count = max(1, round(len(paths) * 0.15))
        validation_count = max(1, round(len(paths) * 0.15))
        if test_count + validation_count >= len(paths):
            test_count = validation_count = 1
        test_paths = paths[:test_count]
        validation_paths = paths[test_count:test_count + validation_count]
        training_paths = paths[test_count + validation_count:]
        train_samples.extend((path, class_index) for path in training_paths)
        validation_samples.extend((path, class_index) for path in validation_paths)
        test_samples.extend((path, class_index) for path in test_paths)

    if len(class_names) < 2:
        raise ValueError(f"Expected at least 2 class folders containing 3+ images in {dataset_dir}.")
    rng.shuffle(train_samples)
    rng.shuffle(validation_samples)
    rng.shuffle(test_samples)
    return class_names, train_samples, validation_samples, test_samples


def make_dataset(samples, batch_size: int, training: bool):
    paths = [str(path) for path, _ in samples]
    labels = [label for _, label in samples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        content = tf.io.read_file(path)
        image = tf.io.decode_image(content, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(samples), 10_000), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(class_count: int):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="image_augmentation",
    )
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)(x)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train an image classifier on a PlantVillage-style dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers.")

    tf.keras.utils.set_random_seed(args.seed)
    class_names, train_samples, validation_samples, test_samples = collect_splits(
        args.data_dir, args.seed
    )
    print(
        f"Classes: {len(class_names)} | "
        f"train: {len(train_samples)} | validation: {len(validation_samples)} | "
        f"test: {len(test_samples)}"
    )

    training_data = make_dataset(train_samples, args.batch_size, training=True)
    validation_data = make_dataset(validation_samples, args.batch_size, training=False)
    test_data = make_dataset(test_samples, args.batch_size, training=False)
    model = build_model(len(class_names))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "plant_disease.keras"
    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
    )

    loss, accuracy = model.evaluate(test_data, verbose=0)
    probabilities = model.predict(test_data, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = np.concatenate([labels.numpy() for _, labels in test_data])
    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))

    model.save(model_path)
    (MODEL_DIR / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
        "class_names": class_names,
        "split_counts": {
            "train": len(train_samples),
            "validation": len(validation_samples),
            "test": len(test_samples),
        },
    }
    (MODEL_DIR / "evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"\nHeld-out test accuracy (measured): {accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Evaluation report saved to: {MODEL_DIR / 'evaluation.json'}")
    print("These metrics describe this dataset split, not guaranteed performance on real-world photos.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ModuleNotFoundError as exc:
    if exc.name != "tensorflow":
        raise
    tf = None
    load_model = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "plant_disease.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
IMAGE_SIZE = (224, 224)
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
CONFIDENCE_THRESHOLD = 0.60
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "local-development-change-before-deployment"
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 64 * 1024
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None
_class_names: list[str] | None = None
INFERENCE_ERRORS = (OSError, ValueError, TypeError, IndexError, RuntimeError)
if tf is not None:
    INFERENCE_ERRORS += (tf.errors.OpError,)

CARE_GUIDANCE = {
    "early blight": {
        "symptoms": "Older leaves may develop brown spots with concentric rings and yellowing.",
        "causes": "A fungal disease favored by warm, humid conditions and infected plant debris.",
        "care": "Remove badly affected leaves, keep foliage dry, improve airflow, and rotate crops.",
    },
    "late blight": {
        "symptoms": "Water-soaked leaf patches can enlarge quickly; humid weather may bring pale growth underneath.",
        "causes": "A fast-spreading oomycete disease favored by cool, wet conditions.",
        "care": "Avoid overhead watering, remove affected material carefully, improve airflow, and seek local extension advice promptly.",
    },
    "bacterial spot": {
        "symptoms": "Small dark or water-soaked spots may appear on leaves and sometimes fruit.",
        "causes": "Bacteria can spread through splashing water, handling, or contaminated plant material.",
        "care": "Avoid working with wet plants, remove badly affected material, sanitize tools, and use clean seed or transplants.",
    },
    "powdery mildew": {
        "symptoms": "White, powder-like patches may develop on leaf surfaces.",
        "causes": "Powdery mildew fungi spread readily in crowded plantings and favorable humidity.",
        "care": "Increase spacing and airflow, remove affected leaves, and avoid excess nitrogen fertilizer.",
    },
    "leaf mold": {
        "symptoms": "Pale patches on upper leaves may correspond to olive or gray growth underneath.",
        "causes": "A fungal disease associated with high humidity and poor airflow.",
        "care": "Ventilate the growing area, space plants, water at the soil line, and remove infected debris.",
    },
    "septoria leaf spot": {
        "symptoms": "Numerous small, round leaf spots may have pale centers and darker edges.",
        "causes": "A fungus that can persist in crop debris and spread by splashing water.",
        "care": "Remove infected lower leaves, mulch soil to reduce splash, rotate crops, and water at the base.",
    },
    "mosaic virus": {
        "symptoms": "Leaves may show mottled light and dark areas, distortion, or stunted growth.",
        "causes": "A plant virus; transmission routes vary and can include insects, tools, or infected plants.",
        "care": "Isolate the plant, clean tools, control weeds around the crop, and consult local extension guidance.",
    },
    "yellow leaf curl virus": {
        "symptoms": "Leaves may curl upward and yellow, with reduced growth or fruit set.",
        "causes": "A virus commonly spread by whiteflies in susceptible crops.",
        "care": "Separate affected plants, inspect for whiteflies, remove severely affected plants where appropriate, and seek local advice.",
    },
    "rust": {
        "symptoms": "Rust-colored or orange-brown pustules may appear on leaves.",
        "causes": "Rust fungi; many types affect only particular host plants.",
        "care": "Remove affected leaves, keep foliage dry, improve airflow, and avoid moving infected plant material.",
    },
    "scab": {
        "symptoms": "Dark, rough, scabby lesions may develop on leaves or fruit.",
        "causes": "A host-specific fungal disease that can persist in fallen leaves or infected material.",
        "care": "Remove fallen infected leaves, improve airflow, and choose resistant varieties when replanting.",
    },
    "black rot": {
        "symptoms": "Leaf spots may have tan centers and dark borders; fruit or stems can also be affected.",
        "causes": "A fungal disease that can persist in infected plant debris.",
        "care": "Remove infected debris, avoid overhead watering, improve airflow, and use clean planting material.",
    },
    "greening": {
        "symptoms": "Leaves may yellow unevenly and fruit may be small, misshapen, or poorly colored.",
        "causes": "A serious citrus disease associated with a bacterium spread by psyllid insects.",
        "care": "Contact a local agricultural extension service for confirmation and regional guidance; do not move potentially infected citrus plants.",
    },
}

PLANT_CARE = {
    "tomato": ("Water deeply at the soil line when the top layer begins to dry.", "Full sun (about 6–8 hours daily) and fertile, well-draining soil.", "A balanced vegetable fertilizer can support establishment; follow its label and avoid excess nitrogen once flowering begins."),
    "potato": ("Keep soil evenly moist, especially while tubers form; avoid waterlogging.", "Full sun and loose, well-draining soil.", "Use a fertilizer labeled for potatoes or vegetables; follow the product label and avoid excess nitrogen."),
    "pepper": ("Water when the top few centimeters of soil feel dry; do not leave roots waterlogged.", "Warm, sunny conditions and well-draining soil.", "A balanced vegetable fertilizer is usually suitable; follow the label and avoid over-fertilizing."),
    "apple": ("Water young trees deeply during dry periods; avoid persistently soggy soil.", "Full sun and well-draining soil.", "Base fertilizer on a soil test or local tree-care guidance rather than applying it automatically."),
    "grape": ("Water deeply during establishment and prolonged dry periods; avoid wet foliage.", "Full sun and well-drained soil.", "Use a soil test to guide fertilizer; excessive nitrogen can encourage leafy growth."),
    "corn": ("Provide consistent moisture, especially around flowering and ear development.", "Full sun and fertile, well-draining soil.", "A soil-test-guided nitrogen plan is preferable; follow local crop guidance."),
    "default": ("Water at the soil line when the top layer of soil begins to dry; adjust for the plant and climate.", "Provide bright light suited to the identified crop and use a clean, well-draining growing medium.", "Choose a balanced fertilizer labeled for the plant or crop. Follow its label; fertilizer supports growth but does not cure disease."),
}


class ModelUnavailableError(RuntimeError):
    """Raised when a valid trained classifier has not been installed."""


def model_is_ready() -> bool:
    return MODEL_PATH.is_file() and CLASS_NAMES_PATH.is_file()


def get_classifier():
    global _model, _class_names
    if _model is None or _class_names is None:
        if not model_is_ready():
            raise ModelUnavailableError
        if load_model is None:
            raise ModelUnavailableError("TensorFlow is not installed in the active Python environment.")
        _model = load_model(MODEL_PATH, compile=False)
        with CLASS_NAMES_PATH.open(encoding="utf-8") as labels_file:
            loaded_names = json.load(labels_file)
        if not isinstance(loaded_names, list) or not loaded_names or not all(
            isinstance(name, str) for name in loaded_names
        ):
            raise ValueError("The saved class_names.json must contain a non-empty list of strings.")
        _class_names = loaded_names
    return _model, _class_names


def decode_upload(file_storage) -> np.ndarray:
    suffix = Path(file_storage.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Choose a JPG, JPEG, or PNG image.")

    contents = file_storage.stream.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise ValueError("The image is larger than the 8 MB upload limit. Choose a smaller image.")

    try:
        image = Image.open(BytesIO(contents))
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("The file contents must be a real JPEG or PNG image.")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("This image has too many pixels. Choose an image smaller than 20 megapixels.")
        image.verify()
        image = Image.open(BytesIO(contents))
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The image could not be opened. Try a different JPG or PNG file.") from exc


def split_class_name(label: str) -> tuple[str, str]:
    plant, separator, condition = label.partition("___")
    if not separator:
        return label.replace("_", " "), ""
    return plant.replace("_", " "), condition.replace("_", " ")


def care_for_prediction(label: str, is_healthy: bool) -> dict[str, str]:
    plant, condition = split_class_name(label)
    plant_key = plant.lower()
    plant_care = PLANT_CARE.get(plant_key, PLANT_CARE["default"])

    if is_healthy:
        symptoms = "No disease pattern was identified by the model. Continue monitoring for changes."
        causes = "The classifier matched this image to a healthy training category; this does not rule out every plant problem."
        care = "Continue routine care and inspect leaves regularly, including their undersides."
    else:
        guidance = CARE_GUIDANCE.get(condition.lower())
        if guidance:
            symptoms = guidance["symptoms"]
            causes = guidance["causes"]
            care = guidance["care"]
        else:
            symptoms = "The model matched a disease category, but this app does not have verified symptom details for that category."
            causes = "The model cannot establish the cause from an image alone."
            care = "Check the plant and growing conditions closely, isolate it if practical, and contact a local plant-health expert for confirmation."

    return {
        "symptoms": symptoms,
        "causes": causes,
        "care": care,
        "watering": plant_care[0],
        "sunlight_soil": plant_care[1],
        "fertilizer": plant_care[2],
    }


@app.get("/")
def index():
    return render_template("index.html", model_ready=model_is_ready())


@app.get("/about")
def about():
    return render_template("about.html", model_ready=model_is_ready())


@app.post("/analyze")
def analyze():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        flash("Choose a plant-leaf image before starting the analysis.", "error")
        return redirect(url_for("index"))

    try:
        image_array = decode_upload(uploaded_file)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))

    try:
        classifier, class_names = get_classifier()
    except ModelUnavailableError:
        logger.info("Analysis requested before a trained model was installed.")
        return render_template(
            "result.html",
            unavailable=True,
            tensorflow_unavailable=tf is None,
            model_ready=False,
        ), 503
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.exception("Unable to load the trained plant classifier.")
        return render_template("result.html", model_error=True, model_ready=False), 500

    try:
        probabilities = classifier.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
        if (
            len(probabilities) != len(class_names)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            raise ValueError("The model output does not match the saved class labels.")
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        label = class_names[best_index]

        if confidence < CONFIDENCE_THRESHOLD:
            return render_template(
                "result.html",
                uncertain=True,
                confidence=confidence,
                threshold=CONFIDENCE_THRESHOLD,
                model_ready=True,
            )

        plant, condition = split_class_name(label)
        healthy = condition.lower() == "healthy" or label.lower().endswith("healthy")
        return render_template(
            "result.html",
            uncertain=False,
            confidence=confidence,
            plant=plant,
            condition="Healthy" if healthy else condition,
            healthy=healthy,
            guidance=care_for_prediction(label, healthy),
            model_ready=True,
        )
    except INFERENCE_ERRORS:
        logger.exception("Plant image analysis failed.")
        return render_template("result.html", analysis_error=True, model_ready=model_is_ready()), 500


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    flash("The image is larger than the 8 MB upload limit. Choose a smaller image.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)

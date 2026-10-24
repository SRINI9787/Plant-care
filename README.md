# AI Plant Doctor — Leafwise

A local Flask website for an AI-assisted plant-leaf check and practical plant-care guidance. The project includes the full application and model-training code, but **does not include a trained disease model or the PlantVillage dataset**. It will not make predictions until you add a dataset and train the classifier yourself.

## What is included

- A responsive home page, image-upload flow, result/report page, care dashboard, and about page.
- JPG, JPEG, and PNG validation in the browser and server; an 8 MB upload limit and image-pixel limit.
- A MobileNetV2 transfer-learning script with deterministic per-class train, validation, and held-out test splits.
- Measured test accuracy, a per-class precision/recall/F1 report, and a confusion matrix saved after training.
- A visible uncertainty response for model scores under 60%. The score is a model confidence score, not a calibrated probability or diagnosis.
- Practical care information that avoids pesticide recommendations and chemical dosages.

The browser photo preview is local. The Flask app processes each submitted image in memory and does not save the original image. Model weights and evaluation outputs are generated on your computer by the training command.

## Project files

```text
AI Plant Doctor/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── model/                   # Trained files are generated here
├── dataset/
│   └── PlantVillage/
│       ├── class-name-1/
│       └── class-name-2/
├── templates/
│   ├── index.html
│   ├── result.html
│   └── about.html
└── static/
    ├── css/style.css
    ├── js/script.js
    └── images/leaf.svg
```

## Windows setup in VS Code

### 1. Install Python and open the project

Install **64-bit Python 3.11** from [python.org](https://www.python.org/downloads/) and select **Add python.exe to PATH** in the installer. In VS Code, open this project folder, then choose **Terminal → New Terminal**. In the VS Code Python extension, select the `.venv` interpreter after creating it below.

### 2. Create and activate a virtual environment

Run these commands in the VS Code PowerShell terminal, from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run this once in the same terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Alternatively, use Command Prompt in VS Code:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

TensorFlow may take several minutes to install. The first training run also downloads MobileNetV2's ImageNet starting weights, so it needs internet access. Training runs on your CPU; a GPU is not required.

### 4. Download and arrange the dataset

PlantVillage is a public research dataset available through [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease). You must accept the dataset's current terms on Kaggle yourself. **Do not upload private or sensitive photos to a public dataset.**

One download option is Kaggle's official command-line tool:

1. Create/sign in to a Kaggle account and generate an API token from your Kaggle account settings.
2. Follow Kaggle's instructions to put `kaggle.json` in `%USERPROFILE%\.kaggle\kaggle.json`. Keep this credential private; do not put it in the project.
3. Install and download/extract the public dataset:

```powershell
python -m pip install kaggle
kaggle datasets download -d emmarex/plantdisease -p dataset --unzip
```

You can also download the archive from the Kaggle website and extract it yourself. Arrange the extracted image files so each disease/healthy category is its own folder directly inside `dataset\PlantVillage\`. For example:

```text
dataset\PlantVillage\Tomato___Early_blight\image1.JPG
dataset\PlantVillage\Tomato___Early_blight\image2.JPG
dataset\PlantVillage\Tomato___healthy\image1.JPG
dataset\PlantVillage\Potato___Late_blight\image1.JPG
```

The exact top-level extracted folder name can vary. If Kaggle creates `dataset\PlantVillage\PlantVillage\...`, use that inner `PlantVillage` directory as the training folder, or move its class folders up one level. Each category folder needs at least three JPG/JPEG/PNG images. The training script skips categories with fewer than three usable images.

### 5. Train and evaluate the classifier

From the project folder with the virtual environment active:

```powershell
python train_model.py
```

Training uses 70% of the images per category for training, 15% for validation, and 15% for a held-out test set (rounding can make the exact counts vary). It may take a while, depending on your computer and dataset size. On completion, the script creates:

```text
model\plant_disease.keras
model\class_names.json
model\evaluation.json
```

`evaluation.json` contains the measured held-out test accuracy, class-level precision/recall/F1, confusion matrix, and split counts. These are results on this particular dataset split, not a promise of accuracy on garden photos. No model is claimed to be trained until this command finishes successfully.

Optional training controls:

```powershell
python train_model.py --epochs 5 --batch-size 16
python train_model.py --data-dir "D:\datasets\PlantVillage" --epochs 10
```

### 6. Start the website

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Leave the terminal running while you use the site. Press `Ctrl+C` in the terminal to stop Flask.

If you start Flask before training, the website and care dashboard still work. Upload analysis clearly reports that the model has not been trained and does not display a fabricated result.

## How to test the site

1. Start Flask and open `http://127.0.0.1:5000`.
2. Before training, verify the page shows the model setup notice; submitting a valid image must show that no prediction was made.
3. Train the model, restart Flask, and upload a well-lit JPG or PNG leaf photo.
4. Confirm the result includes the category, plant group, model confidence, report, care recommendations, and diagnosis disclaimer—or an uncertainty message if the score is below the threshold.
5. Try a text file renamed as `.jpg`, an unsupported extension, and an image over 8 MB. Each should produce an error instead of a prediction.
6. Review `model\evaluation.json` for the actual held-out evaluation measurements.

## Common issues

- **`py` is not recognized:** reinstall Python 3.11 and enable the PATH option, or use the full Python launcher provided by your installation.
- **Activation is blocked:** use the `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` command above, or switch the VS Code terminal profile to Command Prompt.
- **TensorFlow installation fails:** confirm you are using 64-bit Python 3.11 and run `python --version` in the activated `.venv`. Recreate the environment with `py -3.11 -m venv .venv` if it was created with a different Python version.
- **Dataset folder not found:** check that the class folders (not just a ZIP file or one enclosing directory) are directly under `dataset\PlantVillage\`, or pass the correct path with `--data-dir`.
- **Training reports fewer than two classes:** check that at least two category folders each contain three or more JPG/JPEG/PNG images.
- **Model files missing:** complete `python train_model.py` successfully before requesting predictions. Do not rename unrelated model files to the expected names.
- **Model files won't load:** train and generate `plant_disease.keras` and `class_names.json` together with this project's `train_model.py`; check the Flask terminal for the detailed error.
- **Port 5000 is in use:** stop the other local application using that port, or change the `app.run` port in the last lines of `app.py`.
- **The prediction seems unsure or incorrect:** try a sharp, well-lit photo of one leaf against a plain background. The training images and real garden photos can differ; always confirm concerning symptoms with an expert.

## Notes before deployment or hardware integration

This is a local learning project, not a production medical/agricultural diagnostic service. Before deploying it publicly, configure a private random Flask `SECRET_KEY`, turn off Flask debug mode, add production hosting and request protections, and review dataset licensing and privacy obligations. The Flask analysis route is isolated from the presentation, so a future ESP32/soil-moisture integration can be added as a separate sensor-data endpoint without requiring hardware now.

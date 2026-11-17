# Plant Disease Detection

Deep-learning classification of 15 Pepper, Potato and Tomato leaf classes
using the PlantVillage dataset.

## Run the dashboard

From your current folder, run:

```powershell
cd C:\Users\simo\Desktop\crop_disease_detection_notebooks
python -m venv crop_disease_notebooks\.venv
crop_disease_notebooks\.venv\Scripts\Activate.ps1
pip install -r crop_disease_notebooks\requirements.txt
streamlit run crop_disease_notebooks\crop_disease_dashboard\crop_disease_dashboard\app\app.py
```

Streamlit opens the dashboard in your browser. If it does not, open the URL
shown in the terminal, usually `http://localhost:8501`.

The dashboard requires these files, which are already included after training:

```text
models\
evaluation\
class_names.json
PlantVillage\
```

## Test the dashboard

1. Select **Disease Detection** in the sidebar.
2. Upload an image from [dashboard_test_images](dashboard_test_images).
3. Choose Custom CNN, MobileNetV2 or ResNet50.
4. Select **Analyse leaf**.

The test folder contains healthy and diseased Pepper, Potato and Tomato images.

## Train the models

Use Google Colab with a GPU:

1. Open `notebooks/05_complete_training_pipeline.ipynb`.
2. Upload it to [Google Colab](https://colab.research.google.com/).
3. Select **Runtime > Change runtime type > T4 GPU**.
4. Upload `PlantVillage.zip`.
5. Run the notebook from top to bottom.
6. Download the generated `models`, `evaluation` and `class_names.json` files.

The notebook trains and compares:

- Custom CNN
- MobileNetV2
- ResNet50

Training allows a maximum of 15 epochs with early stopping. Always report the
actual completed epochs printed by the notebook.

## Project files

```text
notebooks\05_complete_training_pipeline.ipynb  Main training workflow
models\                                         Saved Keras models
evaluation\                                     Metrics and confusion matrices
class_names.json                                Model class mapping
crop_disease_dashboard\                         Streamlit dashboard
Plant_Disease_Deep_Learning_Writeup.docx        Final project report
```

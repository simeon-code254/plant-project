from pathlib import Path
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image


def find_project_root() -> Path:
    env_root = os.getenv("CROP_DISEASE_PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    file_path = Path(__file__).resolve()
    for parent in [file_path.parent, *file_path.parents]:
        if (parent / "models").exists() or (parent / "evaluation").exists() or (parent / "class_names.json").exists():
            return parent
    return file_path.parents[2]


ROOT = find_project_root()
MODEL_DIR = ROOT / "models"
EVALUATION_DIR = ROOT / "evaluation"
CLASS_NAMES_FILE = ROOT / "class_names.json" if (ROOT / "class_names.json").exists() else (MODEL_DIR / "class_names.json")
DATASET_ROOT = ROOT / "PlantVillage"
TEST_IMAGES_DIR = ROOT / "dashboard_test_images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
HEALTHY_CLASS = "Pepper__bell___healthy"
BACTERIAL_CLASS = "Pepper__bell___Bacterial_spot"

st.set_page_config(
    page_title="PlantHealth AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --forest:#174936; --green:#2d7a55; --ink:#1f2933; --muted:#667085;
            --line:#dce7df; --surface:#ffffff; --healthy:#278052; --alert:#b65345; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background:#123b2d; border-right:1px solid #0e2d22; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#ffffff !important; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap:.35rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [role="radiogroup"] label *,
    [data-testid="stSidebar"] [role="radio"] { color:#ffffff !important;
      opacity:1 !important; border-radius:10px; }
    [data-testid="stSidebar"] [role="radiogroup"] label { padding:.65rem .75rem;
      border:1px solid transparent; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#255d47;
      border-color:#4f8d70; }
    [data-testid="stSidebar"] [role="radio"] > div:first-child { display:none; }
    [data-testid="stSidebar"] [role="radio"][aria-checked="true"] { background:#2d7a55;
      border-color:#76b493; box-shadow:inset 3px 0 #d9f2df; }
    [data-testid="stSidebar"] [role="radio"][aria-checked="true"] * { color:#ffffff !important; }
    [data-testid="stAppViewContainer"] { background:#fbfdfb; }
    [data-testid="stHeader"] { background:rgba(251,253,251,.92); }
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
      color:var(--ink) !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] { color:var(--ink) !important; }
    [data-testid="stMetricLabel"] p { color:#52665a !important; }
    [data-testid="stCaptionContainer"] { color:#52665a !important; }
    [data-testid="stAlert"] p { color:var(--ink) !important; }
    .brand-subtitle { color:#667085; font-size:.84rem; margin-top:-1rem; }
    .hero { background:#edf5ef; border:1px solid #d8e8dc; border-radius:20px;
            padding:2rem 2.2rem; margin-bottom:1.4rem; }
    .hero h1 { color:var(--forest); font-size:2.35rem; margin:0 0 .45rem; }
    .hero p { color:#52665a; font-size:1.05rem; margin:0; max-width:760px; }
    .detect-hero { background:linear-gradient(135deg,#123b2d 0%,#236b4c 65%,#4a9b70 100%);
                   border-radius:24px; padding:2.1rem 2.3rem; margin-bottom:1.2rem;
                   box-shadow:0 12px 30px rgba(18,59,45,.16); }
    .detect-hero .section-label { color:#bce8ca; }
    .detect-hero h1 { color:#fff; font-size:2.35rem; margin:.25rem 0 .45rem; }
    .detect-hero p { color:#e4f4e8; font-size:1.02rem; margin:0; max-width:760px; }
    .detect-steps { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.25rem; }
    .detect-step { background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.2);
                   border-radius:999px; padding:.38rem .72rem; color:#fff; font-size:.8rem; }
    .detect-panel { background:#fff; border:1px solid var(--line); border-radius:20px;
                    padding:1.25rem; box-shadow:0 6px 22px rgba(23,73,54,.06); height:100%; }
    .detect-panel h3 { margin-top:0; color:var(--forest); }
    .upload-hint { background:#f2f8f3; border:1px dashed #9bc7a8; border-radius:14px;
                   padding:.8rem 1rem; color:#52665a; font-size:.88rem; margin:.75rem 0; }
    .model-chip { display:inline-block; background:#e5f3e8; color:#216542; border-radius:999px;
                  padding:.28rem .65rem; font-size:.78rem; font-weight:700; margin:.2rem .2rem 0 0; }
    .preview-empty { min-height:340px; display:flex; flex-direction:column; align-items:center;
                     justify-content:center; text-align:center; background:#f7fbf8;
                     border:1px dashed #b9d5c1; border-radius:16px; padding:2rem; }
    .preview-empty .icon { font-size:3rem; margin-bottom:.55rem; }
    .preview-empty h3 { margin:.2rem 0; color:var(--forest); }
    .preview-empty p { color:#667085; max-width:300px; margin:0; }
    .prediction-card { border-radius:20px; padding:1.35rem 1.5rem; margin:1.2rem 0;
                       box-shadow:0 8px 24px rgba(23,73,54,.07); }
    .prediction-card h2 { margin:.35rem 0 .25rem; color:var(--forest); }
    .prediction-card p { margin:.35rem 0 0; color:#52665a; }
    .prediction-card .confidence { font-size:1.35rem; font-weight:800; color:var(--forest); }
    .result-healthy { background:#edf8f1; border:1px solid #bce2c8; }
    .result-alert { background:#fff2ef; border:1px solid #efc3bb; }
    .rank-title { color:var(--forest); font-size:1.05rem; font-weight:700; margin:.9rem 0 .45rem; }
    .rank-row { display:flex; align-items:center; gap:.65rem; margin:.45rem 0; }
    .rank-label { width:42%; color:#34443a; font-size:.86rem; }
    .rank-track { flex:1; height:9px; background:#e6efe8; border-radius:99px; overflow:hidden; }
    .rank-fill { height:100%; background:linear-gradient(90deg,#4a9b70,#174936); border-radius:99px; }
    .rank-value { width:52px; text-align:right; color:#52665a; font-size:.82rem; font-weight:700; }
    .section-label { color:var(--green); font-size:.78rem; font-weight:700;
                     letter-spacing:.12em; text-transform:uppercase; }
    .image-card { background:#fff; border:1px solid var(--line); border-radius:16px;
                  padding:.8rem; height:100%; box-shadow:0 4px 16px rgba(23,73,54,.05); }
    .image-card h4 { margin:.6rem 0 .25rem; color:var(--ink); }
    .image-card p { color:var(--muted); font-size:.9rem; margin:0; }
    .result-healthy { background:#edf8f1; border:1px solid #bce2c8; border-radius:16px; padding:1.2rem; }
    .result-alert { background:#fff2ef; border:1px solid #efc3bb; border-radius:16px; padding:1.2rem; }
    .status { display:inline-block; padding:.25rem .65rem; border-radius:99px; font-size:.78rem;
              font-weight:700; background:#e2f1e7; color:#216542; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line);
                                   border-radius:14px; padding:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def pretty_label(label: str) -> str:
    return label.replace("___", " — ").replace("__", " ").replace("_", " ").strip()


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_class_names() -> list[str]:
    value = load_json(CLASS_NAMES_FILE, [])
    if not value and (MODEL_DIR / "class_names.json").exists():
        value = load_json(MODEL_DIR / "class_names.json", [])
    return value if isinstance(value, list) else []


def available_models() -> dict[str, Path]:
    candidates = {
        "Custom CNN": MODEL_DIR / "custom_cnn.keras",
        "MobileNetV2": MODEL_DIR / "mobilenetv2.keras",
        "ResNet50": MODEL_DIR / "resnet50.keras",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def load_metrics() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(EVALUATION_DIR.glob("*_metrics.json")) if EVALUATION_DIR.exists() else []:
        data = load_json(path)
        if isinstance(data, dict):
            rows.append(data)
    return rows


@st.cache_data(show_spinner=False)
def all_dataset_image_paths() -> list[str]:
    if not DATASET_ROOT.exists():
        return []
    return sorted(
        str(path) for path in DATASET_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "PlantVillage/PlantVillage" not in str(path).replace("\\", "/")
    )


def dataset_image_paths(class_name: str | None = None) -> list[Path]:
    paths = [Path(path) for path in all_dataset_image_paths()]
    if class_name is not None:
        paths = [path for path in paths if path.parent.name == class_name]
    return paths


def discovered_classes() -> list[str]:
    names = load_class_names()
    if names:
        return names
    return sorted({path.parent.name for path in dataset_image_paths()})


def dataset_counts() -> pd.DataFrame:
    classes = discovered_classes()
    rows = [{"Class": pretty_label(name), "class_name": name,
             "Images": len(dataset_image_paths(name))} for name in classes]
    if not rows:
        return pd.DataFrame(columns=["Class", "class_name", "Images"])
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str):
    model_name = Path(model_path).name.lower()
    preprocess_input = (
        tf.keras.applications.resnet50.preprocess_input
        if "resnet50" in model_name
        else tf.keras.applications.mobilenet_v2.preprocess_input
    )
    custom_objects = {
        "preprocess_input": preprocess_input,
    }
    return tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )


@st.cache_data(show_spinner=False)
def representative_images(class_name: str, limit: int = 6) -> list[str]:
    paths = dataset_image_paths(class_name)
    if not paths and TEST_IMAGES_DIR.exists():
        test_img = TEST_IMAGES_DIR / f"{class_name}.jpg"
        if test_img.exists():
            return [str(test_img)]
    if not paths:
        return []
    indices = np.linspace(0, len(paths) - 1, min(limit, len(paths)), dtype=int)
    return [str(paths[index]) for index in indices]


def split_summary() -> dict[str, int]:
    data = load_json(EVALUATION_DIR / "split_assignments.json", {})
    paths = data.get("paths", {}) if isinstance(data, dict) else {}
    return {name: len(values) for name, values in paths.items()}


def model_table() -> pd.DataFrame:
    rows = []
    for data in load_metrics():
        rows.append({
            "Model": data.get("model", "Unknown"),
            "Accuracy": data.get("test_accuracy"),
            "Precision": data.get("macro_precision"),
            "Recall": data.get("macro_recall"),
            "F1": data.get("macro_f1"),
            "Test images": data.get("test_count"),
            "Epochs": len(data.get("history", {}).get("epoch", []))
            if isinstance(data.get("history"), dict)
            else data.get("epochs"),
            "Training seconds": data.get("train_seconds"),
            "Model size (MB)": (data.get("model_size_bytes") or 0) / 1_000_000,
            "details": data,
        })
    return pd.DataFrame(rows)


def render_reference_card(class_name: str, description: str) -> None:
    paths = representative_images(class_name, 1)
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    if paths:
        st.image(paths[0], use_container_width=True)
    else:
        st.info("Reference photograph unavailable.")
    st.markdown(f"<h4>{pretty_label(class_name)}</h4><p>{description}</p></div>",
                unsafe_allow_html=True)


def render_sidebar() -> str:
    st.sidebar.markdown("## ◒ PlantHealth")
    st.sidebar.markdown('<div class="brand-subtitle">Multi-Crop Health Intelligence</div>',
                        unsafe_allow_html=True)
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Disease Detection", "Model Performance", "Dataset Explorer", "About Project"],
        label_visibility="collapsed",
    )
    models_ok = bool(available_models())
    images_ok = bool(dataset_image_paths())
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"{'●' if models_ok else '○'} Models {'ready' if models_ok else 'missing'}")
    st.sidebar.markdown(f"{'●' if images_ok else '○'} Dataset {'connected' if images_ok else 'unavailable'}")
    st.sidebar.caption("PlantHealth AI · v1.0")
    return page


def render_overview() -> None:
    counts = dataset_counts()
    metrics = model_table()
    split = split_summary()
    best_accuracy = metrics["Accuracy"].max() if not metrics.empty else None
    st.markdown(
        '<div class="hero"><div class="section-label">Crop health analysis</div>'
        '<h1>Plant Disease Detection</h1>'
        '<p>Multi-crop leaf classification using trained deep-learning models and real '
        'PlantVillage photographs and measured test-set results.</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    total_images = int(counts["Images"].sum()) if not counts.empty and "Images" in counts else 0
    c1.metric("Dataset images", f"{total_images:,}")
    c2.metric("Classification categories", len(counts))
    c3.metric("Models available", len(available_models()))
    c4.metric("Best test accuracy", f"{best_accuracy:.1%}" if pd.notna(best_accuracy) else "N/A")

    st.markdown("### What the dataset contains")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        if not counts.empty and counts["Images"].sum() > 0:
            fig = px.bar(counts, x="Class", y="Images", color="Class",
                         color_discrete_sequence=["#b65345", "#278052"])
            fig.update_layout(showlegend=False, height=330, margin=dict(t=20, l=10, r=10, b=70),
                              yaxis_title="Photographs", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("The PlantVillage dataset folder is not available locally.")
    with right:
        st.markdown("#### Representative dataset photographs")
        cards = st.columns(3)
        for card, class_name in zip(cards, discovered_classes()[:3]):
            with card:
                render_reference_card(class_name, f"{len(dataset_image_paths(class_name)):,} available images.")

    st.markdown("### Model performance overview")
    if not metrics.empty:
        chart = px.bar(metrics, x="Model", y="Accuracy", color="Model",
                       color_discrete_sequence=["#7b8794", "#2d7a55", "#174936"],
                       text=metrics["Accuracy"].map(lambda value: f"{value:.1%}"))
        chart.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_title="Held-out test accuracy")
        st.plotly_chart(chart, use_container_width=True)
    if split:
        st.caption("Shared experiment split: " + " · ".join(f"{key}: {value:,}" for key, value in split.items()))


def preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize((224, 224))
    return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)


def render_detection() -> None:
    st.markdown(
        '<div class="detect-hero"><div class="section-label">Leaf disease detection</div>'
        '<h1>Check a leaf image</h1>'
        '<p>Upload a leaf photograph, choose a model and view its predicted class.</p>'
        '<div class="detect-steps"><span class="detect-step">1 · Upload image</span>'
        '<span class="detect-step">2 · Choose model</span><span class="detect-step">3 · Analyse result</span></div></div>',
        unsafe_allow_html=True,
    )
    models = available_models()
    classes = load_class_names()
    if not models or not classes:
        st.error("Model files and class_names.json are required for prediction.")
        return
    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        st.markdown('<div class="detect-panel"><h3>Start an analysis</h3>'
                    '<p class="upload-hint">For the best result, use one well-lit leaf on a clear background.</p>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a JPG, JPEG or PNG photograph",
                                     type=["jpg", "jpeg", "png"])
        selected = st.selectbox("Model", list(models))
        st.markdown("**Available models**")
        st.markdown("".join(f'<span class="model-chip">{name}</span>' for name in models),
                    unsafe_allow_html=True)
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.caption(f"Loaded: {uploaded.name} · {image.width} × {image.height}px")
        analyse = st.button("Analyse leaf", type="primary", disabled=uploaded is None)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        if not uploaded:
            st.markdown('<div class="detect-panel preview-empty"><div class="icon">🌿</div>'
                        '<h3>Your preview appears here</h3>'
                        '<p>Upload an image on the left to inspect it before running the model.</p></div>',
                        unsafe_allow_html=True)
            return
        if not analyse:
            st.markdown('<div class="detect-panel"><h3>Image preview</h3>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.info("Image ready. Click **Analyse leaf** when you are ready.")
            return
        st.markdown("### Prediction results")
        with st.spinner("Running inference..."):
            probabilities = load_model(str(models[selected])).predict(
                preprocess_image(image), verbose=0
            )[0]
        if len(probabilities) != len(classes):
            st.error("Model output and class mapping do not match.")
            return
        order = np.argsort(probabilities)[::-1]
        predictions = [(classes[int(index)], float(probabilities[int(index)])) for index in order]
        label, confidence = predictions[0]
        healthy = "healthy" in label.lower()
        style = "result-healthy" if healthy else "result-alert"
        title = "Healthy leaf" if healthy else "Disease pattern detected"
        interpretation = ("The model identifies visual patterns associated with a healthy leaf."
                          if healthy else
                          "The model identifies visual patterns associated with a trained disease class.")
        st.markdown(f'<div class="prediction-card {style}"><span class="status">{title}</span>'
                    f'<h2>{pretty_label(label)}</h2><div class="confidence">{confidence:.1%} confidence</div>'
                    f'<p>{interpretation}</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="rank-title">Top model signals</div>', unsafe_allow_html=True)
        rows = []
        for prediction_label, score in predictions[:5]:
            rows.append(
                f'<div class="rank-row"><div class="rank-label">{pretty_label(prediction_label)}</div>'
                f'<div class="rank-track"><div class="rank-fill" style="width:{score * 100:.2f}%"></div></div>'
                f'<div class="rank-value">{score:.1%}</div></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption(f"Model used: {selected} · Output classes: {len(classes)}")
        st.markdown("#### Uploaded image")
        st.image(image, caption=uploaded.name, width=220)


def render_performance() -> None:
    st.markdown("## Model Performance")
    st.caption("All values below are read from the generated held-out test evaluation artifacts.")
    table = model_table()
    if table.empty:
        st.warning("No evaluation JSON files were found.")
        return
    display = table[["Model", "Accuracy", "Precision", "Recall", "F1", "Test images", "Epochs",
                     "Training seconds", "Model size (MB)"]].copy()
    formatted = display.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1"]:
        formatted[column] = formatted[column].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
        )
    formatted["Epochs"] = formatted["Epochs"].map(
        lambda value: f"{value:.0f}" if pd.notna(value) else "N/A"
    )
    formatted["Test images"] = formatted["Test images"].map(
        lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A"
    )
    formatted["Training seconds"] = formatted["Training seconds"].map(
        lambda value: f"{value:.1f}" if pd.notna(value) else "N/A"
    )
    formatted["Model size (MB)"] = formatted["Model size (MB)"].map(
        lambda value: f"{value:.1f}" if pd.notna(value) else "N/A"
    )
    st.dataframe(formatted, use_container_width=True, hide_index=True)
    chart = px.bar(display.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1"]),
                   x="Model", y="value", color="variable", barmode="group",
                   labels={"value": "Score", "variable": "Metric"})
    chart.update_layout(yaxis_tickformat=".0%", height=390)
    st.plotly_chart(chart, use_container_width=True)
    selected = st.selectbox("Inspect model details", display["Model"].tolist())
    detail = next(row["details"] for row in table.to_dict("records") if row["Model"] == selected)
    matrix = np.asarray(detail.get("confusion_matrix", []))
    labels = detail.get("classes", [])
    if matrix.size and labels:
        matrix_fig = px.imshow(matrix, x=[pretty_label(x) for x in labels],
                                y=[pretty_label(x) for x in labels], text_auto=True,
                                color_continuous_scale="Greens", aspect="auto")
        matrix_fig.update_layout(height=620, xaxis_title="Predicted class", yaxis_title="True class")
        st.plotly_chart(matrix_fig, use_container_width=True)
    report = detail.get("per_class_report")
    if report:
        with st.expander("Classification report"):
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    st.info("Training and validation curves are not included in the exported metric JSON files; "
            "they can be added to the notebook artifacts for display here.")


def render_explorer() -> None:
    st.markdown("## Dataset Explorer")
    counts = dataset_counts()
    total = int(counts["Images"].sum()) if not counts.empty and "Images" in counts else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset images", f"{total:,}")
    c2.metric("Categories", len(counts))
    c3.metric("Image types", "JPG / JPEG / PNG")
    classes = discovered_classes()
    if not classes:
        st.warning("No categories were found under the configured dataset directory or class_names.json.")
        return
    selected = st.selectbox("Filter category", ["All classes", *[pretty_label(x) for x in classes]])
    class_name = None if selected == "All classes" else classes[[pretty_label(x) for x in classes].index(selected)]
    paths = dataset_image_paths(class_name)
    st.caption(f"Showing {min(len(paths), 24):,} representative photographs from {len(paths):,} available.")
    if not paths:
        st.warning("No photographs were found under the configured dataset directory.")
        return
    for start in range(0, min(len(paths), 24), 4):
        columns = st.columns(4)
        for column, path in zip(columns, paths[start:start + 4]):
            with column:
                st.image(str(path), use_container_width=True)
                st.caption(pretty_label(path.parent.name))
    st.markdown("### Split distribution")
    split = split_summary()
    if split:
        split_df = pd.DataFrame({"Split": list(split), "Images": list(split.values())})
        st.plotly_chart(px.bar(split_df, x="Split", y="Images", color="Split",
                                color_discrete_sequence=["#174936", "#2d7a55", "#9ab8a3"]),
                        use_container_width=True)


def render_about() -> None:
    st.markdown("## About Project")
    st.markdown("### Project overview")
    st.write("PlantHealth AI classifies crop leaf photographs using deep-learning models "
             "trained on the supplied PlantVillage dataset.")
    st.markdown("### Problem and objectives")
    st.write("The project investigates automated classification across the Pepper, Potato and Tomato "
             "classes represented in the supplied dataset, while benchmarking a custom CNN against transfer learning.")
    st.markdown("### Methodology")
    st.write("Dataset → image preprocessing → stratified split → model training → held-out evaluation → disease prediction")
    st.markdown("### Models and metrics")
    st.write("Custom CNN, MobileNetV2 and ResNet50 are evaluated using test accuracy, macro precision, "
             "macro recall, macro F1-score and confusion matrices.")
    st.markdown("### Dataset scope")
    st.write("The models recognise the 15 Pepper, Potato and Tomato classes represented in "
             "the supplied PlantVillage dataset.")


page = render_sidebar()
if page == "Overview":
    render_overview()
elif page == "Disease Detection":
    render_detection()
elif page == "Model Performance":
    render_performance()
elif page == "Dataset Explorer":
    render_explorer()
else:
    render_about()

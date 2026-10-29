
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Распознавание цифр — SL и SSL", page_icon="🔢", layout="centered")
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_NAMES = {
    "Логистическая регрессия": "lr",
    "Метод опорных векторов (SVM)": "svm",
    "Случайный лес (Random Forest)": "rf",
    "Метод ближайших соседей (KNN)": "knn",
}
FRACTIONS = {"1%": "1", "2,5%": "2.5", "5%": "5", "10%": "10"}
BEST_THRESHOLDS = {
    "lr": {
        "1": 0.7,
        "2.5": 0.9,
        "5": 0.9,
        "10": 0.9
    },
    "svm": {
        "1": 0.8,
        "2.5": 0.8,
        "5": 0.8,
        "10": 0.8
    },
    "rf": {
        "1": 0.9,
        "2.5": 0.9,
        "5": 0.9,
        "10": 0.9,
    },
    "knn": {
        "1": 0.7,
        "2.5": 0.7,
        "5": 0.7,
        "10": 0.7,
    },
}

@st.cache_resource(show_spinner=False)
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)

def prepare_digit(rgba):
    # Холст: белая линия на чёрном фоне, как в MNIST.
    rgb = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").convert("RGB")
    gray = ImageOps.grayscale(rgb)
    a = np.asarray(gray)
    # Обрезаем пустые поля, вписываем цифру в 20x20, центрируем в 28x28.
    ys, xs = np.where(a > 20)
    if len(xs) == 0:
        raise ValueError("Сначала нарисуйте цифру на холсте.")
    cropped = gray.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    w, h = cropped.size
    ratio = 20.0 / max(w, h)
    resized = cropped.resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.Resampling.LANCZOS)
    digit = Image.new("L", (28, 28), 0)
    digit.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    x = np.asarray(digit, dtype=np.float32).reshape(1, 784) / 255.0
    return digit, x

def prediction(model, x):
    probabilities = model.predict_proba(x)[0]
    idx = int(np.argmax(probabilities))
    label = int(model.classes_[idx])
    confidence = float(probabilities[idx])
    return label, confidence

st.title("Распознавание рукописных цифр")
st.write("Сравнение обучения с учителем (SL) и частично контролируемого обучения (SSL).")

model_label = st.selectbox("Алгоритм", list(MODEL_NAMES))
fraction_label = st.selectbox("Доля размеченных изображений при обучении", list(FRACTIONS), index=2)

algo = MODEL_NAMES[model_label]
fraction = FRACTIONS[fraction_label]

ssl_training_threshold = BEST_THRESHOLDS[algo][fraction]

st.caption(f"Порог уверенности для псевдоразметки SSL подбирается автоматически под выбранные условия для достижения наилучшего результата: {ssl_training_threshold:.1f}")
st.subheader("Нарисуйте цифру от 0 до 9")

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0

canvas = st_canvas(
    fill_color="rgba(0, 0, 0, 0)", stroke_width=18, stroke_color="#FFFFFF",
    background_color="#000000", width=280, height=280,
    drawing_mode="freedraw", update_streamlit=True,
    display_toolbar=False, key=f"digit_canvas_{st.session_state.canvas_version}",
)

if st.button("Очистить холст"):
    st.session_state.canvas_version += 1
    st.rerun()

if st.button("Распознать", type="primary", use_container_width=True):
    if canvas.image_data is None:
        st.warning("Сначала нарисуйте цифру.")
    else:
        try:
            digit, x = prepare_digit(canvas.image_data)
            algo = MODEL_NAMES[model_label]
            fraction = FRACTIONS[fraction_label]

            # Автоматически выбираем оптимальный порог SSL
            ssl_training_threshold = BEST_THRESHOLDS[algo][fraction]

            sl_file = f"{algo}_labeled_{fraction}_sl.joblib"
            ssl_file = f"{algo}_labeled_{fraction}_ssl_{ssl_training_threshold:.1f}.joblib"
            missing = [name for name in (sl_file, ssl_file) if not (MODEL_DIR / name).is_file()]
            if missing:
                st.error("Не найдены сохранённые модели: " + ", ".join(missing))
                st.info("Сначала выполните ячейку экспорта моделей в обучающем ноутбуке.")
            else:
                sl = load_model(sl_file)
                ssl = load_model(ssl_file)
                sl_result = prediction(sl, x)
                ssl_result = prediction(ssl, x)
                st.image(digit.resize((140, 140), Image.Resampling.NEAREST), caption="Изображение после обработки")
                col1, col2 = st.columns(2)
                for col, name, result in ((col1, "SL", sl_result), (col2, "SSL", ssl_result)):
                    with col:
                        st.subheader(name)
                        st.metric("Цифра", result[0])
                        st.metric("Уверенность", f"{result[1]:.1%}")
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Не удалось выполнить распознавание: {exc}")

st.caption("SL и SSL используют одну долю исходных меток. Уверенность — оценка модели, а не гарантия правильного ответа.")
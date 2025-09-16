import streamlit as st
from PIL import Image
import io
import requests
from money_counter import MoneyCounter

st.set_page_config(page_title="MoneyCounter", layout="centered")
st.title("Russian Money Counter")
st.caption("Загрузи фото или URL, чтобы получить разметку и сумму рублей")

# загрузка модели
model_path = "models/model.pt"
if "counter" not in st.session_state:
    st.session_state.counter = MoneyCounter(model_path=model_path, device="cpu")

uploaded_file = st.file_uploader("Загрузить файл", type=['jpg','jpeg','png'])
url_input = st.text_input("или вставь URL картинки")

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif url_input:
    try:
        resp = requests.get(url_input, timeout=8)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")

if image is not None:
    st.image(image, caption="Исходное изображение", use_column_width=True)

    if st.button("Посчитать сумму"):
        annotated_img, total = st.session_state.counter.process(image)
        st.success(f"Итого: {total} руб.")
        st.image(annotated_img, caption="Размеченное изображение", use_column_width=True)

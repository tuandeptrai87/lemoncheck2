import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Đặt set_page_config đầu tiên
st.set_page_config(page_title="🌿 Nhận diện bệnh cây", page_icon="🌿", layout="centered")

# Load model (cache để nhanh hơn)
@st.cache_resource
def load_vgg_model():
    return load_model("F:\\LEARNING\\nam 4\\Chuyen_de_2\\merchin\\vgg16_4.h5")

@st.cache_resource
def load_resnet_model():
    return load_model("F:\\LEARNING\\nam 4\\Chuyen_de_2\\merchin\\resnet50_leaf_disease.h5")

@st.cache_resource
def load_mobilenet_model():
    return load_model("F:\\LEARNING\\nam 4\\Chuyen_de_2\\merchin\\mobilenetv2_leaf_disease.h5")

vgg_model = load_vgg_model()
resnet_model = load_resnet_model()
mobilenet_model = load_mobilenet_model()

# Danh sách nhãn
labels = [
    "Bệnh nấm lá",
    "Bệnh bạc lá do vi khuẩn",
    "Bệnh loét cam quýt",
    "Virus xoăn lá",
    "Bệnh thiếu đinh dưỡng lá",
    "Lá bị khô",
    "Lá khỏe mạnh",
    "Nấm bồ hóng",
    "Vết thâm do bọ phá hoại",
]

# Thông tin chi tiết
disease_details = {
    "Bệnh nấm lá": "Đây là một bệnh do nấm gây ra, phổ biến trên nhiều loại cây trồng, gây đốm nâu hoặc đen trên lá và quả.",
    "Bệnh bạc lá do vi khuẩn": "Bệnh do vi khuẩn gây ra, làm úa vàng và rụng lá sớm.",
    "Bệnh loét cam quýt": "Bệnh do vi khuẩn Xanthomonas citri gây ra, gây loét trên vỏ quả và cành.",
    "Virus xoăn lá": "Virus gây biến dạng, xoăn vặn lá, làm giảm khả năng quang hợp.",
    "Bệnh thiếu đinh dưỡng lá": "Do thiếu các chất dinh dưỡng thiết yếu như Nitrogen, Phosphorus, Potassium, gây vàng lá.",
    "Lá bị khô": "Lá khô do thiếu nước hoặc nhiễm nấm, vi khuẩn.",
    "Lá khỏe mạnh": "Lá bình thường, xanh mượt, không có dấu hiệu bệnh.",
    "Nấm bồ hóng": "Nấm gây các đốm đen trên lá khi điều kiện ẩm ướt.",
    "Vết thâm do bọ phá hoại": "Côn trùng nhỏ gây vàng lá bằng cách hút nhựa cây.",
}

# Hàm dự đoán
def predict_image(image: Image.Image, model_choice: str):
    image = image.resize((180, 180))
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array_expanded = img_array.reshape(1, 180, 180, 3)

    if model_choice == "VGG16":
        prediction = vgg_model.predict(img_array_expanded)[0]
    elif model_choice == "ResNet50":
        prediction = resnet_model.predict(img_array_expanded)[0]
    elif model_choice == "MobileNetV2":
        prediction = mobilenet_model.predict(img_array_expanded)[0]
    else:
        prediction = np.zeros(len(labels))

    predicted_idx = np.argmax(prediction)
    predicted_label = labels[predicted_idx]
    confidence = prediction[predicted_idx]
    return predicted_label, confidence, prediction

# ------------------- Giao diện chính -------------------

st.title("🌿 Ứng dụng Phát hiện Bệnh trên Lá Cây")
st.caption("Hệ thống nhận diện bằng AI với nhiều mô hình lựa chọn")
st.divider()

# 1. Upload ảnh
uploaded_file = st.file_uploader("📤 Tải lên ảnh lá cây cần nhận diện", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Ảnh đã tải lên", use_container_width=True)

    # 2. Chọn mô hình
    st.subheader("🧠 Chọn mô hình dự đoán")
    model_option = st.selectbox("", ("VGG16", "ResNet50", "MobileNetV2"))

    # 3. Nút Dự đoán
    if st.button("🔍 Dự đoán bệnh"):
        with st.spinner('⏳ Đang phân tích...'):
            predicted_label, confidence, full_prediction = predict_image(image, model_option)

        st.success(f"🩺 **Kết quả:** {predicted_label}")
        st.info(disease_details.get(predicted_label, "Không có thông tin chi tiết."))

        # 4. Thanh tiến trình
        st.progress(int(confidence * 100))

        # 5. Biểu đồ xác suất (dọc)
        st.subheader("📊 Xác suất dự đoán các loại bệnh")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(labels, full_prediction, color='skyblue')
        ax.set_ylabel('Xác suất')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

st.divider()

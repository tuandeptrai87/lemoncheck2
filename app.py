import os
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import sqlite3
from datetime import datetime

# === Xác định thư mục cơ sở ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Thư mục chứa templates và static ===
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

# === Đường dẫn đến database và model ===
DATABASE_PATH = os.path.join(DOCS_DIR, 'history.db')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# === Khởi tạo Flask ứng dụng ===
app = Flask(
    __name__,
    template_folder=DOCS_DIR,
    static_folder=DOCS_DIR,
    static_url_path=''  # phục vụ static tại /<filename>
)

# --- 1. Khởi tạo database nếu chưa có ---
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model TEXT,
            label TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- 2. Load các model từ thư mục model/ ---
vgg_model = load_model(os.path.join(MODEL_DIR, 'vgg16_4.h5'))
resnet_model = load_model(os.path.join(MODEL_DIR, 'resnet50_leaf_disease.h5'))
mobilenet_model = load_model(os.path.join(MODEL_DIR, 'mobilenetv2_leaf_disease.h5'))

# --- 3. Nhãn và thông tin chi tiết bệnh ---
labels = [
    "Bệnh nấm lá", "Bệnh bạc lá do vi khuẩn", "Bệnh loét cam quýt", "Virus xoăn lá",
    "Bệnh thiếu dinh dưỡng lá", "Lá bị khô", "Lá khỏe mạnh", "Nấm bồ hóng", "Vết thâm do bọ phá hoại"
]

disease_details = {
    "Bệnh nấm lá": {
        "cause": "Nấm phát triển mạnh trong điều kiện ẩm cao, bào tử lây lan qua nước bắn và gió.",
        "prevention": "Loại bỏ lá bệnh, giữ vườn khô thoáng, phun thuốc chống nấm theo hướng dẫn."
    },
    "Bệnh bạc lá do vi khuẩn": {
        "cause": "Vi khuẩn Xanthomonas gây bệnh, thường lây lan qua nước mưa và dụng cụ cắt tỉa.",
        "prevention": "Vệ sinh dụng cụ, loại bỏ lá bệnh, phun thuốc kháng khuẩn định kỳ."
    },
    "Bệnh loét cam quýt": {
        "cause": "Bệnh do vi khuẩn gây loét vỏ, thường xuất hiện khi ẩm độ cao.",
        "prevention": "Loại bỏ quả bệnh, giữ khoảng cách cây, phun thuốc bảo vệ thực vật."
    },
    "Virus xoăn lá": {
        "cause": "Virus lây truyền qua côn trùng chích hút (rầy, sâu).",
        "prevention": "Diệt côn trùng trung gian, sử dụng giống kháng virus."
    },
    "Bệnh thiếu dinh dưỡng lá": {
        "cause": "Thiếu một số nguyên tố vi lượng như Mg, N, Fe.",
        "prevention": "Bón phân cân đối, kiểm tra pH đất, bổ sung vi lượng."
    },
    "Lá bị khô": {
        "cause": "Thiếu nước hoặc do điều kiện môi trường khô nóng.",
        "prevention": "Tưới đủ ẩm, phủ gốc giữ độ ẩm, che nắng."
    },
    "Lá khỏe mạnh": {
        "cause": "Không phát hiện triệu chứng bệnh trên lá.",
        "prevention": "Tiếp tục chăm sóc bình thường."
    },
    "Nấm bồ hóng": {
        "cause": "Nấm phát triển trên bề mặt lá, thường xuất hiện khi có mật đường và chân không thoáng.",
        "prevention": "Rửa sạch lá, giảm độ ẩm, phun thuốc diệt nấm."
    },
    "Vết thâm do bọ phá hoại": {
        "cause": "Côn trùng cắn phá làm chấm thâm trên lá.",
        "prevention": "Sử dụng thuốc trừ sâu sinh học, bẫy đèn LED."
    }
}

# --- 4. Route chính hiển thị index ---
@app.route('/')
def index():
    return render_template('index.html')

# --- 5. Xử lý dự đoán ---
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    model_name = request.form.get('model')

    # Đọc và tiền xử lý ảnh
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((180, 180))
    arr = np.array(img, dtype='float32') / 255.0
    arr = arr.reshape(1, 180, 180, 3)

    # Chọn model và dự đoán
    if model_name == 'VGG16':
        preds = vgg_model.predict(arr)[0]
    elif model_name == 'ResNet50':
        preds = resnet_model.predict(arr)[0]
    else:
        preds = mobilenet_model.predict(arr)[0]

    idx = int(np.argmax(preds))
    label = labels[idx]
    confidence = float(preds[idx])
    detail = disease_details.get(label, {})

    # Lưu lịch sử vào DB
    timestamp = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute(
        'INSERT INTO history (timestamp, model, label, confidence) VALUES (?, ?, ?, ?)',
        (timestamp, model_name, label, confidence)
    )
    conn.commit()
    conn.close()

    return jsonify({
        'label': label,
        'confidence': confidence,
        'probs': [float(p) for p in preds],
        'cause': detail.get('cause', ''),
        'prevention': detail.get('prevention', '')
    })

# --- 6. Route lịch sử ---
@app.route('/history')
def history():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('SELECT timestamp, model, label, confidence FROM history ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)

# --- 7. Chạy ứng dụng ---
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import sqlite3
from datetime import datetime

# Khởi tạo Flask, dùng thư mục docs cho cả templates và static
app = Flask(
    __name__,
    template_folder='docs',
    static_folder='docs',
    static_url_path=''  # Phục vụ các file static (style.css, script.js) tại đường dẫn gốc
)

# Path đến file DB trong docs
database_path = 'SourceCode\\history.db'

# --- 1. Khởi tạo database và bảng nếu chưa có ---
def init_db():
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            model       TEXT,
            label       TEXT,
            confidence  REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- 2. Load các model ---
vgg_model       = load_model(r"F:\LEARNING\nam 4\Chuyen_de_2\merchin\SourceCode\model\vgg16_4.h5")
resnet_model    = load_model(r"F:\LEARNING\nam 4\Chuyen_de_2\merchin\SourceCode\model\resnet50_leaf_disease.h5")
mobilenet_model = load_model(r"F:\LEARNING\nam 4\Chuyen_de_2\merchin\SourceCode\model\mobilenetv2_leaf_disease.h5")

# --- 3. Labels và details ---
labels = [
    "Bệnh nấm lá", "Bệnh bạc lá do vi khuẩn", "Bệnh loét cam quýt", "Virus xoăn lá",
    "Bệnh thiếu đinh dưỡng lá", "Lá bị khô", "Lá khỏe mạnh", "Nấm bồ hóng", "Vết thâm do bọ phá hoại"
]
disease_details = {
    "Bệnh nấm lá": {
        "cause": "Nấm phát triển mạnh trong điều kiện ẩm cao, bào tử lây lan qua nước bắn và gió.",
        "prevention": "Loại bỏ lá bệnh, giữ vườn khô thoáng, phun thuốc chống nấm theo hướng dẫn."
    },
    # ... phần còn lại giống như trước ...
}

# --- 4. Route chính ---
@app.route('/')
def index():
    # index.html trong docs/index.html
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    model_name = request.form['model']

    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((180, 180))
    arr = np.array(img, dtype='float32') / 255.0
    arr = arr.reshape(1, 180, 180, 3)

    if model_name == 'VGG16':
        preds = vgg_model.predict(arr)[0]
    elif model_name == 'ResNet50':
        preds = resnet_model.predict(arr)[0]
    else:
        preds = mobilenet_model.predict(arr)[0]

    idx = int(np.argmax(preds))
    label = labels[idx]
    confidence = float(preds[idx])
    detail = disease_details[label]

    # Lưu history
    timestamp = datetime.utcnow().isoformat()
    conn = sqlite3.connect(database_path)
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
        'cause': detail['cause'],
        'prevention': detail['prevention']
    })

@app.route('/history')
def history():
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute('SELECT timestamp, model, label, confidence FROM history ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)

if __name__ == '__main__':
    app.run(debug=True)
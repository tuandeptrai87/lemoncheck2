from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import sqlite3
from datetime import datetime

# Khởi tạo Flask, trỏ templates và static vào thư mục docs
app = Flask(
    __name__,
    template_folder='docs/templates',
    static_folder='docs/static'
)

# Path đến file DB
DATABASE = 'SourceCode\\history.db'

# --- 1. Khởi tạo database và bảng nếu chưa có ---
def init_db():
    conn = sqlite3.connect(DATABASE)
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
    "Bệnh nấm lá","Bệnh bạc lá do vi khuẩn","Bệnh loét cam quýt","Virus xoăn lá",
    "Bệnh thiếu đinh dưỡng lá","Lá bị khô","Lá khỏe mạnh","Nấm bồ hóng","Vết thâm do bọ phá hoại"
]
disease_details = {
    "Bệnh nấm lá": {
        "cause": "Nấm phát triển mạnh trong điều kiện ẩm cao, bào tử lây lan qua nước bắn và gió.",
        "prevention": "Loại bỏ lá bệnh, giữ vườn khô thoáng, phun thuốc chống nấm theo hướng dẫn."
    },
    "Bệnh bạc lá do vi khuẩn": {
        "cause": "Vi khuẩn Xanthomonas tấn công qua vết thương nhỏ trên lá, gây hoại tử mô lá.",
        "prevention": "Tránh để lá ướt lâu, phun thuốc chứa đồng hoặc kháng sinh sinh học."
    },
    "Bệnh loét cam quýt": {
        "cause": "Vi khuẩn Xanthomonas citri xâm nhập, hình thành vết loét trên vỏ quả và cành.",
        "prevention": "Cắt bỏ cành bệnh, vệ sinh vườn, phun thuốc kháng khuẩn định kỳ."
    },
    "Virus xoăn lá": {
        "cause": "Virus lây truyền qua côn trùng chích hút khiến lá biến dạng, xoăn vặn.",
        "prevention": "Kiểm soát côn trùng trung gian, trồng giống kháng virus, loại bỏ cây nhiễm."
    },
    "Bệnh thiếu đinh dưỡng lá": {
        "cause": "Cây thiếu Nitrogen, Phosphorus, Potassium hoặc vi khoáng dẫn đến vàng lá, chậm lớn.",
        "prevention": "Bón cân đối phân hóa học và hữu cơ, theo định kỳ xét nghiệm đất."
    },
    "Lá bị khô": {
        "cause": "Thiếu nước tưới, nắng gắt hoặc nấm tấn công làm lá héo, giòn.",
        "prevention": "Tưới đủ ẩm, che nắng khi quá gắt, phun thuốc diệt nấm nếu có dấu hiệu."
    },
    "Lá khỏe mạnh": {
        "cause": "–",
        "prevention": "Duy trì chăm sóc đúng kỹ thuật: tưới, bón phân, phòng trừ dịch hại."
    },
    "Nấm bồ hóng": {
        "cause": "Nấm bám trên lớp bồ hóng bám vào lá, phát triển khi ẩm ướt.",
        "prevention": "Vệ sinh bồ hóng, tăng cường thông gió, phun thuốc diệt nấm bồ hóng."
    },
    "Vết thâm do bọ phá hoại": {
        "cause": "Côn trùng nhỏ hút nhựa, đục lỗ trên lá, tạo vết thâm đen hoặc nâu.",
        "prevention": "Phun chế phẩm sinh học hoặc hạ ngưỡng áp lực phun thuốc trừ sâu phù hợp."
    },
}

# --- 4. Route chính và các chức năng ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file       = request.files['file']
    model_name = request.form['model']

    # Tiền xử lý ảnh
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((180, 180))
    arr = np.array(img, dtype='float32') / 255.0
    arr = arr.reshape(1, 180, 180, 3)

    # Dự đoán
    if model_name == 'VGG16':
        preds = vgg_model.predict(arr)[0]
    elif model_name == 'ResNet50':
        preds = resnet_model.predict(arr)[0]
    else:
        preds = mobilenet_model.predict(arr)[0]

    idx        = int(np.argmax(preds))
    label      = labels[idx]
    confidence = float(preds[idx])
    detail     = disease_details[label]

    # Lưu vào database
    timestamp = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        'INSERT INTO history (timestamp, model, label, confidence) VALUES (?, ?, ?, ?)',
        (timestamp, model_name, label, confidence)
    )
    conn.commit()
    conn.close()

    # Trả về JSON
    return jsonify({
        'label'      : label,
        'confidence' : confidence,
        'probs'      : [float(p) for p in preds],
        'cause'      : detail['cause'],
        'prevention' : detail['prevention']
    })

@app.route('/history')
def history():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT timestamp, model, label, confidence FROM history ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)

if __name__ == '__main__':
    app.run(debug=True)

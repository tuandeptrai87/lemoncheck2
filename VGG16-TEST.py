import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
import cv2

MODEL_PATH = "Lemon\\vgg16_4.h5"
# Load model
model2 = load_model(MODEL_PATH)

# Định nghĩa các nhãn
labels = [
    "Bệnh nấm lá",
    "Bệnh bạc lá do vi khuẩn",
    "Bệnh loét cam quýt",
    "Virus xoăn lá",
    "Bệnh thiếu đinh dưỡng lá",
    "Lá bị khô",
    "Lá khỏe mạnh",
    "Nấm bồ hóng",
    "Bọ nhện đỏ",
]

# Thông tin chi tiết về các bệnh
disease_details = {
    'Bệnh nấm lá': 'Đây là một bệnh do nấm gây ra, rất phổ biến trên nhiều loại cây trồng, đặc biệt là cây ăn quả, rau màu và cây cảnh. Bệnh gây ra vết đốm màu nâu hoặc đen trên lá, quả, cành non hoặc thân cây. Khi nặng, nó có thể làm rụng lá, thối quả và ảnh hưởng nghiêm trọng đến năng suất. Tên khoa học của một số loài nấm gây bệnh thán thư thường thuộc chi Colletotrichum (ví dụ: Colletotrichum gloeosporioides, Colletotrichum acutatum...).',
    'Bệnh bạc lá do vi khuẩn': 'Bệnh do vi khuẩn gây ra, thường ảnh hưởng đến các loại cây ăn quả. Bệnh làm cho lá bị úa vàng và rụng sớm.',
    'Bệnh loét cam quýt': 'Bệnh do vi khuẩn Xanthomonas citri gây ra, làm xuất hiện các vết loét trên vỏ quả, gây giảm năng suất.',
    'Virus xoăn lá': 'Virus gây ra sự biến dạng và xoăn vặn của lá, làm giảm khả năng quang hợp và năng suất của cây.',
    'Bệnh thiếu đinh dưỡng lá': 'Bệnh do thiếu các chất dinh dưỡng quan trọng như Nitrogen, Phosphorus, Potassium, gây ra hiện tượng vàng lá và sự phát triển yếu.',
    'Lá bị khô': 'Lá cây bị khô do thiếu nước hoặc do bệnh nấm hoặc vi khuẩn gây ra, ảnh hưởng đến sự phát triển của cây.',
    'Lá khỏe mạnh': 'Lá cây bình thường, không có dấu hiệu bệnh, phát triển tốt và xanh mượt.',
    'Nấm bồ hóng': 'Nấm gây ra các đốm đen trên lá, thường do nấm gây ra khi có điều kiện ẩm ướt.',
    'Bọ nhện đỏ': 'Côn trùng nhỏ, gây hại cho lá cây bằng cách hút nhựa, làm cho lá bị vàng và có đốm.',
}

# Initialise GUI
top = tk.Tk()
top.geometry('1180x650')  # Tăng kích thước của cửa sổ
top.title('Phát hiện bệnh cây khoai tây')
top.configure(background='#D5E8D4')  # Chỉnh lại màu nền giao diện

label = Label(top, background='#D5E8D4', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Ô chứa thông tin chi tiết về bệnh
disease_info = Text(top, height=8, width=70, wrap=WORD, font=('arial', 12), padx=10, pady=10, bd=3, relief=SOLID)
disease_info.configure(state='disabled', bg='#F5F5F5')  # Đặt nền ô thông tin và không cho chỉnh sửa

def classify(file_path):
    arr = cv2.imread(file_path)
    if arr is not None:
        arr = cv2.resize(arr, (180, 180))
        arr = arr.reshape(1, 180, 180, 3)
    prediction = model2.predict(arr)
    # Giải mã nhãn dự đoán
    predicted_label = labels[np.argmax(prediction)]
    print(predicted_label)
    
    # Hiển thị kết quả dự đoán
    label.configure(foreground='#011638', text=predicted_label)

    # Hiển thị thông tin chi tiết bệnh
    disease_info.configure(state='normal')  # Cho phép chỉnh sửa để cập nhật nội dung
    disease_info.delete(1.0, END)  # Xóa thông tin cũ
    disease_info.insert(END, disease_details.get(predicted_label, "Thông tin không có sẵn."))
    disease_info.configure(state='disabled')  # Không cho phép chỉnh sửa sau khi cập nhật

def show_classify_button(file_path):
    classify_b = Button(top, text="Detection", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#4CAF50', foreground='white', font=('arial', 12, 'bold'))  # Thay đổi màu nút
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        disease_info.configure(state='normal')
        disease_info.delete(1.0, END)  # Xóa thông tin trước khi tải ảnh mới
        disease_info.configure(state='disabled')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#4CAF50', foreground='white', font=('arial', 12, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
disease_info.pack(side=BOTTOM, expand=True)  # Đặt ô thông tin chi tiết xuống dưới
heading = Label(top, text="Phát hiện bệnh cây khoai tây", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#D5E8D4', foreground='#364156')
heading.pack()

top.mainloop()
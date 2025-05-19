import os
import shutil
import random

# Thư mục gốc chứa dữ liệu ban đầu
input_root = 'SourceCode\\Original Dataset'
output_root = 'SourceCode\\Leemon'  # Kết quả sẽ được lưu ở đây
splits = ['train', 'val', 'test']
ratios = [0.8, 0.1, 0.1]

# Lấy danh sách class (tên thư mục con)
classes = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

# Tạo thư mục output và thư mục con theo class
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

# Chia ảnh cho từng class
for cls in classes:
    class_dir = os.path.join(input_root, cls)
    images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    random.shuffle(images)

    total = len(images)
    train_end = int(ratios[0] * total)
    val_end = train_end + int(ratios[1] * total)

    split_data = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split in splits:
        for img_name in split_data[split]:
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(output_root, split, cls, img_name)
            shutil.copy2(src, dst)

    print(f"✅ Đã chia lớp '{cls}': {len(images)} ảnh "
          f"→ train: {len(split_data['train'])}, val: {len(split_data['val'])}, test: {len(split_data['test'])}")

print("\n🎉 Hoàn tất chia dữ liệu!")

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

# Settings
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 9

data_dir = r"F:\LEARNING\nam 4\Chuyen_de_2\merchin\SourceCode\Leemon"
train_dir = os.path.join(data_dir, 'train')
val_dir   = os.path.join(data_dir, 'val')
test_dir  = os.path.join(data_dir, 'test')
plot_dir  = 'plots'
model_path = os.path.join(data_dir, 'mobilenetv2_leaf_disease.h5')

# Ensure output directories
os.makedirs(plot_dir, exist_ok=True)

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ TensorFlow đang sử dụng GPU.")
else:
    print("⚠️ Không tìm thấy GPU, TensorFlow sẽ dùng CPU.")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
generator_train = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)
generator_val = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)
generator_test = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# Build MobileNetV2 model
base_model = MobileNetV2(
    weights='imagenet', include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
tf.keras.backend.clear_session()
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)
model.summary()

# Train and capture history
history = model.fit(
    generator_train,
    epochs=EPOCHS,
    validation_data=generator_val
)

# Save trained model
model.save(model_path)
print(f"✅ Training hoàn tất! Model đã lưu thành '{model_path}'")

# Utility to plot & save metrics
def save_plot(metric, title, filename):
    plt.figure()
    plt.plot(history.history[metric], label=f'train_{metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.tight_layout()
    path = os.path.join(plot_dir, filename)
    plt.savefig(path)
    plt.show()

# Generate and save plots for key metrics
save_plot('accuracy', 'Model Accuracy over Epochs', 'accuracy_mobilenetv2.png')
save_plot('loss', 'Model Loss over Epochs', 'loss_mobilenetv2.png')
save_plot('precision', 'Model Precision over Epochs', 'precision_mobilenetv2.png')
save_plot('recall', 'Model Recall over Epochs', 'recall_mobilenetv2.png')

# Evaluate on validation and test sets
val_results = model.evaluate(generator_val, verbose=0)
test_results = model.evaluate(generator_test, verbose=0)
labels = ['loss', 'accuracy', 'precision', 'recall']

# Bar charts: validation vs test
for idx, label in enumerate(labels):
    plt.figure()
    plt.bar(['Validation', 'Test'], [val_results[idx], test_results[idx]])
    plt.title(f'{label.capitalize()} on Validation vs Test')
    plt.ylabel(label.capitalize())
    plt.tight_layout()
    fname = f'{label}_val_test_mobilenetv2.png'
    plt.savefig(os.path.join(plot_dir, fname))
    plt.show()

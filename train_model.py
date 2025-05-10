import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
import os

# ✅ Set random seed
tf.random.set_seed(42)

# ✅ Define dataset paths
train_path = r'C:\Users\ABHISHEK BUDHWAT\OneDrive\Desktop\NEW_NUTRISNAP\Dataset\Indian_Food_Dataset\train'
valid_path = r'C:\Users\ABHISHEK BUDHWAT\OneDrive\Desktop\NEW_NUTRISNAP\Dataset\Indian_Food_Dataset\valid'
# ✅ Check dataset existence
if not os.path.exists(train_path) or not os.path.exists(valid_path):
    raise FileNotFoundError(f"❌ Dataset folders not found! Check paths:\nTrain: {train_path}\nValid: {valid_path}")

# ✅ Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# ✅ Load datasets
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = valid_datagen.flow_from_directory(valid_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# ✅ MobileNetV2 with fine-tuning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze all layers initially

num_classes = len(train_generator.class_indices)
if num_classes == 0:
    raise ValueError("❌ No classes found! Check dataset structure.")
print(f"✅ Number of Classes: {num_classes}")

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('train_model.h5', monitor='val_loss', save_best_only=True)

# ✅ Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=5)

# ✅ Fine-tune last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=15, callbacks=[early_stopping, reduce_lr, checkpoint])

# ✅ Load best model before saving
if os.path.exists("train_model.h5"):
    model = tf.keras.models.load_model("train_model.h5")
    print("✅ Loaded Best Model for Final Evaluation!")

# ✅ Save final model
if not os.path.exists('model'):
    os.makedirs('model')
model.save('model/final_trained_model.keras')

print("🎉 Model saved successfully!")


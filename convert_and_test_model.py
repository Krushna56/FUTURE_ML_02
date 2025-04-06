import os
from tensorflow.keras.models import load_model

# 🔁 STEP 1: Path to your existing .h5 model
h5_model_path = r"C:\Users\krush\Desktop\Internship\FUTURE_ML_02\your_model.h5"  # ← Change this
keras_model_path = r"C:\Users\krush\Desktop\Internship\FUTURE_ML_02\Stock Prediction Model.h5"

# ✅ STEP 2: Load the old .h5 model
if not os.path.exists(h5_model_path):
    print("❌ .h5 model not found at:", h5_model_path)
    exit()

print("📦 Loading .h5 model...")
model = load_model(h5_model_path)

# ✅ STEP 3: Save as new `.keras` format (zip archive)
print("💾 Saving model as .keras format...")
model.save(keras_model_path, save_format="keras")

# ✅ STEP 4: Verify file is accessible
print("\n🔍 Verifying new .keras model file...")
print("✔ File exists:", os.path.exists(keras_model_path))
print("✔ Is file:", os.path.isfile(keras_model_path))
print("📐 File size:", os.path.getsize(keras_model_path), "bytes")

# ✅ STEP 5: Try loading it
print("🚀 Loading the new .keras model to verify...")
model_loaded = load_model(keras_model_path)
print("🎉 Model loaded successfully from .keras file!")

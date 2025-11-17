# ML Model Integration Guide

## 🎯 Where to Add Your ML Model

### Location in Code

File: `venv\app.py`
Line: ~42-54 (in the `classify_image` function)

### Current Mock Code

```python
def classify_image(image):
    """
    Simulate ML model classification
    Replace this with your actual model inference code
    """
    # TODO: Replace with actual model prediction
    # For now, randomly classify with weighted probability
    is_good = random.random() > 0.2  # 80% good parts
    confidence = random.uniform(0.85, 0.98) if is_good else random.uniform(0.75, 0.95)
    
    return {
        "status": "Good" if is_good else "Defective",
        "confidence": confidence
    }
```

## 📋 Step-by-Step Integration

### Step 1: Add Required Imports

At the top of `venv\app.py`, add your ML library imports:

```python
import streamlit as st
from PIL import Image
import os
import random
from datetime import datetime, timedelta
import io

# ADD YOUR IMPORTS HERE
import numpy as np
import cv2
import torch  # or tensorflow, keras, etc.
# import any other required libraries
```

### Step 2: Load Your Model (Cached)

Add this function after the imports:

```python
@st.cache_resource
def load_model():
    """
    Load the trained ML model once and cache it
    This prevents reloading on every classification
    """
    try:
        # Option 1: PyTorch
        model = torch.load('path/to/your/model.pth', map_location='cpu')
        model.eval()
        
        # Option 2: TensorFlow/Keras
        # from tensorflow import keras
        # model = keras.models.load_model('path/to/your/model.h5')
        
        # Option 3: Pickle
        # import pickle
        # with open('path/to/model.pkl', 'rb') as f:
        #     model = pickle.load(f)
        
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load model globally
ML_MODEL = load_model()
```

### Step 3: Create Preprocessing Function

```python
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input
    Adjust based on your model's requirements
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, target_size)
    
    # Normalize pixel values (adjust range based on your model)
    # Option 1: Scale to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Option 2: Standardize with mean/std
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # img_normalized = (img_resized / 255.0 - mean) / std
    
    # Add batch dimension if needed
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch
```

### Step 4: Replace classify_image Function

```python
def classify_image(image):
    """
    Classify cathode cup image using trained ML model
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: {
            'status': 'Good' or 'Defective',
            'confidence': float (0-1)
        }
    """
    if ML_MODEL is None:
        # Fallback to mock if model not loaded
        st.warning("Model not loaded, using mock classification")
        is_good = random.random() > 0.2
        confidence = random.uniform(0.85, 0.98)
        return {
            "status": "Good" if is_good else "Defective",
            "confidence": confidence
        }
    
    try:
        # Preprocess image
        img_tensor = preprocess_image(image)
        
        # Run inference
        # PyTorch example:
        with torch.no_grad():
            output = ML_MODEL(torch.FloatTensor(img_tensor))
            # Assuming binary classification with sigmoid output
            confidence = torch.sigmoid(output).item()
        
        # TensorFlow example:
        # output = ML_MODEL.predict(img_tensor, verbose=0)
        # confidence = float(output[0][0])
        
        # Determine classification
        # Adjust threshold as needed (0.5 is common)
        threshold = 0.5
        if confidence >= threshold:
            status = "Good"
        else:
            status = "Defective"
            confidence = 1 - confidence  # Invert for defective confidence
        
        return {
            "status": status,
            "confidence": float(confidence)
        }
        
    except Exception as e:
        st.error(f"Classification error: {e}")
        # Return fallback result
        return {
            "status": "Good",
            "confidence": 0.5
        }
```

## 🔧 Model-Specific Examples

### Example 1: YOLOv8 Object Detection

```python
from ultralytics import YOLO

@st.cache_resource
def load_model():
    model = YOLO('path/to/best.pt')
    return model

def classify_image(image):
    if ML_MODEL is None:
        return {"status": "Good", "confidence": 0.5}
    
    # Run inference
    results = ML_MODEL(image)
    
    # Parse results
    boxes = results[0].boxes
    if len(boxes) > 0:
        # Get highest confidence detection
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        max_idx = confidences.argmax()
        
        class_name = ML_MODEL.names[int(classes[max_idx])]
        confidence = float(confidences[max_idx])
        
        status = "Good" if class_name == "good" else "Defective"
    else:
        # No detections
        status = "Good"
        confidence = 0.5
    
    return {"status": status, "confidence": confidence}
```

### Example 2: ResNet Classifier

```python
import torch
import torchvision.transforms as transforms
from torchvision import models

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes
    model.load_state_dict(torch.load('path/to/resnet50_cathode.pth'))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def classify_image(image):
    if ML_MODEL is None:
        return {"status": "Good", "confidence": 0.5}
    
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = ML_MODEL(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    status = "Good" if predicted.item() == 0 else "Defective"
    
    return {
        "status": status,
        "confidence": float(confidence.item())
    }
```

### Example 3: TensorFlow/Keras CNN

```python
from tensorflow import keras
import numpy as np

@st.cache_resource
def load_model():
    model = keras.models.load_model('path/to/cathode_model.h5')
    return model

def preprocess_image(image):
    img_array = np.array(image.resize((224, 224)))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(image):
    if ML_MODEL is None:
        return {"status": "Good", "confidence": 0.5}
    
    img_tensor = preprocess_image(image)
    
    # Predict
    prediction = ML_MODEL.predict(img_tensor, verbose=0)
    
    # Assuming binary classification [good_prob, defective_prob]
    good_prob = float(prediction[0][0])
    defective_prob = float(prediction[0][1])
    
    if good_prob > defective_prob:
        status = "Good"
        confidence = good_prob
    else:
        status = "Defective"
        confidence = defective_prob
    
    return {
        "status": status,
        "confidence": confidence
    }
```

## ⚙️ Configuration

### Model Path

Store model path in a configuration variable at the top of the file:

```python
# Configuration
MODEL_PATH = "path/to/your/model.pth"
INPUT_SIZE = (224, 224)
CLASSIFICATION_THRESHOLD = 0.5
```

### Alternative: Environment Variables

```python
import os

MODEL_PATH = os.getenv('MODEL_PATH', 'default/path/model.pth')
```

## 🧪 Testing Your Integration

### Test Steps

1. **Load Test**: Verify model loads without errors
   ```python
   model = load_model()
   print(f"Model loaded: {model is not None}")
   ```

2. **Single Image Test**: Test with one image
   ```python
   from PIL import Image
   test_img = Image.open('test_cathode.jpg')
   result = classify_image(test_img)
   print(result)
   ```

3. **Batch Test**: Test with multiple images
4. **Error Handling**: Test with invalid images
5. **Performance**: Measure inference time

### Add Timing

```python
import time

def classify_image(image):
    start_time = time.time()
    
    # ... your classification code ...
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Store inference time in session state
    if 'inference_times' not in st.session_state:
        st.session_state.inference_times = []
    st.session_state.inference_times.append(inference_time)
    
    return result
```

## 🐛 Debugging Tips

### Add Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_image(image):
    logger.info(f"Classifying image: {image.size}")
    try:
        result = # ... classification ...
        logger.info(f"Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise
```

### Display Debug Info

```python
def classify_image(image):
    # ... classification code ...
    
    # Debug info (remove in production)
    st.write("Debug Info:")
    st.write(f"Image size: {image.size}")
    st.write(f"Model input shape: {img_tensor.shape}")
    st.write(f"Raw output: {output}")
    
    return result
```

## ✅ Checklist

Before deploying:

- [ ] Model loads successfully
- [ ] Preprocessing matches training pipeline
- [ ] Output format is correct (Good/Defective + confidence)
- [ ] Error handling works for edge cases
- [ ] Inference time is acceptable (< 1 second recommended)
- [ ] Results are accurate on test images
- [ ] Mock fallback works if model fails
- [ ] Logging is set up for debugging

## 📞 Need Help?

If you encounter issues:

1. Check model file path is correct
2. Verify input preprocessing matches training
3. Test model inference separately (outside Streamlit)
4. Check model output format
5. Review error messages in terminal
6. Test with known good/bad images

---

**Ready to integrate?** Follow the steps above and replace the mock `classify_image()` function with your real model!

# ML Model Classification System - Complete Guide

## ✅ Classification is NOW Done by Your Trained ML Model!

Yes, the classification is being performed by **your actual trained ML model** (the pickle file), not mock/random data.

---

## 🎯 How the System Works

### 1. Model Loading (Automatic on Startup)

```python
# Located at: venv/app.py, lines ~22-57
@st.cache_resource
def load_model():
    """Load the trained ML model from pickle file"""
    # Loads model.pkl from venv folder
    # Maps CUDA tensors to CPU automatically
    # Sets model to evaluation mode
    # Cached so it only loads once
```

**What happens:**
- On dashboard startup, looks for `model.pkl` in the `venv` folder
- If found: Loads and caches the model (✅ Real classification enabled)
- If not found: Shows warning and uses mock fallback

### 2. Image Classification Process

```
User uploads image
        ↓
classify_image() function called
        ↓
Model.predict(image) → ML inference
        ↓
Returns: {boxes, labels, confidences}
        ↓
Draw bounding boxes on image
        ↓
Determine overall status (Good/Defective)
        ↓
Display results + annotated image
```

### 3. What the Model Returns

Your trained model should return a dictionary with:

```python
output = {
    "boxes": [[x1, y1, x2, y2], ...],      # Bounding box coordinates
    "labels": ["crack", "good", ...],       # Defect type or "good"
    "confidences": [0.95, 0.87, ...]        # Confidence scores
}
```

---

## 📊 Classification Logic

### Status Determination

```python
# From classify_image() function

if not labels or all(label.lower() == "good" for label in labels):
    status = "Good"
    # No defects detected or all labeled as good
else:
    status = "Defective"
    # At least one defect detected
```

### Confidence Calculation

- **For Good parts**: Uses highest confidence from detections
- **For Defective parts**: Uses highest confidence from defect detections (excluding "good" labels)

---

## 🖼️ Bounding Box Display

### What Gets Drawn

1. **Rectangle**: Around detected regions
   - Green for "good" labels
   - Red for defect labels

2. **Label Text**: Shows defect type and confidence
   - Format: `"crack: 0.95"`
   - Background colored same as box

3. **Multiple Detections**: All boxes drawn if model detects multiple regions

### Drawing Function

```python
def draw_bounding_boxes(image_np, boxes, labels, confidences):
    """Draws boxes on numpy array image"""
    # Green (0, 255, 0) for good
    # Red (255, 0, 0) for defects
    # Returns annotated numpy array
```

---

## 🎨 UI Display Features

### 1. Upload Section (Expanded View)

**Left Column: Original Image**
- Shows uploaded image as-is

**Right Column: Annotated Result**
- Shows image with bounding boxes drawn
- Boxes around detected defects
- Labels showing defect types

### 2. Result Messages

**Good Classification:**
```
✅ Good - Confidence: 95.3%
```

**Defective Classification:**
```
❌ Defective - Confidence: 87.2%
⚠️ Defect Types Detected: crack, chip, dent
```

### 3. Feed Display

Each feed item shows:
- **Thumbnail**: Annotated image with boxes (if defects)
- **Timestamp**: When classified
- **Cup ID**: Unique identifier
- **Status Badge**: Green (Good) or Red (Defective)
- **Confidence**: Percentage
- **Defect Types**: List of detected defects (if any)

---

## 🔧 Model Integration Details

### Your Model's Expected API

```python
# Your model should have a predict() method:
output = model.predict(image_numpy_array)

# Expected output format:
{
    "boxes": [
        [x1, y1, x2, y2],  # First detection
        [x1, y1, x2, y2],  # Second detection
        ...
    ],
    "labels": [
        "crack",     # Label for first box
        "good",      # Label for second box
        ...
    ],
    "confidences": [
        0.95,        # Confidence for first detection
        0.87,        # Confidence for second detection
        ...
    ]
}
```

### If Your Model Has Different Output Format

If your model returns different keys or structure, modify the `classify_image()` function:

```python
# Example adjustments:

# If your model uses "scores" instead of "confidences":
confidences = output.get("scores", [])

# If your model uses "classes" instead of "labels":
labels = output.get("classes", [])

# If boxes are in different format [y1, x1, y2, x2]:
boxes = [[box[1], box[0], box[3], box[2]] for box in output["boxes"]]

# If your model returns class indices instead of names:
label_map = {0: "good", 1: "crack", 2: "chip", 3: "dent"}
labels = [label_map[idx] for idx in output["class_ids"]]
```

---

## 📍 Where Your Model File Should Be

### Correct Location

```
Project Root/
└── venv/
    ├── app.py           ← Your dashboard code
    └── model.pkl        ← PUT YOUR MODEL HERE
```

### To Add Your Model

1. Copy your trained `model.pkl` file
2. Paste it in the `venv` folder (same folder as app.py)
3. Restart the dashboard
4. You should see: "✅ ML Model loaded successfully!"

---

## ⚠️ Fallback Behavior

### If Model Not Found

```python
if ML_MODEL is None:
    st.warning("⚠️ Using mock classification - Model not loaded")
    # Uses random classification for demo purposes
    # Still shows boxes and labels (randomly generated)
```

**You'll see:**
- Warning message at top of dashboard
- Random Good/Defective results
- Mock bounding boxes (if defective)

### When Model Loads Successfully

```
✅ ML Model loaded successfully!
```
- All classifications use your trained model
- Real predictions with actual confidence scores
- Accurate bounding boxes and defect types

---

## 🧪 Testing Your Model Integration

### Verification Checklist

1. **Model File Present**
   ```
   Check: venv/model.pkl exists
   ```

2. **Model Loads on Startup**
   ```
   Look for: "✅ ML Model loaded successfully!"
   OR: "⚠️ Model file not found"
   ```

3. **Upload Test Image**
   ```
   - Upload known good cathode cup
   - Should classify as "Good"
   ```

4. **Upload Defective Image**
   ```
   - Upload image with known defects
   - Should classify as "Defective"
   - Should show bounding boxes
   - Should list defect types
   ```

5. **Check Bounding Boxes**
   ```
   - Boxes should appear on defects
   - Red color for defects
   - Labels should match defect types
   ```

### Test Commands

```python
# Test model loading independently:
import pickle
import torch

with open('venv/model.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"Model type: {type(model)}")
print(f"Model: {model}")

# Test prediction:
import numpy as np
from PIL import Image

img = Image.open('test_image.jpg')
img_np = np.array(img)
result = model.predict(img_np)
print(result)
```

---

## 📊 Expected Results

### Good Part Example

```
Status: Good
Confidence: 95.3%
Boxes: [] or [[x1,y1,x2,y2]] with label="good"
Defect Types: None
```

### Defective Part Example

```
Status: Defective
Confidence: 87.5%
Boxes: [[120, 150, 300, 400], [450, 200, 600, 350]]
Labels: ["crack", "chip"]
Defect Types: crack, chip
```

---

## 🔍 Debugging Model Issues

### Issue: Model Not Loading

**Check:**
- File path correct? Should be `venv/model.pkl`
- File permissions OK?
- Model saved correctly? (Try loading manually in Python)

**Fix:**
```python
# Test loading manually:
import pickle
with open('venv/model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Issue: Wrong Predictions

**Check:**
- Image preprocessing matches training?
- Model in eval mode? `model.eval()`
- Input format correct? (numpy array vs tensor)

**Debug:**
```python
# Add debug prints in classify_image():
print(f"Image shape: {image_np.shape}")
print(f"Image dtype: {image_np.dtype}")
print(f"Model output: {output}")
```

### Issue: No Bounding Boxes

**Check:**
- Model returns boxes in correct format?
- Boxes have valid coordinates?
- Labels correspond to boxes?

**Debug:**
```python
print(f"Boxes: {boxes}")
print(f"Labels: {labels}")
print(f"Box count: {len(boxes)}")
```

### Issue: Wrong Colors/Labels

**Check:**
- Label names match expected ("good", "crack", etc.)?
- Case sensitivity? (code uses `.lower()`)

**Fix:**
```python
# Normalize labels to lowercase:
labels = [label.lower() for label in output["labels"]]
```

---

## 🎯 Customization Options

### Adjust Classification Threshold

```python
# If you want to require higher confidence:
MIN_CONFIDENCE = 0.7

# Filter low-confidence detections:
filtered_boxes = []
filtered_labels = []
for box, label, conf in zip(boxes, labels, confidences):
    if conf >= MIN_CONFIDENCE:
        filtered_boxes.append(box)
        filtered_labels.append(label)
```

### Change Defect Type Display

```python
# Map model labels to readable names:
DEFECT_NAMES = {
    "crack": "Surface Crack",
    "chip": "Edge Chipping",
    "dent": "Surface Dent",
    "scratch": "Surface Scratch"
}

display_names = [DEFECT_NAMES.get(label, label) for label in defect_types]
```

### Adjust Bounding Box Appearance

```python
# In draw_bounding_boxes():

# Thicker lines:
draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

# Different colors:
color = (0, 255, 0)     # Green
color = (255, 0, 0)     # Red
color = (255, 165, 0)   # Orange

# Larger font:
font = ImageFont.truetype("arial.ttf", 30)
```

---

## ✅ Summary

**YES - Classification is done by your trained ML model!**

✅ Loads `model.pkl` from venv folder
✅ Uses `model.predict()` for inference
✅ Draws real bounding boxes on detected defects
✅ Shows actual defect types from model output
✅ Displays confidence scores from model
✅ Falls back to mock only if model not found

**To enable real classification:**
1. Place `model.pkl` in the `venv` folder
2. Restart dashboard
3. Look for "✅ ML Model loaded successfully!"
4. Upload and classify!

---

**Questions about model integration?** Check the terminal output when uploading images to see actual model results and any errors.

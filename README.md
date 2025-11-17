# Quality Control Dashboard

A real-time quality control dashboard for cathode cup classification and manufacturing workflow monitoring, built with Streamlit.

## Features

- **Real-time Image Upload**: Capture and upload images directly from your camera system
- **AI Classification**: Automatic quality classification (Good/Defective) with confidence scores
- **Live Metrics**: Monitor parts processed, good parts, defective parts, and average inference time
- **Classification Feed**: View real-time classification history with uploaded images
- **Quick Actions**: Start/pause production, export reports, and clear history
- **Alerts System**: Track recent system events and notifications
- **Quality Trends**: Visualize quality metrics over time (placeholder for future implementation)

## How It Works

1. **Capture Image**: Someone captures an image of a cathode cup using the camera system
2. **Upload**: The image is uploaded through the dashboard interface
3. **Classification**: AI model analyzes the image and classifies it as Good or Defective
4. **Results**: Decision is displayed with confidence score and added to the real-time feed
5. **Metrics**: Dashboard metrics update automatically with each classification

## Project Structure

```
Cathode Cup Website/
├── venv/
│   └── app.py              # Main Streamlit application
├── static/
│   └── styles.css          # Custom CSS styling
├── assets/                 # Placeholder images (optional)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation & Setup

### 1. Create Virtual Environment (if not already created)

```cmd
python -m venv venv
```

### 2. Activate Virtual Environment

```cmd
venv\Scripts\activate
```

### 3. Install Dependencies

```cmd
pip install -r requirements.txt
```

## Running the Dashboard

From the project root directory, run:

```cmd
streamlit run venv\app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

## Using the Dashboard

### Uploading and Classifying Images

1. **Open the Upload Section**: Click on "📸 Upload Cathode Cup Image for Classification" expander
2. **Choose Image**: Click "Browse files" and select a cathode cup image from your camera system
3. **Preview**: The uploaded image will appear in the preview pane
4. **Classify**: Click "🔍 Classify Image" button to run the classification
5. **View Results**: 
   - Classification result appears (Good ✅ or Defective ❌)
   - Confidence score is displayed
   - Image is automatically added to the real-time feed
   - Dashboard metrics update automatically

### Real-time Feed

- Shows up to 10 most recent classifications
- Each entry displays:
  - Timestamp of classification
  - Unique Cathode Cup ID
  - Classification status (Good/Defective) with color-coded badge
  - Confidence percentage
  - Thumbnail of the uploaded image

### Quick Actions

- **Start Production**: Begin production monitoring
- **Pause System**: Temporarily halt monitoring
- **Export Report**: Generate classification report (placeholder)
- **Clear History**: Reset all metrics and clear classification feed

## Integrating Your ML Model

The current app uses a mock classification function. To integrate your actual ML model:

1. Open `venv\app.py`
2. Find the `classify_image(image)` function (around line 42)
3. Replace the mock code with your model inference:

```python
def classify_image(image):
    """
    Classify cathode cup image using ML model
    """
    # Example integration:
    # 1. Preprocess image
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    
    # 2. Run inference
    prediction = your_model.predict(img_resized)
    
    # 3. Get results
    status = "Good" if prediction > 0.5 else "Defective"
    confidence = prediction if status == "Good" else 1 - prediction
    
    return {
        "status": status,
        "confidence": float(confidence)
    }
```

## Configuration

### Session State
The dashboard uses Streamlit session state to maintain:
- Classification feed history (last 10 items)
- Total parts processed
- Good/defective counts
- Uploaded image data

Data persists during the browser session and resets when the page is refreshed.

### Alerts
Update the `get_recent_alerts()` function to customize:
- Alert types (info, warning, success)
- Alert messages
- Timestamps

## Customization

### Styling
Edit `static\styles.css` to customize:
- Colors and themes
- Card layouts
- Typography
- Button styles

### Layout
Modify `venv\app.py` to adjust:
- Column widths
- Component placement
- Responsive behavior

## Troubleshooting

### Dashboard doesn't load
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)
- Verify virtual environment is activated

### Upload doesn't work
- Check file format (only jpg, jpeg, png supported)
- Verify file size is reasonable (< 10MB recommended)
- Check browser console for errors

### Metrics not updating
- Ensure you clicked "Classify Image" button after upload
- Check that session state is working (refresh page to reset)
- Verify no errors in terminal console

### Styling issues
- Clear browser cache
- Verify `static\styles.css` exists
- Check console for CSS errors

## Future Enhancements

- [ ] Connect to actual ML model for real inference
- [ ] Add batch upload for multiple images
- [ ] Implement actual quality trend charts with historical data
- [ ] Add database integration for persistent storage
- [ ] Create authentication system
- [ ] Add CSV/PDF export functionality for reports
- [ ] Implement camera API integration for direct capture
- [ ] Add email/SMS notifications for alerts
- [ ] Create REST API for external system integration

## Technology Stack

- **Frontend**: Streamlit
- **Styling**: Custom CSS
- **Image Processing**: Pillow (PIL)
- **Python**: 3.8+

## License

© 2025 QualityControl AI. All rights reserved.

## Support

For issues or questions, please contact your development team.

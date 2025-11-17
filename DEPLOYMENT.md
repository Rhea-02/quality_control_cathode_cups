# Streamlit Quality Control Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

A real-time quality control dashboard for cathode cup classification and manufacturing workflow monitoring.

## 🚀 Live Demo

Deploy this app to Streamlit Community Cloud:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select this repository
6. Main file: `app_working.py`
7. Click "Deploy"

## 📋 Features

- **Real-time Image Classification**: Upload cathode cup images for instant AI-powered quality assessment
- **Interactive Dashboard**: Monitor production metrics, defect rates, and quality trends
- **Statistical Analysis**: SPC charts, Cp/Cpk indices, trend analysis
- **Report Generation**: Generate and export quality reports (CSV, Excel, PDF)
- **Multi-shift Tracking**: Track quality metrics across different production shifts
- **Defect Pattern Analysis**: Identify common failure modes and root causes

## 🛠️ Local Development

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Rhea-02/quality_control_cathode_cups.git
cd quality_control_cathode_cups

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app_working.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
├── app_working.py          # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   ├── config.toml        # Streamlit configuration
│   └── runtime.txt        # Python version specification
├── static/
│   └── styles.css         # Custom CSS styling
├── model.pkl              # ML model (optional)
└── README.md              # This file
```

## 🔧 Configuration

The app uses a mock classification system by default. To use your own ML model:

1. Place your trained model file as `model.pkl` in the root directory
2. The model should accept PIL Image objects and return classification results
3. Format: `{"status": "Good/Defective", "confidence": 0.0-1.0}`

## 📊 Tabs Overview

### 1. Dashboard 📊
- Upload and classify images
- View real-time metrics
- Monitor recent classifications

### 2. Analyzers 🔬
- **Trend Analysis**: Defect rate trends, shift comparisons
- **Defect Patterns**: Distribution by type, failure modes
- **SPC Charts**: Statistical process control with control limits

### 3. Settings ⚙️
- Model management
- Alert configuration
- User management
- System preferences

### 4. Reports 📈
- Generate custom reports
- Export data (CSV, Excel, PDF)
- View report history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Rhea**
- GitHub: [@Rhea-02](https://github.com/Rhea-02)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Charts powered by [Plotly](https://plotly.com/)
- ML model integration using PyTorch

---

**Note**: The model file (`model.pkl`) is 158MB. If you encounter GitHub size limits, consider using [Git LFS](https://git-lfs.github.com/) or hosting the model separately.

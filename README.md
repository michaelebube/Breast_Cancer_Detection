# 🩺 Breast Cancer Detection System

An AI-powered breast cancer risk assessment system using machine learning. This project uses the Wisconsin Breast Cancer Dataset to predict whether a tumor is **Benign** or **Malignant**.

## 📁 Project Structure

```
Breast Cancer Prediction System/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── model/               # Model files (from Colab training)
│       ├── breast_cancer_model.joblib
│       ├── scaler.joblib
│       └── dataset_stats.json
├── frontend/
│   ├── index.html           # Main HTML page
│   ├── styles.css           # Styling
│   └── app.js               # Frontend JavaScript
├── notebooks/
│   └── Breast_Cancer_Model_Training.ipynb  # Colab training notebook
├── model/                   # Model files placeholder
├── Dockerfile               # Docker configuration
├── render.yaml              # Render deployment config
├── railway.json             # Railway deployment config
├── fly.toml                 # Fly.io deployment config
└── README.md
```

## 🚀 Getting Started

### Step 1: Train the Model on Google Colab

1. Open the notebook `notebooks/Breast_Cancer_Model_Training.ipynb` in Google Colab
2. Run all cells to train the model
3. Download the generated files:
   - `breast_cancer_model.joblib`
   - `scaler.joblib`
   - `dataset_stats.json`
4. Place these files in `backend/model/` directory

### Step 2: Set Up the Backend

```bash
# Navigate to project directory
cd "Breast Cancer Prediction System"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the API
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Step 3: Open the Frontend

Simply open `frontend/index.html` in a web browser, or serve it with a local server:

```bash
# Using Python
cd frontend
python -m http.server 3000

# Or using Node.js
npx serve frontend
```

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

Returns API status and model loading state.

### Predict (User-Friendly Input)

```http
POST /predict
Content-Type: application/json

{
    "lump_size": 2.5,
    "lump_shape": "irregular",
    "lump_texture": "rough",
    "lump_hardness": 70,
    "growth_rate": "fast",
    "pain_present": false,
    "skin_changes": true,
    "nipple_discharge": false,
    "family_history": true,
    "patient_age": 55
}
```

### Predict (Raw Features)

```http
POST /predict/raw
Content-Type: application/json

{
    "features": [17.99, 10.38, 122.8, ...]  // 30 features
}
```

### Feature Information

```http
GET /features
```

Returns feature names and mapping documentation.

## 📊 Input Fields

| Input            | Type               | Description                           |
| ---------------- | ------------------ | ------------------------------------- |
| Lump Size        | Slider (0.1-10 cm) | Approximate diameter of detected lump |
| Lump Shape       | Select             | Regular / Irregular                   |
| Lump Texture     | Select             | Smooth / Rough                        |
| Lump Hardness    | Slider (0-100)     | Softness to hardness scale            |
| Growth Rate      | Select             | Slow / Fast                           |
| Pain Present     | Toggle             | Yes / No                              |
| Skin Changes     | Toggle             | Yes / No                              |
| Nipple Discharge | Toggle             | Yes / No                              |
| Family History   | Toggle             | Yes / No                              |
| Patient Age      | Number             | Age in years                          |

## 🔄 Feature Mapping

The backend converts user-friendly inputs to the 30 features required by the model:

- **Lump Size** → `radius_mean`, `perimeter_mean`, `area_mean`
- **Lump Texture** → `texture_mean`
- **Lump Shape** → `concavity_mean`, `symmetry_mean`
- **Lump Hardness** → `compactness_mean`
- **Growth Rate** → `fractal_dimension_mean`, `smoothness_mean`
- **Binary inputs** → Risk factor adjustments

## 🚢 Deployment

### Render

1. Push code to GitHub
2. Connect repository to Render
3. Render will auto-detect `render.yaml` configuration
4. Deploy!

### Railway

1. Push code to GitHub
2. Create new project on Railway
3. Connect repository
4. Railway will use `railway.json` configuration

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Deploy
fly launch
fly deploy
```

### Docker

```bash
# Build image
docker build -t breast-cancer-prediction .

# Run container
docker run -p 8000:8000 breast-cancer-prediction
```

## 📋 API Response Example

```json
{
  "prediction": "Benign",
  "confidence": 0.9234,
  "probability_benign": 0.9234,
  "probability_malignant": 0.0766,
  "risk_level": "Low"
}
```

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

- This is NOT a medical diagnostic tool
- Results should NOT be used for medical decisions
- Always consult qualified healthcare professionals for medical advice
- The model is trained on historical data and may not reflect current medical standards

## 🧪 Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, joblib
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML Model**: Gradient Boosting Classifier
- **Dataset**: Wisconsin Breast Cancer Dataset (sklearn)

## 📈 Model Performance

The trained model typically achieves:

- **Accuracy**: ~96-98%
- **F1 Score**: ~96-98%
- **AUC-ROC**: ~99%

_Actual performance may vary based on hyperparameter tuning._

## 📝 License

This project is for educational purposes as part of CSC415 AI Practicals coursework.

## 👤 Author

Michael Ebube - 400 Level AS - 2025/26

---

Made with ❤️ for CSC415 AI Practicals

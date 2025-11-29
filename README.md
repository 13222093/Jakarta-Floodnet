# 🌊 Jakarta FloodNet - AI-Powered Flood Early Warning System

Jakarta FloodNet is a comprehensive early warning system designed to predict and detect flooding in Jakarta. It integrates real-time sensor data, AI-based prediction models, and computer vision for flood detection into a unified dashboard for decision-makers.

## 🚀 System Overview

The system consists of three main components:

1.  **API Gateway**: A robust FastAPI backend that orchestrates data flow, handles model inference requests, and manages system logic.
2.  **AI Models**:
    -   **LSTM (Long Short-Term Memory)**: Predicts flooding probability 24 hours in advance based on rainfall and water level data.
    -   **YOLOv8**: Detects flooded areas and estimates flood extent from CCTV or satellite imagery.
3.  **Dashboard**: An interactive Streamlit-based user interface for visualizing real-time data, predictions, and flood detection results.

## ✨ Key Features

-   **Real-time Monitoring**: Tracks rainfall (mm) and water levels (cm) from various monitoring points.
-   **Flood Prediction**: Provides a 24-hour forecast with probability scores and risk levels (LOW, MEDIUM, HIGH).
-   **Visual Flood Detection**: Automatically identifies flooded regions in images and calculates the affected area.
-   **Actionable Insights**: Generates automated recommendations (e.g., "Initiate evacuation") based on risk levels.

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI, Uvicorn
-   **Frontend**: Streamlit, Plotly
-   **AI/ML**: TensorFlow (LSTM), Ultralytics YOLOv8, OpenCV
-   **Data Processing**: Pandas, NumPy

## 📦 Installation

### Prerequisites
-   Python 3.9+
-   pip

### Setup
1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd Jakarta-Floodnet
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚦 Usage

### 1. Run the API Gateway
Start the backend server to handle requests and model inference:
```bash
uvicorn services.api_gateway.src.main:app --reload
```
-   **API URL**: `http://localhost:8000`
-   **Swagger Docs**: `http://localhost:8000/docs`

### 2. Run the Dashboard
Launch the user interface to visualize data and interact with the system:
```bash
streamlit run services/dashboard/src/app.py
```
-   **Dashboard URL**: `http://localhost:8501`

## 📚 API Endpoints

### Prediction & Detection
-   `POST /predict`: Predict flooding probability based on sensor data.
-   `POST /detect`: Detect flooded areas from images.

### System
# 🌊 Jakarta FloodNet - AI-Powered Flood Early Warning System

Jakarta FloodNet is a comprehensive early warning system designed to predict and detect flooding in Jakarta. It integrates real-time sensor data, AI-based prediction models, and computer vision for flood detection into a unified dashboard for decision-makers.

## 🚀 System Overview

The system consists of three main components:

1.  **API Gateway**: A robust FastAPI backend that orchestrates data flow, handles model inference requests, and manages system logic.
2.  **AI Models**:
    -   **LSTM (Long Short-Term Memory)**: Predicts flooding probability 24 hours in advance based on rainfall and water level data.
    -   **YOLOv8**: Detects flooded areas and estimates flood extent from CCTV or satellite imagery.
3.  **Dashboard**: An interactive Streamlit-based user interface for visualizing real-time data, predictions, and flood detection results.

## ✨ Key Features

-   **Real-time Monitoring**: Tracks rainfall (mm) and water levels (cm) from various monitoring points.
-   **Flood Prediction**: Provides a 24-hour forecast with probability scores and risk levels (LOW, MEDIUM, HIGH).
-   **Visual Flood Detection**: Automatically identifies flooded regions in images and calculates the affected area.
-   **Actionable Insights**: Generates automated recommendations (e.g., "Initiate evacuation") based on risk levels.

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI, Uvicorn
-   **Frontend**: Streamlit, Plotly
-   **AI/ML**: TensorFlow (LSTM), Ultralytics YOLOv8, OpenCV
-   **Data Processing**: Pandas, NumPy

## 📦 Installation

### Prerequisites
-   Python 3.9+
-   pip

### Setup
1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd Jakarta-Floodnet
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚦 Usage

### 1. Run the API Gateway
Start the backend server to handle requests and model inference:
```bash
uvicorn services.api_gateway.src.main:app --reload
```
-   **API URL**: `http://localhost:8000`
-   **Swagger Docs**: `http://localhost:8000/docs`

### 2. Run the Dashboard
Launch the user interface to visualize data and interact with the system:
```bash
streamlit run services/dashboard/src/app.py
```
-   **Dashboard URL**: `http://localhost:8501`

## 📚 API Endpoints

### Prediction & Detection
-   `POST /predict`: Predict flooding probability based on sensor data.
-   `POST /detect`: Detect flooded areas from images.

### System
-   `GET /health`: Check system status.
-   `GET /metrics`: View model performance metrics.

## 👥 Team
| Role | Person | Responsibility |
|------|--------|-----------------|
| Backend & ML | Ari Aziz | API, LSTM, YOLOv8, Integration |
| Frontend & Ops | Naufarrel | Dashboard, Deployment, CI/CD |
| Strategy | Dejet | Pitch, BPBD relations |

---
**Last Updated:** Nov 29, 2025
# PPE_GUARD_V1

PPE Detection System
This project is a real-time Personal Protective Equipment (PPE) detection system using deep learning models (CustomCNN, ResNet18, EfficientNet) with a web interface built on FastAPI. It detects PPE items (e.g., Hardhat, Safety Vest) in video streams or uploaded videos, logs violations, and provides admin statistics.
Prerequisites

Python 3.8+
PostgreSQL (for database)
CUDA-enabled GPU (optional, for faster inference)
Git
Model checkpoints (e.g., ResNet18_best.pt)

Setup

Clone the Repository:
git clone https://github.com/your-username/ppe-detection.git
cd ppe-detection


Set Up Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

If using a CUDA GPU, install PyTorch with CUDA support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118


Set Up Database:

Install PostgreSQL and create a database named ppe_detection.
Update DB_PARAMS in original_code.py with your PostgreSQL credentials:DB_PARAMS = {
    "dbname": "ppe_detection",
    "user": "your-username",
    "password": "your-password",
    "host": "localhost",
    "port": "5432"
}


The database tables are created automatically on startup.


Download Model Checkpoints:

Place model files (CustomCNN_best.pt, ResNet18_best.pt, EfficientNet_best.pt) in the models/ directory.
[Download link or instructions for obtaining models, e.g., Google Drive]


Place ApexCharts:

Download apexcharts.min.js and place it in the static/ directory to avoid CDN issues.



Running the Application

Start the FastAPI server:uvicorn server:app --host 0.0.0.0 --port 8000 --reload


Open http://localhost:8000/static/index.html in a browser.
Log in, select a model and target class, and use the camera or upload a video.

Project Structure

original_code.py: Core logic for model loading, database setup, and PPE detection.
server.py: FastAPI server handling API endpoints and WebSocket for real-time updates.
static/index.html: Web interface with ApexCharts for visualizations.
models/: Directory for model checkpoints (not included in Git).
requirements.txt: Python dependencies.
.gitignore: Excludes large files and sensitive data.

Notes

Model checkpoints are excluded from the repository due to size. Ensure they are placed in models/.
The system expects a 4-channel input (RGB + edge) for models. Checkpoints must match this configuration.
Database credentials should be secured and not committed to Git.


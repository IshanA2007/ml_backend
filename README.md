# ML Backend

A Flask-based REST API for machine learning model serving.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### GET /
Welcome message and status check

### GET /health
Health check endpoint

### POST /api/predict
Prediction endpoint (expects JSON payload)

Example request:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": "your_input_data"}'
```

## Project Structure

```
ml_backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Development

To add your ML model:
1. Load your model in `app.py`
2. Update the `/api/predict` endpoint with your prediction logic
3. Add any additional endpoints as needed

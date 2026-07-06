# ml_backend

This is the backend for a grid question scoring project I worked on with ML@UVA and the UVA Math Team. It is a Flask API that takes a photo of a worksheet or grid and turns it into structured data the rest of the app can score. It pairs with the Flutter client in [ml_frontend](https://github.com/IshanA2007/ml_frontend).

## What it does

The API exposes two scanning endpoints. Each one takes an uploaded image, runs it through a computer vision and OCR pipeline, and returns JSON.

- `POST /api/sudoku/scan` reads a Sudoku-style number grid from an image and returns the digits it found, cell by cell. The pipeline preprocesses the image, finds and straightens the grid, isolates each cell, and recognizes the digit inside it.
- `POST /api/polynomial/scan` finds the answer boxes on a polynomial worksheet and reads what a student wrote in them.

There are also `GET /` and `GET /health` endpoints for status checks.

Uploads are validated by file type and cleaned up after each request.

## Project layout

- `app.py` sets up the Flask app, the routes, upload handling, and error responses
- `sudoku_scanner_advanced.py` holds the grid detection and digit recognition
- `detect_polynomial.py` holds the worksheet box detection and reading
- `requirements.txt` lists the dependencies

## Running it locally

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the server with `python app.py`.

The server runs on `http://localhost:5000`. Point the ml_frontend app at that address, or send an image directly, for example:

```bash
curl -F "image=@grid.jpg" http://localhost:5000/api/sudoku/scan
```

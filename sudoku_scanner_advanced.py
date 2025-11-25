import cv2
import numpy as np
import imutils
import argparse
import scipy
from tensorflow.keras.models import load_model

def show(img, name='img', scale=0.6):
    """Helper function to show an image scaled down for debugging."""
    h, w = img.shape[:2]
    cv2.imshow(name, cv2.resize(img, (int(w*scale), int(h*scale))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_contour(img, contour, name='contour'):
    """Draws and displays a given contour on the image."""
    display = img.copy()
    cv2.drawContours(display, [contour.astype(int)], -1, (0, 255, 0), 3)
    #show(display, name)

def preprocess_image(path, debug=False):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    if debug:
        show(thresh, 'thresh')
    
    return img, gray, thresh

def find_largest_square_contour(thresh):
    """Finds the largest 4-point contour that likely represents the Sudoku grid."""
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def order_points(pts):
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      
    rect[2] = pts[np.argmax(s)]     
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  
    rect[3] = pts[np.argmax(diff)]   
    return rect

def warp_board(img, pts, size=450):
    rect = order_points(pts)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, M

def split_cells(warped_gray, cell_size=50):
    size = warped_gray.shape[0]
    assert size % 9 == 0, "Warp target must be divisible by 9. Choose size like 450."
    cell_w = size // 9
    cells = []
    for r in range(9):
        row = []
        for c in range(9):
            x = c * cell_w
            y = r * cell_w
            cell = warped_gray[y:y+cell_w, x:x+cell_w]
            row.append(cell)
        cells.append(row)
    return cells

def extract_digit(cell, debug=False):
    h, w = cell.shape[:2]
    margin = int(min(h,w) * 0.12)
    roi = cell[margin:h-margin, margin:w-margin]
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if not contours:
        return None 
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.01 * (roi.shape[0] * roi.shape[1]):
        return None
    x,y,w_box,h_box = cv2.boundingRect(c)
    digit = th[y:y+h_box, x:x+w_box]
    size_out = 28
    h_d, w_d = digit.shape
    scale = size_out / max(h_d, w_d)
    new_w = int(w_d * scale)
    new_h = int(h_d * scale)
    digit_resized = cv2.resize(digit, (new_w, new_h))
    pad_top = (size_out - new_h) // 2
    pad_bottom = size_out - new_h - pad_top
    pad_left = (size_out - new_w) // 2
    pad_right = size_out - new_w - pad_left
    digit_padded = cv2.copyMakeBorder(digit_resized, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)
    if debug:
        show(digit_padded, 'digit')
    return digit_padded


def classify_digit(digit_img, model, use_tta=True, n_augments=3):
    #uses tta to improve classification accuracy
    digit_img = digit_img.astype("float32") / 255.0
    digit_img = np.expand_dims(digit_img, axis=-1)
    digit_img = np.expand_dims(digit_img, axis=0)
    
    if use_tta:
        predictions = []
        predictions.append(model.predict(digit_img, verbose=0))
        
        # do small rotations
        for angle in [-5, 5]:
            rotated = scipy.ndimage.rotate(digit_img[0], angle, reshape=False)
            rotated = np.expand_dims(rotated, axis=0)
            predictions.append(model.predict(rotated, verbose=0))
        
        # average predictions for final classificaion
        pred = np.mean(predictions, axis=0)
    else:
        pred = model.predict(digit_img, verbose=0)
    
    predicted_digit = np.argmax(pred[0])
    confidence = np.max(pred[0])
    
    if confidence < 0.6:
        return None
    
    return int(predicted_digit)

cnn_model = load_model("sudoku_digit_model_attention.h5")


def image_to_board(filepath: str):
    img, gray, thresh = preprocess_image(filepath, debug=False)
    square_contour = find_largest_square_contour(thresh)
    if square_contour is None:
        raise ValueError("No square contour found in the image.")

    show_contour(img, square_contour, 'Largest Square Contour')
    warped_color, M = warp_board(img, square_contour, size=450)
    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    #show(warped_color, 'Warped Sudoku Board')

    cells = split_cells(warped_gray, cell_size=450//9)
    grid = [[0]*9 for _ in range(9)]

    for r in range(9):
        for c in range(9):
            cell = cells[r][c]
            digit_img = extract_digit(cell, debug=False)
            if digit_img is None:
                grid[r][c] = "-"
                continue
            val = classify_digit(digit_img, cnn_model)
            print(f"Cell {r},{c} classified as: {val}")
            grid[r][c] = val if val is not None else "-"

    print("\nExtracted Sudoku Grid:")
    for row in grid:
        print(row)

    return grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudoku board image parser")
    parser.add_argument("--image", required=True, help="Path to Sudoku image file")
    args = parser.parse_args()

    image_to_board(args.image)
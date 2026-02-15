import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL = load_model("mnist_digit_model.h5") # CHANGE THIS TO ISHAN'S MODEL

def preprocess(img):
    """Convert to grayscale, denoise, threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # adaptive threshold handles variable lighting
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )
    return th, gray

def find_boxes(binary_img, img_shape):
    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w
    
    cnts, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for i, c in enumerate(cnts):
        # We still look for top-level boxes
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        area_ratio = area / img_area
        aspect = w / float(h)
        
        # Initial loose filtering
        if 0.005 < area_ratio < 0.20 and 0.6 < aspect < 1.6:
            candidates.append([x, y, w, h])

    if not candidates:
        return []

    # --- Deduplication Logic (NMS) ---
    # Sort by area descending so we process larger boxes first
    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    final_boxes = []

    for cand in candidates:
        cx, cy, cw, ch = cand
        is_duplicate = False
        
        for f_box in final_boxes:
            fx, fy, fw, fh = f_box
            
            # Calculate Intersection
            ix = max(cx, fx)
            iy = max(cy, fy)
            iw = min(cx + cw, fx + fw) - ix
            ih = min(cy + ch, fy + fh) - iy
            
            if iw > 0 and ih > 0:
                intersection_area = iw * ih
                # If they overlap by more than 50% of the smaller box's area, it's a double count
                if intersection_area / (cw * ch) > 0.5:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            final_boxes.append(cand)

    return final_boxes

def sort_boxes(boxes):
    """Sort boxes strictly left-to-right."""
    return sorted(boxes, key=lambda b: b[0])


def extract_rois(boxes, gray_img):
    """Extract regions of interest with border cropping (15%) and preprocessing."""
    rois = []

    for (x, y, w, h) in boxes:
        # Original bounds (clipped to image)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(gray_img.shape[1], x + w)
        y2 = min(gray_img.shape[0], y + h)

        # Compute 15% margins
        dx = int(0.15 * (x2 - x1))
        dy = int(0.15 * (y2 - y1))

        # Shrink box on all sides
        cx1 = min(x2, x1 + dx)
        cy1 = min(y2, y1 + dy)
        cx2 = max(cx1 + 1, x2 - dx)
        cy2 = max(cy1 + 1, y2 - dy)

        roi = gray_img[cy1:cy2, cx1:cx2]

        # Safety check (skip empty crops)
        if roi.size == 0:
            continue

        # Clean ROI for OCR
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)

        rois.append(roi)

    return rois


def draw_detected_boxes(img, boxes):
    """Debug visualization with labels."""
    out = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 3)
        cv2.putText(out, str(i), (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return out


# ========================
# MAIN PIPELINE
# ========================

def detect_worksheet_boxes(path_to_img):
    """Main detection pipeline."""
    img = cv2.imread(path_to_img)
    if img is None:
        raise ValueError("Could not load image.")
    
    binary, gray = preprocess(img)
    
    # Detect contours that look like boxes
    boxes = find_boxes(binary, img.shape)
    
    # Sort them into reading order
    boxes = sort_boxes(boxes)
    
    # Extract each box region for OCR
    rois = extract_rois(boxes, gray)
    
    # Visualize detected boxes
    debug_img = draw_detected_boxes(img, boxes)
    
    return boxes, rois, debug_img # use rois for OCR

def build_polynomial(detections):
    """
    Converts detections into a 2D array:
    [
      [exp1, exp2, exp3, ...],
      [coef1, coef2, coef3, ...]
    ]
    """

    if not detections:
        return []

    exponents = []
    coefficients = []
    for box in detections:
        if box["role"] == "exponent":
            exponents.append(box["digit"] if box["digit"] != "-" else 0)
        elif box["role"] == "coefficient":
            coefficients.append(box["digit"] if box["digit"] != "-" else 0)
    
    return [
        exponents,
        coefficients
    ]

def preprocess_roi_for_mnist(roi):
    # 1. Grayscale
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    roi = cv2.fastNlMeansDenoising(roi, None, h=10)

    # 3. Resize to MNIST size
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    # 4. Threshold
    try:
        _, roi = cv2.threshold(
            roi, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    except cv2.error:
        _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)

    # 5. Normalize to [0,1]
    roi = roi.astype("float32") / 255.0

    # 6. Add channel + batch dimensions
    roi = roi.reshape(1, 28, 28, 1)

    return roi

def extract_polynomial_boxes(path_to_img):
    """
    Detect worksheet boxes, classify digits with CNN,
    and build a structured polynomial representation.
    """
    # 1. Detect boxes + ROIs
    boxes, rois, debug_img = detect_worksheet_boxes(path_to_img)

    if len(rois) == 0:
        return {
            "num_boxes": 0,
            "detections": [],
            "polynomial": []
        }

    # 2. Load CNN model
    model = MODEL

    detections = []

    # 3. Run inference on each ROI
    for i, roi in enumerate(rois):
        x, y, w, h = boxes[i]
        role = "coefficient" if i % 2 == 0 else "exponent"

        if is_box_empty(roi):
            detections.append({
                "index": i,
                "digit": "-",
                "confidence": 0.0,
                "role": role,
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                }
            })
            continue

        roi_input = preprocess_roi_for_mnist(roi)

        probs = model.predict(roi_input, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs))

        detections.append({
            "index": i,
            "digit": digit,
            "confidence": confidence,
            "role": role,
            "bbox": {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
        })

    # 4. Build polynomial structure (NEW)
    polynomial = build_polynomial(detections)

    # 5. Return JSON-safe output
    return {
        "num_boxes": len(detections),
        "detections": detections,
        "polynomial": polynomial,
        "debug_img": debug_img # for visualization (optional)
    }

def is_box_empty(roi, role="coefficient", debug=False):
    """
    Your exact Sudoku-based logic adapted for worksheet boxes.
    """
    # Ensure grayscale
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    h, w = roi.shape
    total_pixels = h * w
    
    # 1. Blur slightly to reduce noise
    gray = cv2.GaussianBlur(roi, (3, 3), 0)

    # 2. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 3. Remove vertical and horizontal lines (grid artifacts)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 4, 4), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 4, 4)))

    detected_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    thresh_no_lines = cv2.subtract(thresh, detected_h)

    detected_v = cv2.morphologyEx(thresh_no_lines, cv2.MORPH_OPEN, vertical_kernel)
    thresh_no_lines = cv2.subtract(thresh_no_lines, detected_v)
    
    # 4. Remove tiny specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean = cv2.morphologyEx(thresh_no_lines, cv2.MORPH_OPEN, kernel)

    # 5. Connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    
    if num_labels <= 1:
        return True
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    coords = stats[1:, :4]

    # 6. Filter components touching the border
    border_margin = 2
    filtered_components = []
    
    for area, (bx, by, bw, bh) in zip(areas, coords):
        if (bx > border_margin and 
            by > border_margin and 
            bx + bw < w - border_margin and 
            by + bh < h - border_margin):
            
            if bw > 0 and bh > 0:
                aspect_ratio = max(bw, bh) / min(bw, bh)
                if aspect_ratio < 20.0:
                    filtered_components.append((area, bw, bh))

    if len(filtered_components) == 0:
        white = cv2.countNonZero(clean)
        # Threshold: Exponents (smaller) need a lower floor than coefficients
        thresh_limit = 0.003 if role == "exponent" else 0.006
        return white <= max(15, total_pixels * thresh_limit)

    # 7. Analyze the largest valid component
    largest_area, largest_w, largest_h = max(filtered_components, key=lambda x: x[0])
    
    # Adjust sensitivity based on box role
    min_area_ratio = 0.004 if role == "exponent" else 0.006
    
    max_area_ratio = 0.6
    min_dimension = max(3, min(w, h) * 0.05)
    sum_top_n_ratio_thresh = min_area_ratio * 2
    filtered_components_sorted = sorted([a[0] for a in filtered_components], reverse=True)
    top_n = 2
    sum_top_n = sum(filtered_components_sorted[:top_n]) if filtered_components_sorted else 0

    if largest_area < total_pixels * min_area_ratio:
        return True
    if sum_top_n >= total_pixels * sum_top_n_ratio_thresh:
        return False
    if largest_area > total_pixels * max_area_ratio:
        return True
    if largest_w < min_dimension or largest_h < min_dimension:
        return True
    
    return False

# ========================
# Example Usage
# ========================

if __name__ == "__main__":
    image_path = "worksheet_photo.jpg"

    result = extract_polynomial_boxes(image_path)

    _, rois, debug_img = detect_worksheet_boxes(image_path)

    if result["debug_img"] is not None:
        cv2.imwrite("detected_boxes.png", result["debug_img"])
        print("Success: Visualization saved as 'detected_boxes.png'")
    
    for i in range(len(rois)):
        filename = f"debug_roi_{i}.jpg"
        cv2.imwrite(filename, rois[i])
        print(f"Saved ROI {i} to {filename}")

    print(f"Detected {result['num_boxes']} boxes")

    # Print raw detections (optional, good for debugging)
    for d in result["detections"]:
        print(
            f"Box {d['index']}: digit={d['digit']}, "
            f"conf={d['confidence']:.2f}, "
            f"role={d.get('role', 'unknown')}"
        )

    # Pretty-print polynomial
    polynomial = result["polynomial"]

    if not polynomial:
        print("No polynomial detected.")
    else:
        exponents, coefficients = polynomial

        terms = []
        for coef, exp in zip(coefficients, exponents):
            if exp == 0:
                terms.append(str(coef))
            elif exp == 1:
                terms.append(f"{coef}x")
            else:
                terms.append(f"{coef}x^{exp}")

        polynomial_str = " + ".join(terms)
        print("\nDetected polynomial:")
        print(polynomial_str)

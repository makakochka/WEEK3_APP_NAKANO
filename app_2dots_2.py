import cv2
import numpy as np
import sqlite3
from datetime import datetime
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def init_db():
    conn = sqlite3.connect("broccoli.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            field_line INTEGER NOT NULL,
            diameter REAL NOT NULL,
            image_path TEXT NOT NULL,
            size_category TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


# Modified to only store positions without diameters
BROCCOLI_DATA = {
    "Line 1": [(200, 0), (200, 200), (200, 330), (200, 560), (200, 760), (200, 900)],
    "Line 2": [(500, 130), (500, 330), (500, 550), (500, 750), (500, 900)],
    "Line 3": [(870, 55), (870, 330), (870, 540), (870, 750), (870, 930)],
    "Line 4": [(1250, 110), (1250, 330), (1250, 570), (1250, 840)],
    "Line 5": [(1600, 110), (1600, 330), (1600, 570), (1600, 800), (1600, 970)],
}

CIRCLE_DIAMETER = 130  # pixels
FONT_SCALE = 1.0  # Increased font size


def get_size_category(diameter):
    if diameter < 8:
        return "S"
    elif 8 <= diameter < 11:
        return "M"
    elif 11 <= diameter < 14:
        return "L"
    elif 14 <= diameter < 17:
        return "2L"
    else:
        return "XL"


def get_category_color(category):
    return {
        "S": (128, 0, 0),  # Dark Blue
        "M": (0, 255, 0),  # Green
        "L": (255, 165, 0),  # Orange
        "2L": (0, 0, 255),  # Red
        "XL": (128, 0, 128),  # Purple
    }[category]


def calculate_pixels_per_cm(point1, point2):
    """Calculate pixels per cm using known 66cm distance."""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    distance_pixels = np.sqrt(dx * dx + dy * dy)
    return distance_pixels / 66.0


def detect_broccoli_boundary(image, center, pixels_per_cm):
    """
    Detect broccoli head boundary using the center point.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Initial radius estimation (about 10cm typical broccoli head)
    initial_radius = int(10 * pixels_per_cm)
    cv2.circle(mask, center, initial_radius, 255, -1)

    _, shadow_binary = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_and(mask, shadow_binary)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined_mask


def calculate_diameter(mask, center, pixels_per_cm):
    """Calculate diameter from mask and center point."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    distances = []
    for point in contour:
        dx = point[0][0] - center[0]
        dy = point[0][1] - center[1]
        distance = np.sqrt(dx * dx + dy * dy)
        distances.append(distance)

    sorted_distances = sorted(distances)
    final_radius = np.mean(sorted_distances[len(sorted_distances) // 4 : 3 * len(sorted_distances) // 4])

    diameter_cm = (2 * final_radius) / pixels_per_cm
    return round(diameter_cm, 1)


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")

    results = {line_name: [] for line_name in BROCCOLI_DATA.keys()}
    marked_image = image.copy()

    # Calculate pixels_per_cm using first two lines at y=330
    reference_point1 = (200, 330)
    reference_point2 = (500, 330)
    pixels_per_cm = calculate_pixels_per_cm(reference_point1, reference_point2)

    for line_name, centers in BROCCOLI_DATA.items():
        line_number = int(line_name.split()[-1])

        for center in centers:
            # Detect broccoli boundary and calculate diameter
            mask = detect_broccoli_boundary(image, center, pixels_per_cm)
            diameter = calculate_diameter(mask, center, pixels_per_cm)

            if diameter is None:
                continue

            size_category = get_size_category(diameter)
            color = get_category_color(size_category)

            results[line_name].append(
                {
                    "center": center,
                    "diameter": diameter,
                    "size_category": size_category,
                    "field_line": line_number,
                }
            )

            # Draw detection results
            radius = int(diameter * pixels_per_cm / 2)
            cv2.circle(marked_image, center, radius, color, 3)

            label = f"{diameter}cm ({size_category})"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] - radius - 10

            # White outline for visibility
            cv2.putText(
                marked_image,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (255, 255, 255),
                4,
            )
            # Colored text
            cv2.putText(marked_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 2)

    output_path = image_path.replace(".", "_marked.")
    cv2.imwrite(output_path, marked_image)

    return results, output_path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            results, marked_image_path = process_image(filepath)

            # Save to database
            conn = sqlite3.connect("broccoli.db")
            c = conn.cursor()

            for line_data in results.values():
                for result in line_data:
                    c.execute(
                        """
                        INSERT INTO measurements
                        (date, field_line, diameter, image_path, size_category)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            datetime.now().strftime("%Y-%m-%d"),
                            result["field_line"],
                            result["diameter"],
                            marked_image_path,
                            result["size_category"],
                        ),
                    )

            conn.commit()
            conn.close()

            return jsonify({"results": results, "marked_image": marked_image_path.replace("static/", "")})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400


if __name__ == "__main__":
    init_db()
    app.run(debug=True)

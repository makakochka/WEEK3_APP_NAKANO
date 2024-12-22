from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Make sure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Database setup
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


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_size_category(diameter):
    if diameter < 8:
        return "Too Small"
    elif 8 <= diameter < 11:
        return "Size M"
    elif 11 <= diameter < 14:
        return "Size L"
    else:
        return "Too Large"


def process_image(image_path):
    """Process the image and detect broccoli heads"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")

    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define green color range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create mask for green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    marked_image = image.copy()

    for contour in contours:
        # Filter small areas
        area = cv2.contourArea(contour)
        if area < 100:  # Adjust this threshold based on your images
            continue

        # Fit circle to contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Calculate diameter in cm (you'll need to calibrate this)
        diameter_cm = radius * 0.2  # This is an example conversion

        # Store result
        results.append({"center": center, "diameter": diameter_cm, "area": area})

        # Mark on image
        cv2.circle(marked_image, center, radius, (0, 255, 0), 2)
        cv2.putText(
            marked_image,
            f"{diameter_cm:.1f}cm",
            (center[0] - 20, center[1] - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Save marked image
    output_path = image_path.replace(".", "_marked.")
    cv2.imwrite(output_path, marked_image)

    return results, output_path


@app.route("/")
def index():
    return render_template("index_1st_ver.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:  # Added None check
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename and allowed_file(file.filename):  # Added filename check
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process image
            results, marked_image_path = process_image(filepath)

            # Save to database
            conn = sqlite3.connect("broccoli.db")
            c = conn.cursor()
            for result in results:
                size_category = get_size_category(result["diameter"])
                c.execute(
                    """
                    INSERT INTO measurements
                    (date, field_line, diameter, image_path, size_category)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        datetime.now().strftime("%Y-%m-%d"),
                        1,  # Default field line
                        result["diameter"],
                        marked_image_path,
                        size_category,
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

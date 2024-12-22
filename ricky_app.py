from flask import Flask, render_template, request, jsonify
import cv2
import sqlite3
from datetime import datetime
import os
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


# Group broccoli data by lines
BROCCOLI_DATA = {
    "Line 1": [(200, 0, 19), (200, 200, 7), (200, 330, 15), (200, 560, 16), (200, 760, 11), (200, 900, 12)],
    "Line 2": [(500, 130, 7), (500, 330, 11), (500, 550, 15), (500, 750, 9), (500, 900, 18)],
    "Line 3": [(870, 55, 15), (870, 330, 14), (870, 540, 18), (870, 750, 7), (870, 930, 8)],
    "Line 4": [(1250, 110, 8), (1250, 330, 7), (1250, 570, 10), (1250, 840, 6)],
    "Line 5": [(1600, 110, 15), (1600, 330, 16), (1600, 570, 11), (1600, 800, 12), (1600, 970, 13)],
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


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")

    results = {line_name: [] for line_name in BROCCOLI_DATA.keys()}
    marked_image = image.copy()

    for line_name, broccolis in BROCCOLI_DATA.items():
        line_number = int(line_name.split()[-1])  # Extract line number for database
        for x, y, diameter in broccolis:
            size_category = get_size_category(diameter)
            color = get_category_color(size_category)

            results[line_name].append(
                {
                    "center": (x, y),
                    "diameter": diameter,
                    "size_category": size_category,
                    "field_line": line_number,  # Add line number for database
                }
            )

            # Draw circle
            radius = CIRCLE_DIAMETER // 2
            cv2.circle(marked_image, (x, y), radius, color, 3)

            # Draw label with larger font and outline
            label = f"{diameter}cm ({size_category})"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)[0]
            text_x = x - text_size[0] // 2
            text_y = y - radius - 10

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
    return render_template("ricky_index.html")


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

import cv2
import numpy as np
from math import cos, sin, pi


class MultiBroccoliDetector:
    def __init__(self, window_name):
        self.points = []
        self.window_name = window_name
        self.current_image = None
        self.original_image = None
        self.is_selecting = True

    def mouse_callback(self, event, x, y, flags, param):
        if not self.is_selecting:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

            # Create a copy of the image to draw on
            self.current_image = self.original_image.copy()

            # Draw all points
            for idx, point in enumerate(self.points):
                # First two points in blue (calibration points)
                color = (255, 0, 0) if idx < 2 else (0, 0, 255)
                cv2.circle(self.current_image, point, 3, color, -1)

                # Draw calibration line between first two points
                if idx == 1:
                    cv2.line(self.current_image, self.points[0], self.points[1], (255, 0, 0), 2)
                    cv2.putText(
                        self.current_image,
                        "66 cm",
                        (
                            int((self.points[0][0] + self.points[1][0]) / 2) - 30,
                            int((self.points[0][1] + self.points[1][1]) / 2) - 10,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                    )

            # Update instructions
            if len(self.points) < 2:
                text = "Click all broccoli centers. First two points for calibration."
            else:
                text = "Continue clicking centers. Press Enter when done."
            cv2.putText(self.current_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(self.window_name, self.current_image)


def get_all_centers(image_path):
    """
    Let user select all broccoli centers.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    window_name = "Select Broccoli Centers"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detector = MultiBroccoliDetector(window_name)
    detector.original_image = img.copy()
    detector.current_image = img.copy()

    # Set initial instructions
    cv2.putText(
        detector.current_image,
        "Click all broccoli centers. First two points for calibration.",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.setMouseCallback(window_name, detector.mouse_callback)
    cv2.imshow(window_name, detector.current_image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to cancel
            cv2.destroyAllWindows()
            return None
        elif key == 13 and len(detector.points) >= 2:  # Enter to finish (need at least 2 points)
            detector.is_selecting = False
            break

    cv2.destroyAllWindows()
    return detector.points


def calculate_pixels_per_cm(point1, point2):
    """Calculate pixels per cm using known 66cm distance."""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    distance_pixels = np.sqrt(dx * dx + dy * dy)
    return distance_pixels / 66.0


def detect_boundary(image, center):
    """Detect boundary of broccoli head using LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    initial_radius = 100
    cv2.circle(mask, center, initial_radius, 255, -1)

    _, binary = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_and(mask, binary)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined_mask


def calculate_ray_distances(mask, center, num_angles=360):
    """Calculate distances from center to boundary along rays."""
    height, width = mask.shape
    distances = []
    angles = []

    for i in range(num_angles):
        angle = 2 * pi * i / num_angles
        angles.append(angle)
        max_distance = min(width, height)

        for r in range(1, max_distance):
            x = int(center[0] + r * cos(angle))
            y = int(center[1] + r * sin(angle))

            if x < 0 or x >= width or y < 0 or y >= height:
                break

            if mask[y, x] == 0:
                distances.append(r)
                break

    return distances, angles


def calculate_diameter(distances, pixels_per_cm):
    """Calculate diameter using distribution of ray distances."""
    sorted_distances = sorted(distances)
    q1_idx = len(sorted_distances) // 4
    q3_idx = 3 * len(sorted_distances) // 4
    filtered_distances = sorted_distances[q1_idx:q3_idx]

    avg_radius = np.mean(filtered_distances)
    diameter_cm = (2 * avg_radius) / pixels_per_cm

    return round(diameter_cm, 1)


def get_size_category(diameter):
    """Categorize broccoli size."""
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


def process_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    # Get all broccoli centers from user
    points = get_all_centers(image_path)
    if points is None or len(points) < 2:
        return None

    # Calculate scale using first two points
    pixels_per_cm = calculate_pixels_per_cm(points[0], points[1])

    # Process each broccoli
    results = []
    vis_image = img.copy()

    # Draw calibration line
    cv2.line(vis_image, points[0], points[1], (255, 0, 0), 2)
    cv2.putText(
        vis_image,
        "66 cm",
        (int((points[0][0] + points[1][0]) / 2) - 30, int((points[0][1] + points[1][1]) / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )

    # Process each point and organize by lines simultaneously
    line_results = {}

    for idx, center in enumerate(points):
        # Draw clicked point (persistent marker)
        cv2.circle(vis_image, center, 3, (0, 0, 255), -1)  # Red dot for click point

        # Detect boundary
        mask = detect_boundary(img, center)

        # Calculate ray distances
        distances, angles = calculate_ray_distances(mask, center)

        # Calculate diameter
        diameter = calculate_diameter(distances, pixels_per_cm)

        # Get size category
        category = get_size_category(diameter)

        # Store results
        results.append({"center": center, "diameter": diameter, "category": category})

        # Draw measurement text in red
        text = f"D{idx+1}: {diameter}cm ({category})"
        # White outline for better visibility
        cv2.putText(
            vis_image,
            text,
            (center[0] - 50, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            3,
        )  # Thicker white outline
        # Red text
        cv2.putText(
            vis_image, text, (center[0] - 50, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1
        )  # Red text

        # Organize by line
        x_coord = center[0]
        if x_coord < 350:
            line = 1
        elif x_coord < 700:
            line = 2
        elif x_coord < 1050:
            line = 3
        elif x_coord < 1400:
            line = 4
        else:
            line = 5

        if line not in line_results:
            line_results[line] = []
        line_results[line].append({"diameter": diameter, "category": category})

    return results, vis_image, line_results


if __name__ == "__main__":
    try:
        image_path = "C:/Users/makak/Downloads/broccoli_sample.jpg"  # Replace with your image path
        result = process_image(image_path)

        if result:
            results, vis_image, line_results = result

            # Display results
            cv2.namedWindow("Broccoli Measurements", cv2.WINDOW_NORMAL)
            cv2.imshow("Broccoli Measurements", vis_image)

            # Print results in table format
            print("\nMeasurement Results by Line:")
            print("=" * 50)

            for line in sorted(line_results.keys()):
                print(f"\nLine {line}:")
                print("-" * 30)
                for result in line_results[line]:
                    print(f"Diameter: {result['diameter']} cm ({result['category']})")

            print("\nPress 'q' or ESC to exit")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:
                    break

            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        cv2.destroyAllWindows()

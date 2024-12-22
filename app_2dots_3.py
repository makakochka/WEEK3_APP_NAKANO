import cv2
import numpy as np
from math import cos, sin, pi


class BroccoliDetector:
    def __init__(self, window_name):
        self.points = []
        self.window_name = window_name
        self.current_image = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                # Draw the point and update display
                cv2.circle(self.current_image, (x, y), 3, (0, 0, 255), -1)
                if len(self.points) == 1:
                    cv2.putText(
                        self.current_image,
                        "Now click center of second broccoli",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                cv2.imshow(self.window_name, self.current_image)


def get_center_points(image_path):
    """
    Let user select two broccoli centers.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    window_name = "Select Broccoli Centers"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detector = BroccoliDetector(window_name)
    detector.current_image = img.copy()

    # Set instructions
    cv2.putText(
        detector.current_image,
        "Click center of first broccoli",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.setMouseCallback(window_name, detector.mouse_callback)
    cv2.imshow(window_name, detector.current_image)

    while len(detector.points) < 2:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
            cv2.destroyAllWindows()
            return None, None

    cv2.destroyAllWindows()
    return detector.points[0], detector.points[1]


def calculate_pixels_per_cm(point1, point2):
    """Calculate pixels per cm using known 66cm distance."""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    distance_pixels = np.sqrt(dx * dx + dy * dy)
    return distance_pixels / 66.0


def detect_boundary(image, center):
    """
    Detect the boundary of broccoli head using LAB color space.
    Returns binary mask of the detected region.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Create initial circular mask
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    initial_radius = 100  # Starting with a reasonable size
    cv2.circle(mask, center, initial_radius, 255, -1)

    # Threshold for broccoli head
    _, binary = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_and(mask, binary)

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined_mask


def calculate_ray_distances(mask, center, num_angles=360):
    """
    Calculate distances from center to boundary along rays.
    """
    height, width = mask.shape
    distances = []
    angles = []

    for i in range(num_angles):
        angle = 2 * pi * i / num_angles
        angles.append(angle)
        max_distance = min(width, height)

        # Check points along the ray until we hit boundary
        for r in range(1, max_distance):
            x = int(center[0] + r * cos(angle))
            y = int(center[1] + r * sin(angle))

            # Check if point is within image bounds
            if x < 0 or x >= width or y < 0 or y >= height:
                break

            # If we hit black pixel (boundary), record distance
            if mask[y, x] == 0:
                distances.append(r)
                break

    return distances, angles


def visualize_rays(image, center, distances, angles):
    """
    Draw rays and their intersections with boundary.
    """
    output = image.copy()

    for d, angle in zip(distances, angles):
        end_x = int(center[0] + d * cos(angle))
        end_y = int(center[1] + d * sin(angle))
        cv2.line(output, center, (end_x, end_y), (0, 255, 255), 1)
        cv2.circle(output, (end_x, end_y), 2, (0, 0, 255), -1)

    return output


def calculate_diameter(distances, pixels_per_cm):
    """
    Calculate diameter using the distribution of ray distances.
    """
    # Filter out outliers
    sorted_distances = sorted(distances)
    q1_idx = len(sorted_distances) // 4
    q3_idx = 3 * len(sorted_distances) // 4
    filtered_distances = sorted_distances[q1_idx:q3_idx]

    # Calculate average radius and diameter
    avg_radius = np.mean(filtered_distances)
    diameter_cm = (2 * avg_radius) / pixels_per_cm

    return round(diameter_cm, 1)


def process_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    # Get broccoli centers from user
    center1, center2 = get_center_points(image_path)
    if center1 is None or center2 is None:
        return None

    # Calculate scale using known distance
    pixels_per_cm = calculate_pixels_per_cm(center1, center2)

    # Process each broccoli
    results = []
    vis_image = img.copy()

    for center in [center1, center2]:
        # Detect boundary
        mask = detect_boundary(img, center)

        # Calculate ray distances
        distances, angles = calculate_ray_distances(mask, center)

        # Calculate diameter
        diameter = calculate_diameter(distances, pixels_per_cm)

        # Store results
        results.append({"center": center, "diameter": diameter})

        # Visualize rays
        vis_image = visualize_rays(vis_image, center, distances, angles)

    # Draw calibration line
    cv2.line(vis_image, center1, center2, (255, 0, 0), 2)
    cv2.putText(
        vis_image,
        "66 cm",
        (int((center1[0] + center2[0]) / 2) - 30, int((center1[1] + center2[1]) / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )

    # Draw measurements
    for i, result in enumerate(results):
        text = f"D{i+1}: {result['diameter']}cm"
        cv2.putText(vis_image, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return results, vis_image


if __name__ == "__main__":
    try:
        image_path = "C:/Users/makak/Downloads/broccoli_sample_2.jpg"  # Replace with your image path
        results = process_image(image_path)

        if results:
            results, vis_image = results

            # Display results
            cv2.namedWindow("Broccoli Measurement", cv2.WINDOW_NORMAL)
            cv2.imshow("Broccoli Measurement", vis_image)

            print("\nResults:")
            for i, result in enumerate(results):
                print(f"Broccoli {i+1}: {result['diameter']} cm")

            print("\nPress 'q' or ESC to exit")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:
                    break

            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        cv2.destroyAllWindows()

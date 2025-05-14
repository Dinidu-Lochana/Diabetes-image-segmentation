# yolov8/manual_box.py

import cv2  # OpenCV for image display and mouse events
import numpy as np  # For saving prompt points

# ----------- Load Image -----------
image_path = "images/test2.jpg"  # Path to the input image
image = cv2.imread(image_path)  # Load the image
clone = image.copy()  # Keep a copy to reset if needed

# List to store bounding boxes in the format: [x1, y1, x2, y2]
boxes = []

# Drawing state variables
drawing = False  # True when mouse is dragging
ix, iy = -1, -1  # Initial x, y when mouse is pressed

# ----------- Mouse Callback Function -----------
def draw_rectangle(event, x, y, flags, param):
    """
    Mouse callback to draw rectangles (bounding boxes).
    - Press and hold left mouse button to start drawing.
    - Release to finish and save the box.
    """
    global ix, iy, drawing, image, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True  # Start drawing
        ix, iy = x, y  # Record starting point

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False  # Stop drawing
        # Draw a green rectangle from start (ix, iy) to end (x, y)
        cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
        # Save the box coordinates
        boxes.append([ix, iy, x, y])

# ----------- Window Setup -----------
cv2.namedWindow("Image")  # Create a window named "Image"
cv2.setMouseCallback("Image", draw_rectangle)  # Set callback function

# ----------- Main Loop -----------
while True:
    cv2.imshow("Image", image)  # Display the image with drawn boxes
    key = cv2.waitKey(1) & 0xFF  # Wait for key press (1ms)

    if key == ord("r"):
        # Reset image and bounding boxes
        print("Resetting drawing...")
        image = clone.copy()
        boxes = []

    elif key == ord("q"):
        # Quit the loop and proceed to save
        break

# ----------- Cleanup -----------
cv2.destroyAllWindows()  # Close the window

# ----------- Convert Boxes to Center Points -----------
# Format: [[cx1, cy1], [cx2, cy2], ...]
prompt_points = [
    [int((x1 + x2) / 2), int((y1 + y2) / 2)]
    for (x1, y1, x2, y2) in boxes
]

# ----------- Save Prompt Points and Image -----------
np.save("outputs//test2/bbox_points.npy", np.array(prompt_points))  # For SAM prompt
cv2.imwrite("outputs/test2/boxed_image.jpg", image)  # Save image with boxes

# ----------- Final Message -----------
print(f"✅ Saved {len(prompt_points)} prompt points to 'outputs/bbox_points.npy'")
print(f"✅ Saved boxed image to 'outputs/boxed_image.jpg'")
print("💡 Use these points in the next step with SAM-HQ2.")

import cv2
import numpy as np
import time


# Load the YOLOv3 model with OpenCV.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Get the names of all layers in the network.
layer_names = net.getLayerNames()
# Extract the names of the output layers by finding their indices in layer_names.
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialise a list to store the classes names.
classes = []
# Set each line in "coco.names" to an entry in the list, stripping whitespace.
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define a font to be used when displaying class names.
FONT = cv2.FONT_HERSHEY_PLAIN
# Initialise a random color to represent each class.
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Define a confidence threshold  for detections.
conf_thresh = 0.5

# Initialise a video capture object with the first camera.
cap = cv2.VideoCapture(0)
# Initialise a frame counter and get the current time for FPS calculation purposes.
frame_id = 0
time_start = time.time()

while True:
    # Read the current frame from the camera.
    _, frame = cap.read()
    # Add 1 to the frame count every time a frame is read.
    frame_id += 1

    # Pre-process the frame by applying the same scaling used when training the model, resizing to the size
    # expected by this particular YOLOv3 model, and swapping from BGR (used by OpenCV) to RGB (used by the model).
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True)

    # Pass the processed frame through the neural network to get a prediction.
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialise arrays for storing confidence, class ID and coordinate values for detected boxes.
    confidences = []
    class_ids = []
    boxes = []

    # Loop through all the detections in each of the three output scales of YOLOv3.
    for out in outs:
        for detection in out:
            # Get the class probabilities for this box detection.
            scores = detection[5:]
            # Find the class with the highest score for the box.
            class_id = np.argmax(scores)
            # Extract the score of that class.
            confidence = scores[class_id]
            # If that score is higher than the set threshold, execute the code below.
            if confidence > conf_thresh:
                # Get the shape of the unprocessed frame.
                height, width, channels = frame.shape
                # Use the detected box ratios to get box coordinates which apply to the input image.
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Use the center, width and height coordinates to calculate the coordinates for the top left
                # point of the box, which is required for drawing boxes with OpenCV.
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                # Populate the arrays with the information for this box.
                confidences.append(float(confidence))
                class_ids.append(class_id)
                boxes.append([x, y, w, h])

    # Apply non-max suppression to get rid of overlapping boxes.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.4)

    # Iterate through the detected boxes.
    for i in range(len(boxes)):
        # If the box remained after NMS.
        if i in indexes:
            # Extract the coordinates of the box.
            x, y, w, h = boxes[i]
            # Extract the class label from the class ID.
            label = str(classes[class_ids[i]])
            # Extract the confidence for the detected class.
            confidence = confidences[i]
            # Get the color for that class.
            color = colors[class_ids[i]]
            # Draw the box.
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Display the class label and the confidence inside the box.
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), FONT, 2, color, 2)

    # Calculate the elapsed time since starting the loop.
    elapsed_time = time.time() - time_start
    # Calculate the average FPS performance to this point.
    fps = frame_id/elapsed_time
    # Display the FPS at the top left corner.
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (8, 30), FONT, 2, (0, 0, 0), 2)

    # Show the frame.
    cv2.imshow("Camera", frame)
    # Wait at least 1ms for key event and record the pressed key.
    key = cv2.waitKey(1)
    # If the pressed key is ESC (27), break the loop.
    if key == 27:
        break

# Release the camera and destroy all windows.
cap.release()
cv2.destroyAllWindows()

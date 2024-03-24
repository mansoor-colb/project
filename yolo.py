import cv2
import numpy as np
import pyautogui
import mediapipe as mp

selected_color = None
tracking_mode = "hand"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
print("Layer Names:", layer_names)
output_layers = []

# Get the layer names
layer_names = net.getLayerNames()

# Iterate over the indices of layer_names
for i, name in enumerate(layer_names):
    # Check if the layer is one of the output layers
    if i in net.getUnconnectedOutLayers():
        # Append the layer name to output_layers
        output_layers.append(name)

# Print the output layer names
print("Output Layer Names:", output_layers)
# Function to perform object detection using YOLO
# def detect_color(frame):
#     height, width, _ = frame.shape
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#
#     colors = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5 and class_id == 0:  # We assume class_id 0 corresponds to the color of interest
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 colors.append((center_x, center_y))
#     return colors

def detect_color(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    colors = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Assuming class_id 0 corresponds to the color of interest
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                colors.append((center_x, center_y))
    return colors

def mouse_callback(event, x, y, flags, param):
    global selected_color

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color = frame[y, x]

        print("Selected Color:", selected_color)

def detect_hand(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Assuming only one hand is detected

        index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        height, width, _ = frame.shape

        index_finger_x = int(index_finger.x * width)

        index_finger_y = int(index_finger.y * height)

        return index_finger_x, index_finger_y

    return None

def main():
    global frame, tracking_mode, selected_color

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    screen_width = 1920  # Set your desired screen width
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (screen_width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    cv2.namedWindow("Color Selection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Color Selection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Color Selection", mouse_callback)

    print("Color Selection Phase. Click on a color to track.")

    while selected_color is None:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (screen_width, screen_height))

        detected_colors = detect_color(frame)
        for color in detected_colors:
            cv2.circle(frame, color, 10, (0, 255, 0), -1)

        cv2.imshow("Color Selection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.namedWindow("Object Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Press 'h' for hand detection, 'l' for laser tracking, and 'q' to quit.")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (screen_width, screen_height))

        if tracking_mode == "hand":
            index_finger_position = detect_hand(frame)
            if index_finger_position:
                print(index_finger_position)
                cv2.circle(frame, index_finger_position, 10, (0, 255, 0), -1)
                pyautogui.moveTo(index_finger_position[0], index_finger_position[1])

        elif tracking_mode == "laser":
            detected_colors = detect_color(frame)
            if detected_colors:
                for color in detected_colors:
                    pyautogui.moveTo(color[0], color[1])

        cv2.imshow("Object Detection", frame)

        key = cv2.waitKey(1)

        if key == ord('h'):
            tracking_mode = "hand"
            print("Switched to hand detection mode.")

        elif key == ord('l'):
            tracking_mode = "laser"
            print("Switched to laser tracking mode.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

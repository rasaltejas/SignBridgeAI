import cv2   # installed for video capture and image processing
import mediapipe as mp   # installed for hand detection and landmark extraction

# Step 1: Initialize MediaPipe
mp_hands = mp.solutions.hands                       # For hand detection
mp_drawing = mp.solutions.drawing_utils             # For drawing landmarks and connections
hands = mp_hands.Hands(
    static_image_mode=False,        # For video
    max_num_hands=2,                # Detect both hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Step 2: Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while cap.isOpened():  
    ret, frame = cap.read()  # Read frame
    if not ret:
        break
    
    # Step 3: Flip for mirror + convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    
    # Step 4: Process frame
    results = hands.process(rgb)
    
    # Step 5: Draw landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw 21 landmarks & connections
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
            )
            
            # Optional: Extract coordinates
            for idx, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"Landmark {idx}: ({cx}, {cy})")  # For debugging
    
    # Step 6: Show output
    cv2.imshow('SignBridge AI - Hand Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 7: Cleanup
cap.release()
cv2.destroyAllWindows()
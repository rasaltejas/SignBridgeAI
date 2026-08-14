# SignBridgeAI
Build step by step
## 1. Hand detection

   webcam -> Detect hand -> Draw 21 landmarks
  
   used ( python, opencv, Mediapipe, Mediapipe processing, drawing landmarks )
   
   pip install opencv-python mediapipe tensorflow numpy pandas scikit-learn matplotlib
   ### Packages:- 
   1.[opencv-python used to open and display webcam, read frames, draw on images]
              
   2.[mediapipe Detect hands and return 21 landmarks (x, y, z) per hand]
              
   3.[numpy Required by MediaPipe/OpenCV for array operations]
              
   4.[pandas Manage labeled datasets (e.g., CSV of landmarks + labels)]
              
   5.[tensorflow Build & train CNNs/LSTMs for gesture classification]
              
   6.[scikit-learn Train/test splits, accuracy metrics, preprocessing]
              
   7.[matplotlib Visualize training loss, accuracy, or landmark plots]
#### STEPS
Step 1: Initialize MediaPipe --> Step 2: Open webcam --> Step 3: Flip for mirror + convert BGR to RGB --> Step 4: Process frame --> Step 5: Draw landmarks if detected --> Step6: Show output --> Step 7: Cleanup

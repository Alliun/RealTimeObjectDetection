from function import *
import cv2
from keras.models import load_model

# Load the trained model
model = load_model('model.h5')
test_actions = ['A', 'B', 'C', 'D', 'E']

# Initialize camera
cap = cv2.VideoCapture(0)

# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    sequence = []
    predictions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make detections
        image, results = mediapipe_detection(frame, hands)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only last 30 frames
        
        # Show hand detection status
        if results.multi_hand_landmarks:
            cv2.putText(image, 'Hand Detected', (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'No Hand Detected', (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(image, f'Frames: {len(sequence)}/30', (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(sequence) == 30:
            # Make prediction
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_action = test_actions[np.argmax(res)]
            confidence = np.max(res)
            
            # Show all predictions with lower threshold
            if confidence > 0.3:  # Lower threshold
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                cv2.putText(image, f'Prediction: {predicted_action}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                cv2.putText(image, 'Uncertain', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(image, f'Confidence: {confidence:.2f}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Hand Gesture Recognition', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
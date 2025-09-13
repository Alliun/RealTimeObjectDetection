from function import *
import cv2

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Set mediapipe model 
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Make detections
            image, results = mediapipe_detection(frame, hands)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            print(f"Keypoints shape: {keypoints.shape}")
            
            # Show to screen
            cv2.imshow('Hand Detection', image)
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
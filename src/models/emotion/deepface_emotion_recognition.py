import cv2
from deepface import DeepFace

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    # Analyze the current frame for emotion
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Debugging: Print the result structure
        print("Result:", result)

        # Check if the result is a list and has elements
        if isinstance(result, list) and len(result) > 0:
            # Extract the first result (if multiple faces are detected)
            first_result = result[0]

            # Get the predicted emotion and its confidence score
            dominant_emotion = first_result['dominant_emotion']
            emotion_scores = first_result['emotion']

            # Filter emotions to include only happy, neutral, and surprise
            filtered_emotions = {key: emotion_scores[key] for key in ['happy', 'neutral', 'surprise'] if key in emotion_scores}

            # Display the dominant emotion on the frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Optionally, display the filtered emotions with their confidence scores
            for i, (emotion, score) in enumerate(filtered_emotions.items()):
                text = f"{emotion}: {score:.2f}%"
                cv2.putText(frame, text, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        else:
            print("No faces detected.")

    except Exception as e:
        print(f"Error in DeepFace analysis: {e}")

    # Display the frame with emotion predictions
    cv2.imshow('Emotion Detector', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import speech
import cv2
import detect
import datetime
from ultralytics import YOLO
from transformers import pipeline
import time
import threading
import queue
# Initialize image-to-text model
detect_model = pipeline("image-to-text", model="describe.h5")
model = YOLO("yolov8n.pt")

print("hello")
engine=speech.speech_to_text()
# Global flags and variables
listening = False
intent = " "
frame_count = 0
url = "http://192.168.224.220:8080/video"
max_size=5
frames = []# Circular buffer for frames
  # Lock for thread-safe access to the buffer

# Function to handle speech-to-text conversion in a separate thread
def listen_for_commands():
    global listening
    
    while True:
        if not listening:
            engine.text_speech('powered on')
            print('listening')
            resp = engine.recognize_speech_from_mic()
            print("Speech: ", resp)
            if resp:
                intent, text = detect.detect_intent_texts([resp])
                print("Intent:", intent, "Text:", text)
                if intent == 'wakeup' and resp:
                    listening = True
        else:
            engine.text_speech("What can I help you with?")
            engine.text_speech("Listening...")
            resp = engine.recognize_speech_from_mic()
            print("Speech: ", resp)

            if resp:
                intent, text = detect.detect_intent_texts([resp])
                print("Intent:", intent, "Text:", text)

                # Execute actions based on detected intent
                handle_intent(intent, text, resp)
        time.sleep(0.1)
# Function to handle actions based on detected intent
def handle_intent(intent, text, resp):
    global listening

    # Retrieve the most recent frame from the buffer
    
    
    frame = frames[-1] # Get the most recent frame
    

    if frame is not None:
        if intent == 'Describe':
            detect.describeScene(frame, detect_model,engine)
        elif intent == 'endconvo':
            print(text)
            listening = False
            engine.text_speech(text)
        elif intent == 'Brightness':
            brightness = detect.getBrightness(frame)
            engine.text_speech(f"It is {brightness[0]} outside")
        elif intent == "FillForm":
            detect.detect_form(frame,engine)
        elif intent == "Read":
            detect.detect_text(frame,engine)
        elif intent == "Time":
            currentDT = datetime.datetime.now()
            engine.text_speech(f"The time is {currentDT.hour} hours and {currentDT.minute} minutes")
        elif intent == "objects":
            detect.tellObjects( frame,model,engine)
        # elif intent == "find":
        #     detect.find_user(frame, model)
        # elif intent == "recognize":
        #     pass
        elif resp != 'None':
            engine.text_speech(text)

# Function to capture and process frames
def capture_and_process_frames():
    global frame_count
    cam = cv2.VideoCapture(url)
      # Process every nth frame
    
    while True:
        ret, frame = cam.read()
        

        if  ret:
            
            frames.append(frame)  # Add the frame to the buffer
            if len(frames) >= max_size:
                frames.pop(0)
            # Process the frame if needed (optional)
            # Example: detect.process_frame(frame)

        # Display frame (optional)
        # cv2.imshow("Camera Feed", frames.get())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break


    cam.release()
    cv2.destroyAllWindows()

# Main function to run the program
def main():
    capture_thread = threading.Thread(target=capture_and_process_frames, daemon=True)
    capture_thread.start()
    # Start the listening thread
    listen_for_commands()

    # Start the frame capture and processing thread
    

    # Keep the main thread alive to allow continuous operation
    try:
        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("Exiting program...")
        

if __name__ == "__main__":
    main()

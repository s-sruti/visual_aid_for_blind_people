import cv2
import pytesseract
from PIL import Image
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def getBrightness(frame):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = np.sum(frame)/(frame.shape[0]*frame.shape[1])
    avg=avg/255
    if(avg > 0.6):
        return ("Very bright", avg)
    if(avg > 0.4):
        return ("Bright", avg)
    if(avg>0.2):
        return ("Dim", avg)
    else:
        return ("Dark",avg)


def detect_text(frame,engine):
    
    cv2.imwrite('\\assets\\op.jpg', frame)
    img = cv2.imread('\\assets\\op.jpg')
    text = pytesseract.image_to_string(img)
    print('Detected text:', text)
    engine.text_speech(text)


def detect_form(frame,engine):

    
    
    
    img = cv2.imread('\\assets\\bank.jpg')
    

    text = pytesseract.image_to_string(img)
    
    lines=text.split('/n')
    engine.text_speech("the form is detected")
    full_text = ""
    for i, line in enumerate(lines):
        
        if not line.strip():
            continue

        if i == 0:
            engine.text_speech("The form is entitled as:")
        elif i == 1:
            engine.text_speech("The form asks about these details:")

        # engine.text_speech the recognized text
        engine.text_speech(line)

        # Concatenate the lines into a full text string
        full_text += line + " "

    print(full_text)


def detect_intent_texts(texts):
    # Define intents and their associated keywords
    intents = {
        "wakeup": ["start","wake up","hello"],
        "Describe": ["describe", "tell me about", "explain"],
        "endconvo": ["end", "stop", "quit"],
        "Brightness": ["brightness", "how bright", "light level"],
        "FillForm": ["form", "fill form", "complete form", "submit form"],
        "Read": ["read", "read text", "show text"],
        "Time": ["time", "what time", "current time"],
        "objects": ["find objects", "objects", "things", "thing", "front of me"]
        # "find": ["find", "reach", "navigate", "search", "guide"]
    }
    
    text_lower=' '  
    for text in texts:
        text_lower+=" "+text.lower()
    text_lower=text_lower.split(' ')
    
    detected_intent = "Unknown"
    fulfillment_text = "I'm sorry, I didn't understand that."

    # Iterate over the texts and check for keyword matches in intents
    for text_lower in text_lower:
        for intent, keywords in intents.items():
            
            # Check if any of the keywords are present in the text
            if any(keyword in text_lower for keyword in keywords):
                
                detected_intent = intent
                fulfillment_text = f"Detected intent: {intent}. What would you like to do next?"
                break
        if detected_intent != "Unknown":
            break  # Exit if a valid intent is detected

    return detected_intent, fulfillment_text



def describeScene(frame, model,engine):
    
    cv2.imwrite('\\assets\\des.jpg', frame)
    
    img= Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    description = model(img)[0]['generated_text']
    print(description)
    engine.text_speech(description)


def tellObjects(frame, model, engine):
    # Run object detection
   
    results = model(frame, conf=0.3)
    
    # Convert frame to a copy for drawing bounding boxes
    frame_with_boxes = frame.copy()
    
    for detection in results[0].boxes:
        # Get the label and bounding box coordinates
        label = model.names[int(detection.cls[0])]
       
        # Speak the detected object
        engine.text_speech(label)
        print(label)
    
    # Save the frame with bounding boxes for debugging/viewing
    
    
    return  results




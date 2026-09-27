import io
import speech_recognition as sr
import os
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
from google.cloud import texttospeech
import os
import pygame
import cv2
from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import datetime
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
import torch
import random
import face_recognition
import threading
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize pygame for audio playback
pygame.mixer.init()


# Load the MobileNetV2 model pre-trained on ImageNet
modelMNV2 = MobileNetV2(weights='imagenet')


# Load the pre-trained Faster R-CNN model from torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Global flag for stopping speech
stop_speaking = False
stop_thread = None
long_length = False
usable_mic = 2 #1-Logitech Headphones, 2-PNP Mic, 4 - Airpods

# Trivia Question List
trivia_questions = [
    {"question": "What is the smallest planet in our solar system?", "answer": "Mercury"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"question": "What is the largest island in the world?", "answer": "Greenland"},
    {"question": "Which country is known as the Land of the Free?", "answer": "Thailand"},
    {"question": "What is the hardest natural substance on Earth?", "answer": "Diamond"},
    {"question": "Who wrote 'Pride and Prejudice'?", "answer": "Jane Austen"},
    {"question": "What is the largest organ in the human body?", "answer": "Skin"},
    {"question": "What is the chemical symbol for iron?", "answer": "Fe"},
    {"question": "What is the longest river in South America?", "answer": "The Amazon River"},
    {"question": "Which planet is known as the Morning Star or the Evening Star?", "answer": "Venus"},
    {"question": "Who is the author of the Harry Potter series?", "answer": "J.K. Rowling"},
    {"question": "What is the smallest bone in the human body?", "answer": "The stapes"},
    {"question": "What is the largest city by population in the world?", "answer": "Tokyo"},
    {"question": "What is the name of the largest volcano in the solar system?", "answer": "Olympus Mons"},
    {"question": "Which famous scientist is known for the laws of motion?", "answer": "Isaac Newton"},
    {"question": "Who directed 'Jurassic Park'?", "answer": "Steven Spielberg"},
    {"question": "What is the name of the first manned moon landing mission?", "answer": "Apollo 11"},
    {"question": "What is the main ingredient in sushi?", "answer": "Rice"},
    {"question": "What element is used in thermometers and is a liquid at room temperature?", "answer": "Mercury"},
    {"question": "Who wrote the 'A Song of Ice and Fire' series?", "answer": "George R.R. Martin"},
    {"question": "What is the capital of Canada?", "answer": "Ottawa"},
    {"question": "Which country is famous for inventing pizza?", "answer": "Italy"},
    {"question": "What is the name of the galaxy that contains our solar system?", "answer": "The Milky Way"},
    {"question": "Which ocean is the largest?", "answer": "The Pacific Ocean"},
    {"question": "Who painted 'Starry Night'?", "answer": "Vincent van Gogh"},
    {"question": "What is the main language spoken in Brazil?", "answer": "Portuguese"},
    {"question": "What is the smallest unit of life?", "answer": "The cell"},
    {"question": "Who is the author of 'To Kill a Mockingbird'?", "answer": "Harper Lee"},
    {"question": "What is the currency of Japan?", "answer": "Yen"},
    {"question": "What is the name of the largest desert in the world?", "answer": "The Antarctic Desert"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "What is the capital of Australia?", "answer": "Canberra"},
    {"question": "Who is the Greek goddess of wisdom?", "answer": "Athena"},
    {"question": "What is the tallest mountain in the world?", "answer": "Mount Everest"},
    {"question": "What is the name of the largest moon of Saturn?", "answer": "Titan"},
    {"question": "Who is known as the father of modern physics?", "answer": "Albert Einstein"},
    {"question": "Which famous structure was built as a mausoleum for the Mughal Emperor's wife?", "answer": "The Taj Mahal"},
    {"question": "What is the longest-running American television show?", "answer": "The Simpsons"},
    {"question": "What is the largest species of shark?", "answer": "The Whale Shark"},
    {"question": "What year did the Berlin Wall fall?", "answer": "1989"},
    {"question": "What is the chemical symbol for sodium?", "answer": "Na"},
    {"question": "Who wrote the novel '1984'?", "answer": "George Orwell"},
    {"question": "Which country is known for the pyramids?", "answer": "Egypt"},
    {"question": "What is the largest coral reef system in the world?", "answer": "The Great Barrier Reef"},
    {"question": "Who was the 16th President of the United States?", "answer": "Abraham Lincoln"},
    {"question": "What is the main ingredient in guacamole?", "answer": "Avocado"},
    {"question": "What is the largest city in Africa?", "answer": "Lagos"},
    {"question": "Which artist is famous for the painting 'The Persistence of Memory'?", "answer": "Salvador Dalí"},
    {"question": "What is the capital city of Brazil?", "answer": "Brasília"},
    {"question": "Who invented the telephone?", "answer": "Alexander Graham Bell"},
    {"question": "What is the currency of South Korea?", "answer": "Won"},
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
    {"question": "Who wrote 'The Catcher in the Rye'?", "answer": "J.D. Salinger"},
    {"question": "What is the name of the famous clock tower in London?", "answer": "Big Ben"},
    {"question": "Which country is known for the Eiffel Tower?", "answer": "France"},
    {"question": "What is the chemical symbol for potassium?", "answer": "K"},
    {"question": "Who wrote the play 'Hamlet'?", "answer": "William Shakespeare"},
    {"question": "What is the largest land animal?", "answer": "African Elephant"},
    {"question": "What planet is known as the 'Red Planet'?", "answer": "Mars"},
    {"question": "What is the name of the ship that brought the Pilgrims to America?", "answer": "Mayflower"},
    {"question": "What is the capital city of Italy?", "answer": "Rome"},
    {"question": "Who was the first person to step on the moon?", "answer": "Neil Armstrong"},
    {"question": "What is the most common element in the Earth's crust?", "answer": "Oxygen"},
    {"question": "What is the capital of Spain?", "answer": "Madrid"},
    {"question": "Who painted the 'Birth of Venus'?", "answer": "Sandro Botticelli"},
    {"question": "What is the currency of the United Kingdom?", "answer": "Pound Sterling"},
    {"question": "Which famous building was completed in 1889 and is located in Paris?", "answer": "The Eiffel Tower"},
    {"question": "Who is known for the theory of evolution by natural selection?", "answer": "Charles Darwin"},
    {"question": "What is the name of the ancient city known for its rock-cut architecture in Jordan?", "answer": "Petra"},
    {"question": "Who was the famous Italian explorer who sailed to the Americas in 1492?", "answer": "Christopher Columbus"},
    {"question": "What is the smallest planet in our solar system?", "answer": "Mercury"},
    {"question": "Which country is known for the Great Wall?", "answer": "China"},
    {"question": "What is the chemical symbol for silver?", "answer": "Ag"},
    {"question": "Who was the first woman to win a Nobel Prize?", "answer": "Marie Curie"},
    {"question": "What is the name of the process by which plants make their own food?", "answer": "Photosynthesis"},
    {"question": "What is the capital city of Mexico?", "answer": "Mexico City"},
    {"question": "Which sea is the world's largest inland body of water?", "answer": "The Caspian Sea"},
    {"question": "Who is the author of 'The Hobbit'?", "answer": "J.R.R. Tolkien"},
    {"question": "What is the name of the longest river in Africa?", "answer": "The Nile"},
    {"question": "What is the most abundant gas in the Earth's atmosphere?", "answer": "Nitrogen"},
    {"question": "Who was the British Prime Minister during World War II?", "answer": "Winston Churchill"},
    {"question": "What is the name of the process by which water vapor turns into liquid water?", "answer": "Condensation"},
    {"question": "Who painted the Sistine Chapel ceiling?", "answer": "Michelangelo"},
    {"question": "What is the largest city in the United States by area?", "answer": "Sitka, Alaska"},
    {"question": "Which country is known for its fjords?", "answer": "Norway"},
    {"question": "What is the capital of Sweden?", "answer": "Stockholm"},
    {"question": "Who wrote the novel 'The Great Gatsby'?", "answer": "F. Scott Fitzgerald"},
    {"question": "What is the tallest waterfall in the world?", "answer": "Angel Falls"},
    {"question": "What is the most common blood type in humans?", "answer": "O positive"},
    {"question": "What is the smallest unit of an element?", "answer": "Atom"},
    {"question": "Who invented the World Wide Web?", "answer": "Tim Berners-Lee"},
    {"question": "What is the main language spoken in Egypt?", "answer": "Arabic"},
    {"question": "Who painted the 'School of Athens'?", "answer": "Raphael"},
    {"question": "What is the largest mammal in the ocean?", "answer": "Blue Whale"},
    {"question": "What is the name of the famous clock tower in London?", "answer": "Big Ben"},
    {"question": "Which country is known as the Land of the Rising Sun?", "answer": "Japan"},
    {"question": "What is the name of the river that runs through Egypt?", "answer": "The Nile"},
    {"question": "Who was the 16th President of the United States?", "answer": "Abraham Lincoln"},
    {"question": "What is the name of the phenomenon where the moon completely covers the sun?", "answer": "Total Solar Eclipse"},
    {"question": "Who is known as the 'Father of Modern Physics'?", "answer": "Albert Einstein"},
    {"question": "What is the capital city of Canada?", "answer": "Ottawa"},
    {"question": "Who wrote the play 'Romeo and Juliet'?", "answer": "William Shakespeare"},
    {"question": "What is the largest ocean on Earth?", "answer": "The Pacific Ocean"},
    {"question": "Who was the first man to step on the moon?", "answer": "Neil Armstrong"},
    {"question": "What is the longest river in the world?", "answer": "The Nile"},
    {"question": "Who is the Greek god of the sea?", "answer": "Poseidon"},
    {"question": "What is the currency of Japan?", "answer": "Yen"},
    {"question": "What is the most widely spoken language in the world?", "answer": "Mandarin Chinese"},
    {"question": "Who wrote 'The Catcher in the Rye'?", "answer": "J.D. Salinger"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
    {"question": "Who was the first President of the United States?", "answer": "George Washington"},
    {"question": "What is the chemical symbol for oxygen?", "answer": "O"},
    {"question": "What is the largest desert in the world?", "answer": "The Antarctic Desert"},
    {"question": "Who is known for developing the theory of relativity?", "answer": "Albert Einstein"},
    {"question": "What is the capital of Brazil?", "answer": "Brasília"},
    {"question": "What is the name of the famous landmark in Paris?", "answer": "Eiffel Tower"},
    {"question": "Who is the Greek goddess of wisdom?", "answer": "Athena"},
    {"question": "What is the currency of South Korea?", "answer": "Won"},
    {"question": "What is the smallest country in the world by land area?", "answer": "Vatican City"},
    {"question": "Who painted the 'Last Supper'?", "answer": "Leonardo da Vinci"},
    {"question": "What is the name of the first manned mission to the moon?", "answer": "Apollo 11"},
    {"question": "Who was the famous Italian explorer who sailed to the Americas in 1492?", "answer": "Christopher Columbus"},
    {"question": "What is the name of the tallest mountain in Africa?", "answer": "Mount Kilimanjaro"},
    {"question": "What is the currency of India?", "answer": "Indian Rupee"},
    {"question": "Who wrote 'Brave New World'?", "answer": "Aldous Huxley"},
    {"question": "What is the national sport of Canada?", "answer": "Ice Hockey"},
    {"question": "Which country won the FIFA World Cup in 2018?", "answer": "France"},
    {"question": "Who holds the record for the most goals scored in a single World Cup tournament?", "answer": "Just Fontaine"},
    {"question": "What is the maximum score in a single game of ten-pin bowling?", "answer": "300"},
    {"question": "Which NBA player is known as 'King James'?", "answer": "LeBron James"},
    {"question": "In which city were the first modern Olympic Games held?", "answer": "Athens"},
    {"question": "How many players are on a standard soccer team?", "answer": "11"},
    {"question": "What is the term for a score of 0 in tennis?", "answer": "Love"},
    {"question": "Which country has won the most Rugby World Cups?", "answer": "New Zealand"},
    {"question": "Who is known as the 'Greatest of All Time' (GOAT) in boxing?", "answer": "Muhammad Ali"},
    {"question": "Who was the first President of the United States?", "answer": "George Washington"},
    {"question": "What year did the Titanic sink on its maiden voyage?", "answer": "1912"},
    {"question": "Which empire was ruled by Julius Caesar?", "answer": "The Roman Empire"},
    {"question": "Who was the first woman to win a Nobel Prize?", "answer": "Marie Curie"},
    {"question": "What was the name of the ship that carried the Pilgrims to America in 1620?", "answer": "The Mayflower"},
    {"question": "In what year did the Berlin Wall fall?", "answer": "1989"},
    {"question": "Which ancient civilization built the Machu Picchu complex in Peru?", "answer": "The Inca"},
    {"question": "Who was the leader of the Soviet Union during World War II?", "answer": "Joseph Stalin"},
    {"question": "What was the name of the first manned mission to the moon?", "answer": "Apollo 11"},
    {"question": "Which war was fought between the North and South in the United States from 1861 to 1865?", "answer": "The American Civil War"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Which planet is known as the Red Planet?", "answer": "Mars"},
    {"question": "Who wrote the play 'Romeo and Juliet'?", "answer": "William Shakespeare"},
    {"question": "What is the capital of Australia?", "answer": "Canberra"},
    {"question": "In which year did the first iPhone release?", "answer": "2007"},
    {"question": "What is the hardest natural substance on Earth?", "answer": "Diamond"},
    {"question": "Who painted the ceiling of the Sistine Chapel?", "answer": "Michelangelo"},
    {"question": "What is the largest ocean on Earth?", "answer": "The Pacific Ocean"},
    {"question": "How many continents are there on Earth?", "answer": "Seven"},
    {"question": "Which animal is known as the 'King of the Jungle'?", "answer": "Lion"},
    {"question": "What is the smallest country in the world by land area?", "answer": "Vatican City"},
    {"question": "What gas do plants primarily use for photosynthesis?", "answer": "Carbon Dioxide"},
    {"question": "Who is known as the 'Father of Computers'?", "answer": "Charles Babbage"},
    {"question": "Which element has the atomic number 1?", "answer": "Hydrogen"},
    {"question": "What is the capital city of Japan?", "answer": "Tokyo"},
    {"question": "What is the longest river in the world?", "answer": "The Nile"},
    {"question": "Who invented the lightbulb?", "answer": "Thomas Edison"},
    {"question": "Which famous scientist developed the theory of relativity?", "answer": "Albert Einstein"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What musical instrument has 88 keys?", "answer": "Piano"},
    {"question": "Which country is known as the Land of the Rising Sun?", "answer": "Japan"},
    {"question": "In which city would you find the Eiffel Tower?", "answer": "Paris"},
    {"question": "What is the largest mammal in the world?", "answer": "Blue Whale"},
    {"question": "Who wrote the novel '1984'?", "answer": "George Orwell"},
    {"question": "What is the boiling point of water in degrees Celsius?", "answer": "100°C"},
    {"question": "What is the largest desert in the world?", "answer": "The Sahara Desert"},
    {"question": "What is the name of the fairy in Peter Pan?", "answer": "Tinker Bell"},
    {"question": "Which famous shipwreck is the subject of a 1997 film directed by James Cameron?", "answer": "The Titanic"},
    {"question": "What is the main ingredient in guacamole?", "answer": "Avocado"},
    {"question": "Which planet is known for its rings?", "answer": "Saturn"}
]

# COCO dataset class labels
COCO_INSTANCE_CATEGORY_NAMES = [
   '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
   'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
   'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
   'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
   'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
   'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
   'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv',
   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
   'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
   'hair drier', 'toothbrush'
]


def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    # Iterate over each person in the known_faces directory
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            # Load each image file in the person's directory
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                image = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    known_encodings.append(encoding[0])
                    known_names.append(person_name)
    print("Loading completed")
    return known_encodings

def get_openai_response(prompt):
   """Get a response from OpenAI GPT-4 API using the chat completion endpoint."""
   response = client.chat.completions.create(model="gpt-4o-mini-2024-07-18",  # or "gpt-3.5-turbo"
   messages=[
       {"role": "user", "content": prompt}
   ])
   return response.choices[0].message.content.strip()

def text_to_speech(text):
   """Convert text to speech using Google Cloud Text-to-Speech API."""
   client = texttospeech.TextToSpeechClient()


   synthesis_input = texttospeech.SynthesisInput(text=text)
   voice = texttospeech.VoiceSelectionParams(
       language_code="en-US", name="en-US-Wavenet-D"
   )
   audio_config = texttospeech.AudioConfig(
       audio_encoding=texttospeech.AudioEncoding.MP3
   )


   try:
       response = client.synthesize_speech(
           input=synthesis_input, voice=voice, audio_config=audio_config
       )


       # Use io.BytesIO to handle audio data in memory
       audio_stream = io.BytesIO(response.audio_content)


       return audio_stream


   except Exception as e:
       print(f"An error occurred during text-to-speech conversion: {e}")
       return None

def play_audio(audio_stream):
   """Play audio from an in-memory stream using pygame."""
   global stop_speaking
   global stop_thread
   global long_length

   # Reset the stream position to the beginning
   audio_stream.seek(0)

   # Load the audio stream into pygame mixer
   pygame.mixer.music.load(audio_stream)
   pygame.mixer.music.play()
   if long_length == True:
       stop_thread = threading.Thread(target=listen_for_stop_command)
       stop_thread.start()

   # Wait while the audio is playing
   while pygame.mixer.music.get_busy() and not stop_speaking:
       pygame.time.Clock().tick(10)


def listen_to_microphone():
   """Listen to audio from the selected microphone and convert it to text."""
   global usable_mic
   recognizer = sr.Recognizer()
   mic = sr.Microphone(usable_mic)


   with mic as source:
       print("Listening...")
       audio = recognizer.listen(source)


   try:
       text = recognizer.recognize_google(audio)
       print(f"You said: {text}")
       return text
   except sr.UnknownValueError:
       print("Sorry, I did not understand that.")
       return None
   except sr.RequestError:
       print("Sorry, there was an issue with the speech recognition service.")
       return None

def listen_for_stop_command():
    """Listen for 'Bruce, stop' command and set the stop_speaking flag."""
    global stop_speaking
    global usable_mic
    print("Say stop for dictation to end")
    recognizer = sr.Recognizer()
    mic = sr.Microphone(usable_mic)
    while pygame.mixer.music.get_busy():
        with mic as source:
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            if text:
                if "stop" in text or "shut up" in text:
                    print("Command detected, Bruce stopping")
                    stop_speaking = True
                    pygame.mixer.music.stop()
                    stop_speaking = False
                    audio_stream = text_to_speech("Sorry, stopping dictation")
                    play_audio(audio_stream)
                    break
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")


def capture_and_display_image():
   """Capture an image from the webcam and display it."""
   # Open a connection to the webcam
   cap = cv2.VideoCapture(0)  # 0 is the default webcam


   if not cap.isOpened():
       print("Could not open webcam.")
       return


   # Capture a single frame
   ret, frame = cap.read()
   if not ret:
       print("Failed to capture image.")
       return

   # Release the webcam
   cap.release()

   # Create a unique filename using the current timestamp
   timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
   filename = f"captured_image_{timestamp}.jpg"


   # Save the captured image
   cv2.imwrite(filename, frame)
   print(f"Image saved as {filename}")
   audio_stream = text_to_speech("Image Captured")
   play_audio(audio_stream)


   # Display the captured image
   cv2.imshow('Captured Image', frame)
   cv2.waitKey(0)
   cv2.destroyAllWindows()


def detect_faces_and_display():
   """Capture an image, detect faces, and display it with face boxes."""
   # Open a connection to the webcam
   cap = cv2.VideoCapture(0)  # 0 is the default webcam


   if not cap.isOpened():
       print("Could not open webcam.")
       return


   # Capture a single frame
   ret, frame = cap.read()
   if not ret:
       print("Failed to capture image.")
       return

   # Release the webcam
   cap.release()

   # Convert the image to grayscale
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


   # Load OpenCV's pre-trained Haar Cascade classifier for face detection
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


   # Perform face detection
   faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


   # prints faces detected
   print(f"Number of faces detected: {len(faces)}")


   # Draw rectangles around detected faces
   for (x, y, w, h) in faces:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


   # Create a unique filename using the current timestamp
   timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
   filename = f"face_detection_image_{timestamp}.jpg"


   # Save the captured image
   cv2.imwrite(filename, frame)
   print(f"Analyzed image saved as {filename}")

    # Information
   audio_stream = text_to_speech(f"Sir, {len(faces)} faces has been detected")
   play_audio(audio_stream)


   # Display the image with detected faces
   cv2.imshow('Detected Faces', frame)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

def capture_and_analyze_object():
   """Capture an image from the webcam and analyze it using TensorFlow MobileNetV3."""
   # Initialize the webcam
   cap = cv2.VideoCapture(0)


   if not cap.isOpened():
       print("Error: Could not access the webcam.")
       return


   # Capture a single frame
   ret, frame = cap.read()


   if not ret:
       print("Error: Could not capture image.")
       cap.release()
       return


   # Release the webcam
   cap.release()


   # Load the MobileNetV3-Large model
   img = cv2.resize(frame, (224, 224))


   # Convert to array and preprocess for MobileNetV2
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)


   # Get the predictions from the model
   preds = modelMNV2.predict(x)


   # Print the decoded predictions
   decoded_preds = decode_predictions(preds, top=3)[0]
   print('Predicted:', decoded_preds)


   # Draw a bounding box and label on the original frame (not resized)
   label = decoded_preds[0][1]  # Take the first predicted label
   confidence = decoded_preds[0][2] * 100  # Confidence in percentage


   # Draw a bounding box around the entire frame to simulate object detection
   height, width, _ = frame.shape
   cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
   cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Information
   audio_stream = text_to_speech(f"Sir, the detected object is {label} with a {confidence:.2f}% confidence")
   play_audio(audio_stream)

   # Display the image with the drawn bounding box and label
   cv2.imshow("Item", frame)
   cv2.waitKey(0)
   cv2.destroyAllWindows()


def preprocess_image(image):
   """Convert OpenCV image (BGR) to PIL Image and apply transformations."""
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
   image = Image.fromarray(image)  # Convert to PIL Image
   transform = transforms.Compose([transforms.ToTensor()])  # Convert PIL Image to Tensor
   return transform(image).unsqueeze(0)  # Add batch dimension


def draw_bounding_boxes(image, outputs, threshold=0.5):
   """Draw bounding boxes and labels on the image."""
   boxes = outputs['boxes'].detach().cpu().numpy()
   scores = outputs['scores'].detach().cpu().numpy()
   labels = outputs['labels'].detach().cpu().numpy()

   label_names = []

   for i in range(len(boxes)):
       if scores[i] >= threshold:
           box = boxes[i].astype(int)
           label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
           label_names.append(label)
           cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
           cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Information
   if label_names:
       audio_stream = text_to_speech(f"Sir, {label_names} detected")
       play_audio(audio_stream)
   else:
       audio_stream = text_to_speech(f"Sir, nothing detected")
       play_audio(audio_stream)

   return image


def detect_objects_and_display():
   """Capture image, detect objects, and display results."""
   # Capture image from webcam
   # Initialize the webcam
   cap = cv2.VideoCapture(0)


   if not cap.isOpened():
       print("Error: Could not access the webcam.")
       return


   # Capture a single frame
   ret, frame = cap.read()


   if not ret:
       print("Error: Could not capture image.")
       cap.release()
       return


   # Release the webcam
   cap.release()


   # Preprocess image for the model
   input_tensor = preprocess_image(frame)


   # Perform object detection
   with torch.no_grad():
       outputs = model(input_tensor)[0]


   # Draw bounding boxes and labels
   image_with_boxes = draw_bounding_boxes(frame, outputs)


   # Display the image with detections
   cv2.imshow("Object Detection", image_with_boxes)

   cv2.waitKey(0)
   cv2.destroyAllWindows()


def ask_trivia():
    """Ask a random trivia question and check the user's response."""
    question = random.choice(trivia_questions)
    print(f"Trivia Question: {question['question']}")

    audio_file = text_to_speech(f"Here's a trivia question for you: {question['question']}")
    play_audio(audio_file)

    user_response = listen_to_microphone()
    if user_response:
        if user_response.lower() == question['answer'].lower():
            print("Correct!")
            audio_stream = text_to_speech("Correct!")
            play_audio(audio_stream)
        else:
            print(f"Incorrect. The correct answer was {question['answer']}.")
            audio_stream = text_to_speech(f"Incorrect. The correct answer was {question['answer']}.")
            play_audio(audio_stream)


def capture_and_recognize_faces(known_encodings):
    """Capture an image from the webcam and recognize faces in it."""
    print("Hello, my name is Bruce and before we get started I must make sure you are Aryan")
    audio_stream = text_to_speech("Hello, my name is Bruce and before we get started I must make sure you are Aryan")
    play_audio(audio_stream)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture image.")
        cap.release()
        return

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # List for faces
    faces_detected = []

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            faces_detected.append("Aryan")
        else:
            faces_detected.append("Unknown Face")
    return faces_detected
def capture_image_and_perform_ocr():
    # Initialize the webcam
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture image.")
        cap.release()
        return

    # Release the webcam
    cap.release()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Perform OCR on the captured image
    text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
    if text == "":
        print("No text detected")
        audio_stream = text_to_speech("No text detected")
        play_audio(audio_stream)
    else:
        print(text)
        audio_stream = text_to_speech(text)
        play_audio(audio_stream)

def main():
   """Main function to run the speech-to-text, OpenAI response, and text-to-speech."""
   global stop_speaking
   global long_length
   print("Hello, loading packages...")
   audio_stream = text_to_speech("Hello, loading packages...")
   play_audio(audio_stream)
   encodings = load_known_faces("known_faces")
   faces_detected = capture_and_recognize_faces(encodings)

   if not faces_detected:
       print("No face detected, ending program...")
       audio_stream = text_to_speech("No face detected, ending program...")
       play_audio(audio_stream)
   else:
       if "Aryan" in faces_detected:
           print("Welcome Aryan, how can I help?")
           audio = text_to_speech("Welcome Aryan, how can I help?")
           play_audio(audio)
           while True:
               stop_speaking = False
               long_length = False
               user_input = listen_to_microphone()
               if user_input:
                   ai_user_input = user_input.replace("Bruce", "")
                   if "Bruce" in user_input:
                       if "capture image" in user_input or "take a picture" in user_input:
                           print("Capturing image from webcam...")
                           capture_and_display_image()
                       elif "detect faces" in user_input:
                           print("Capturing and detecting faces...")
                           detect_faces_and_display()
                       elif "what is this" in user_input or "tell me what this is" in user_input or "analyze this" in user_input:
                           print("Capturing and finding out...")
                           capture_and_analyze_object()
                       elif "analyze everything" in user_input or "detect objects" in user_input:
                           print("Capturing and analyzing image...")
                           detect_objects_and_display()
                       elif "give me some trivia" in user_input or "trivia time" in user_input or "more trivia" in user_input:
                           ask_trivia()
                       elif "whats this say" in user_input or "what does this say" in user_input or "read this" in user_input:
                           print("Reading...")
                           capture_image_and_perform_ocr()
                       else:
                           response_text = get_openai_response(ai_user_input)
                           response_textv1 = response_text.replace("#", "")
                           new_response_text = response_textv1.replace("*", "")
                           print(f"OpenAI Response: {new_response_text}")
                           audio_stream = text_to_speech(new_response_text)
                           if len(new_response_text) > 75:
                               long_length = True
                           play_audio(audio_stream)
                           stop_speaking = False
                   else:
                       print("Keyword 'Bruce' not found. No response generated.")
       else:
           print("You are not Aryan, ending program...")
           audio_stream = text_to_speech("You are not Aryan, ending program...")
           play_audio(audio_stream)

if __name__ == "__main__":
   main()
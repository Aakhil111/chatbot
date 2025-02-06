import nltk
import random
import string  # To process standard Python strings
import numpy as np
import speech_recognition as sr
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speed of speech

# Function to make the chatbot speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech and convert to text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)  # Convert speech to text
            print(f"🗣️ You said: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("🤖 Sorry, I couldn't understand that.")
            return ""
        except sr.RequestError:
            print("🤖 Speech service unavailable.")
            return ""


# Chatbot responses
chatbot_responses = {
    "greetings": ["Hello!", "Hi there!", "Hey! How can I assist you?"],
    "goodbye": ["Goodbye!", "See you later!", "Take care!"],
    "thanks": ["You're welcome!", "No problem!", "Glad I could help!"],
    "name": ["I'm an A I chatbot!", "You can call me ChatBot."],
    "feeling": ["I'm doing great! How about you?", "I'm just a chatbot, but I'm feeling good!"],
    "functions": ["I can answer questions, tell jokes, and chat with you!", "I help answer questions using AI!"],
    "jokes": ["Why did the AI break up with the cloud? It needed more space!", "I'm on a seafood diet. I see food, and I eat it!"],
    "creator": ["I was created by an AI developer!", "A programmer built me using Python."],
    "python": ["Python is a programming language used for AI, web development, and more!", "Python is easy to learn and very powerful!"],
    "ai": ["AI stands for Artificial Intelligence. It helps machines think like humans!", "AI is used in chatbots, self-driving cars, and much more!"],
    "machine_learning": ["Machine learning is a branch of AI that helps computers learn from data!", "ML is used in speech recognition, image processing, and recommendation systems!"],
    "default": ["I'm sorry, I don't understand that.", "Can you rephrase your question?"]
}

# Training data
training_sentences = [
    "hello", "hi", "hey",
    "bye", "goodbye", "see you",
    "thank you", "thanks",
    "what is your name", "who are you",
    "how are you", "how are you doing",
    "what can you do", "what is your function",
    "tell me a joke", "say something funny",
    "who created you", "who made you",
    "what is python", "explain python",
    "how does AI work", "what is artificial intelligence",
    "what is machine learning",
]

training_responses = [
    "greetings", "greetings", "greetings",
    "goodbye", "goodbye", "goodbye",
    "thanks", "thanks",
    "name", "name",
    "feeling", "feeling",
    "functions", "functions",
    "jokes", "jokes",
    "creator", "creator",
    "python", "python",
    "ai", "ai",
    "machine_learning",
]

# Convert text into numerical vectors
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_sentences)

# Function to get the best response
def get_response(user_input):
    user_vector = vectorizer.transform([user_input])  # Convert user input into vector
    similarities = cosine_similarity(user_vector, X_train)  # Compute similarity
    best_match = np.argmax(similarities)  # Find the best match index

    if similarities[0][best_match] > 0.2:  # If confidence is high
        category = training_responses[best_match]
        return random.choice(chatbot_responses[category])
    else:
        return random.choice(chatbot_responses["default"])

# Chatbot loop with voice support
def start_chat():
    print("🎤 Chatbot is ready! Type or Speak. Say 'bye' to exit.")
    speak("Hello! I'm your AI chatbot. How can I assist you?")

    while True:
        # Ask user to type or speak
        user_input = input("Type your message or press 'V' to talk: ")

        # If user wants to speak
        if user_input.lower() == "v":
            user_input = recognize_speech()

        if user_input.lower() == "bye":
            speak("Goodbye! Have a nice day!")
            print("Goodbye!")
            break

        response = get_response(user_input)
        print(f"🤖 Chatbot: {response}")
        speak(response)  # Speak the response

# Run chatbot
start_chat()



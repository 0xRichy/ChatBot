import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class ChatBot:
    def __init__(self, intents_file):
        # Load the intents file
        with open(intents_file, 'r') as file:
            self.intents = json.load(file)

        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Initialize the bag of words converter
        self.vectorizer = CountVectorizer()

        # Initialize the classifier
        self.classifier = RandomForestClassifier()

    def train(self):
        # Preprocess the training data
        training_data = []
        training_labels = []
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                words = self.tokenize(pattern)
                training_data.append(' '.join(words))
                training_labels.append(intent['tag'])

        # Convert the training data to a bag of words
        X_train = self.vectorizer.fit_transform(training_data)

        # Train the classifier
        self.classifier.fit(X_train, training_labels)

    def chat(self):
        print("Start chatting with the bot (type quit to stop)!")
        while True:
            # Get user input
            inp = input("You: ")
            if inp.lower() == "quit":
                break

            # Preprocess the input
            words = self.tokenize(inp)
            X_pred = self.vectorizer.transform([' '.join(words)])

            # Predict the class
            pred_class = self.classifier.predict(X_pred)

            # Generate a response
            response = self.get_response(pred_class)
            print(f"Bot: {response}")

    def tokenize(self, text):
        # Tokenize the text
        words = word_tokenize(text)

        # Remove stop words
        words = [word for word in words if word not in stopwords.words('english')]

        # Lemmatize the words
        words = [self.lemmatizer.lemmatize(word) for word in words]

        return words

    def get_response(self, pred_class):
        # Find the intent with the predicted class
        for intent in self.intents['intents']:
            if intent['tag'] == pred_class:
                # Return a random response
                return np.random.choice(intent['responses'])

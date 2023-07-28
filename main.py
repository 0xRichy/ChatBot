from chatbot import ChatBot

def main():
    # Initialize the chatbot with a training set of data
    chatbot = ChatBot('intents.json')

    # Train the chatbot
    chatbot.train()

    # Start a chat session
    chatbot.chat()

if __name__ == "__main__":
    main()

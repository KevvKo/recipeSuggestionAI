import json
import sys
import random
import nltk
import numpy as np 
import pandas as pd
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model

from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

class MealDishAi():

    def __init__(self):

        self._intents = json.loads(open('src/recipes.json').read())
        self._words = pickle.load(open('src/words.pkl', 'rb'))
        self._classes = pickle.load(open('src/classes.pkl', 'rb'))
        self._documents = pickle.load(open('src/documents.pkl', 'rb'))


    #creating a reusable json-file with recipes and further properties
    def createRecipesFile(self):
        with open ('src/chefkoch.json', 'r') as file:
            data = json.load(file)
        
        intents = []
        
        # loop through the rawdata and put the ingredionts and instructions for every meal and 
        # create a single jsonline to append to the new decoded dataset
        for i in range(len(data)):
            
            # intercept the necessary informations
            jsline = {}
            
            mealName = data[i]['Name']
            preparation =  data[i]['Instructions']
            ingredients = data[i]['Ingredients']

            # the line with the informations for any meal
            jsline = {
                'preparation': preparation,
                'ingredients': ingredients
            }

            # the created intents-dataset
            intents.append(
                {mealName: jsline}
            )
        
        # create the decoded json with recipes
        with open('src/recipes.json', 'w') as file:
            json.dump({'intents': intents}, file, indent=4)


    #creating an unique alphabet with words, which are containted in the dataset and
    #further a sorted list of all classes
    def creatingSetsForAI(self):
        words = []
        classes = []
        documents = []
        ignoreWords = ['?', '!']

        with open('src/recipes.json', 'r') as file:
            dataFile = json.load(file)

        #loop through the dataset and creating all necessary lists
        for intent in dataFile['intents']:
            for recipe in intent:
                w = nltk.word_tokenize(recipe)
                words.extend(w)

                documents.append((w, recipe))

                if recipe not in classes: classes.append(recipe)

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignoreWords]

        #save the sorted list of unique words in a binary file (pickle) to archive them
        sorted(list(set(words)))
        pickle.dump(words,open('src/words.pkl','wb'))

        #save the classes as a sorted list in a pkl-file
        sorted(list(set(classes)))
        pickle.dump(classes, open('src/classes.pkl', 'wb'))

        #save the documents in a pickle-file
        pickle.dump(documents, open('src/documents.pkl', 'wb'))


    #creating bag of words
    def getWordbag(self, document, words):
            
            # initializing bag of words
            bag = []

            # list of tokenized words for the pattern
            patternWords = document[0]

            # lemmatize each word - create base word, in attempt to represent related words
            patternWords = [lemmatizer.lemmatize(word.lower()) for word in patternWords]

            # create our bag of words array with 1, if word match found in current pattern
            for w in words:
                bag.append(1) if w in patternWords else bag.append(0)

            return bag

    #building a corpus with word of bags
    def getCorpus(self):

        words = self._words
        classes = self._classes
        documents = self._documents
        
        #initialize the training data
        corpus = []
        emptyCorpus = [0] * len(self._classes)
    
        for doc in documents:
            bag = self.getWordbag(doc, words)
            
            # output is a '0' for each tag and '1' for current tag (for each pattern) .> 
            corpusRow = list(emptyCorpus)
            corpusRow[classes.index(doc[1])] = 1

            corpus.append([bag, corpusRow])
        return corpus

    #train the model and save the result
    def trainModel(self):

        corpus = self.getCorpus()

        #shuffling the features and turn into a numpy array
        random.shuffle(corpus)
        training = np.array(corpus)
        len(training)

        #create training and tests-lists -> x: pattern (recipe words), y: intent (entire recipe string)
        trainingX = list(training[:,0])
        trainingY = list(training[:,1])

        #create the model - 3 layers, first layer 128 neurons, second layer 64 neurons and 3rd output
        model = Sequential()
        #first layer - 128 neurons
        model.add(Dense(256, input_shape=(len(trainingX[0]),), activation='relu'))
        model.add(Dropout(0.5))
        #second layer - 64 neurons    
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        
        #third layer - output
        model.add(Dense(len(trainingY[0]), activation='softmax'))

        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
        model.summary()
        #fitting and saving the model
        hist = model.fit(np.array(trainingX), np.array(trainingY), epochs=200, batch_size=100, verbose=1) 
        model.save('chatbot_model.h5', hist)
    
    #make a prediction and return the computed match
    def prediction(self):
        pass


    def getResponse(self, message):
        sentence = self.cleanUpSentence(message)
        return sentence 


    def cleanUpSentence(self, message):
        return message
    

    #main-loop for the bot
    def run(self):

        print('Hello, my name is MJ.')
        print("Have you any questions?")
        
        #treated the input stream and gives a computed answers
        for inp in sys.stdin:
            print('Please enter a recipe')
            rsp = self.getResponse(inp)
            print(rsp[0])

            #keyword to kill the programm 
            if(rsp[1] == "adoption"): sys.exit()

###################################################################################################################

if __name__ == "__main__":

    bot = MealDishAi()
    bot.run()
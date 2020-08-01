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
        self.model = load_model('src/chatbot_model.h5')

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

    #creating a bag of words for the preciction 
    def bow(self, sentence):
        
        words = self._words

        # tokenize the pattern
        sentenceWords = self.cleanUpSentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)

        for s in sentenceWords:
            for i,w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1

        return(np.array(bag))


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
    def prediction(self, sentence, model):

        # filter out predictions below a threshold
        bow = self.bow(sentence)

        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self._classes[r[0]], "probability": str(r[1])})
  
        return return_list

    def getResponse(self, intents):
        
        recipe = intents[0]['intent']
        list_of_intents = self._intents['intents']
        
        for i in list_of_intents:
  
            if(list(i.keys())[0] == recipe):

                return i
               
        


    def cleanUpSentence(self, sentence):
        sentenceWwords = nltk.word_tokenize(sentence)
        sentenceWwords = [lemmatizer.lemmatize(word.lower()) for word in sentenceWwords]
        return sentenceWwords
    

    def botRepsonse(self, message):
        intents = self.prediction(message, self.model)   
        response = self.getResponse(intents)

        return response


    #main-loop for the bot
    def run(self):
        print('Please enter a recipe')

        #treated the input stream and gives a computed answers
        for inp in sys.stdin:

            response = self.botRepsonse(inp)
            print(response)

            print('Please enter a recipe:')
###################################################################################################################

if __name__ == "__main__":

    bot = MealDishAi()
    bot.run()
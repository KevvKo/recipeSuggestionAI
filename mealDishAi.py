from network import Network
import json
import sys
import random
import nltk
import numpy as np 
import pickle

from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from forwardLayer import ForwardLayer
from activationFunctions import tanh, tanh_prime
from losses import mse, mse_prime 
from activationLayer import ActivationLayer

class MealDishAi(Network):

    def __init__(self):

        self._net = Network()
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

                documents.append((recipe, w))

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
            patternWords = document[1]

            # lemmatize each word - create base word, in attempt to represent related words
            patternWords = [lemmatizer.lemmatize(word.lower()) for word in patternWords]

            # create our bag of words array with 1, if word match found in current pattern
            for w in words:
                bag.append(1) if w in patternWords else bag.append(0)

            return np.array(bag)


    #train the model and save the result
    def trainModel(self):

        words = self._words
        classes = self._classes
        documents = self._documents
        
        #initialize the training data
        training = []
        outputEmpty = [0] * len(self._classes)
    
        for doc in documents:
            bag = self.getWordbag(doc, words)
            
            # output is a '0' for each tag and '1' for current tag (for each pattern) .> 
            outputRow = list(outputEmpty)
        
            outputRow[classes.index(doc[0])] = 1
            outputRow = np.array(outputRow)

            training.append([outputRow, bag])

        #shuffling the features and turn into a numpy array
        random.shuffle(training)
        training = np.array(training)

        #create training and tests-lists -> x: pattern (recipe words), y: intent (entire recipe string)
        trainingX = list(training[:,1])
        trainingY = list(training[:,0])

        self._net.addLayer(ForwardLayer(1,len(trainingX)))
        self._net.addLayer(ActivationLayer(tanh, tanh_prime))
        self._net.addLayer(ForwardLayer(len(trainingX), len(trainingX)*2))
        self._net.addLayer(ActivationLayer(tanh, tanh_prime))
        self._net.addLayer(ForwardLayer(len(trainingX)*2, 1))
        self._net.addLayer(ActivationLayer(tanh, tanh_prime))

        #self._net.load('modelState.json')

        # train
        self._net.useLoss(mse, mse_prime)
        self._net.fit(np.array([trainingX]), np.array([trainingY]), epochs=1000, learning_rate=0.1)


    #make a prediction and return the computed match
    def prediction(self):
        pass


    def getResponse(self):
        pass


    def cleanUpSentence(self):
        pass
    

    def run(self):
        pass

###################################################################################################################

if __name__ == "__main__":

    bot = MealDishAi()
    bot.loadJson('src/recipes.json')
    #bot.creatingSetsForAI()
    bot.trainModel()
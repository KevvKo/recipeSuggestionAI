# recipeSuggestionAI

##Description
The recipeSuggestionAI is a simple AI, feeded and trained with a dataset over 13.000 recipes. There is for every recipe a list of ingredients and the belonging preparation. So, the AI response with a recipe, they can found in the dataset. The response including the belonging preparation and the list of ingredients. The 

##Requirements
- nltk
- numpy
- pickle
- pandas
- keras

##Getting started
Just declare a new instance of **mealDishAi** and execute the **run()**-function:
```python
mealAI = mealDishAi()
mealAI.run()
```
##neural-network-architecture
The neural network is build with the open-source framework keras. 
I used a multilayer perceptron with 3 layers:
- first layer:  256 neurons 
- second layer: 128 neurons 
- third layer:  output. 

As optimizer, adam is used and for the activation-functions, I use **relu**.
This works fine for simple results, but there is a lot of scope for improvements.
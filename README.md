# Womens_Ecommerce
Sentiment Analysis of Women's Ecommerce Clothing Reviews

This program uses a Naive Bayes classifier to perform sentiment analysis on women's ecommerce clothing reviews. The program reads a CSV file containing the reviews and corresponding ratings, and then creates a binary sentiment column based on the ratings.

The program then drops any rows with missing values and removes columns that are not required for sentiment analysis. The text of the reviews is transformed into a matrix of token counts using a CountVectorizer. The data is then split into training and test sets, and a Naive Bayes classifier is trained using the training data.

The program then makes predictions on the test set and calculates the accuracy of the classifier. Finally, the program predicts the likelihood of good reviews for clothing items based on user input.

To run the program, ensure that the required packages are installed, and then execute the Python script. The program will read the data from the CSV file and output the accuracy of the classifier and the predicted likelihood of good reviews for clothing items.

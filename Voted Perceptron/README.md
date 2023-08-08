# Voted Perceptron

Perceptron algorithm is one of the classic algorithms which has been used in machine learning from early 1960s. It is an online learning algorithm for learning a linear threshold function which works as a classifier. We will also apply the Voted Perceptron algorithm on the classification task in this project.

Let D = {(xi, ti)}<sup>m</sup> i = 1 be the training data, let xi be the feature vectors and ti ∈ {−1, +1} be the corresponding labels. The goal of the Perceptron algorithm is to find a vector w which defines a linear function such that ∀i, ti(w ⋅ xi) > 0. For the Voted Perceptron, you need a list of weighted vectors {(wk , ck)}<sup>K</sup> k=1 where Wk is the vector and Ck is its weight.

For this project, download the training data from https://www.dropbox.com/s/gqye1fydkdg8ig4/hw3dataq1.zip?dl=0. It contains two CSV files:

Xtrain.csv Each row is a feature vector. The values in the i-th columns are integer values in the i-th dimension.

Ytrain.csv The CSV file provides the binary labels for corresponding feature vectors in the file Xtrain.csv.

I was able to achieve an accuracy of 98.5196078431%
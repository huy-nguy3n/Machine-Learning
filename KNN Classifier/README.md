# KNN Classifier

For this problem, you can download the training data from https://www.dropbox.com/s/nzbu1km1010hku7/hw3dataq2.zip?dl=0. The training data contains two CSV files:

Xtrain.csv Each row is a feature vector. The values in the i-th columns are float numbers in the i-th dimension.

Ytrain.csv The CSV file provides the multi-class labels for corresponding feature vectors in the file Xtrain.csv . Please note the labels will be integer numbers between 0 and 10.

The program use a vote among the k nearest neighbors to determine the output label of a test point; in the case of a tie vote, choose the label of the closest neighbor among the tied exemplars. In the case of a distance tie (e.g., the two nearest neighbors are at the same distance but have two different labels), choose the lowest-numbered label (e.g., choose label 3 over label 7). Euclidian distance was used to determine distance/nearness in this project.

I was able to achieve an accuracy of 100.0%
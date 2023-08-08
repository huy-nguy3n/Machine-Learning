import numpy as np

def discriminant(data_point, class_0_centroid, class_1_centroid, class_2_centroid): 
    # For each centroid, calculate the distance between it and the data point. 
    c0 = np.linalg.norm(class_0_centroid - data_point)
    c1 = np.linalg.norm(class_1_centroid - data_point)
    c2 = np.linalg.norm(class_2_centroid - data_point)
    
    # If the distance between the class 0 centroid and the data point is the minimum, return 0
    if (c0 < c1) and (c0 < c2):
        return 0
    # Else, if the distance between the class 1 centroid and the data point is less than the distance between the class 2 centroid and the data point, return 1
    elif (c1 < c2):
        return 1
    # Else, return 2
    else: 
        return 2
 
def run(train_input_dir, train_label_dir, test_input_dir, pclass_0_file):
    # Using np.loadtxt, load the data from train_input_dir and train_label_dir. 
    train_data = np.loadtxt(train_input_dir,skiprows=0)
    train_labels = np.loadtxt(train_label_dir,skiprows=0)

    # Get the number of features by taking the length of the first element in train_data. - code to 3
    # Split the train_data into three smaller objects by sorting based on the train labels. 
    # Calculate the centroids based on the three-dimensional coordinates of each class. 
    # Use the discriminant function to determine the prediction for each coordinate in the testing data, by calculating the distance between the point and the three centroids.     
    class0 = []
    class1 = []
    class2 = []
    
    # Split the train_data into three smaller objects by sorting based on the train labels. 
    for i in range(len(train_data)):
        if train_labels[i] == 0:
            class0.append(train_data[i])
        elif train_labels[i] == 1:
            class1.append(train_data[i])
        else:
            class2.append(train_data[i])

    # Calculate the centroids based on the three-dimensional coordinates of each class. 
    class_0_centroid = np.array(class0, dtype=object)
    class_1_centroid = np.array(class1, dtype=object)
    class_2_centroid = np.array(class2, dtype=object)


    # Reading data
    test_data = np.loadtxt(test_input_dir,skiprows=0)
    prediction = np.array([discriminant(data, class_0_centroid, class_1_centroid, class_2_centroid) for data in test_data], dtype=object)

    # Saving you prediction to pclass_0_file directory (Saving can't be changed)
    np.savetxt(pclass_0_file, prediction, fmt='%1d', delimiter=",")
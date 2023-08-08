import numpy as np

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num


def run(Xtrain_file, Ytrain_file, test_data_file, pred_file): 
    train_data = np.loadtxt(Xtrain_file, delimiter=",", dtype=float)
    train_labels = np.loadtxt(Ytrain_file, dtype=float)
    test_data = np.loadtxt(test_data_file, delimiter=",", dtype=float)

    distances = np.zeros((len(train_data), len(test_data)))
    d_labels = np.zeros_like(distances)

    for ind_test,test in enumerate(test_data): 
        for ind_train,train in enumerate(train_data): 
            # Set d to be the euclidean distance between test and train
            d = np.linalg.norm(test - train)
            # Set the corresponding value in the two-dimensional array distances equal to d, using numpy two dimensional notation
            distances [ind_train] [ind_test] = d
            # Set the corresponding label in d_labels as the corresponding value in train_labels
            d_labels [ind_train] [ind_test] = train_labels [ind_train]
            pass

    k = 10
    predictions = []

    for ind_test,test in enumerate(test_data): 
        # Using two dimensional numpy notation...
        # set dists equal to the ind_test column of distances, and labels equal to the ind_test column in d_labels
        dists = distances[:, ind_test]
        labels = d_labels[:, ind_test]

        # Sort dists and labels based on the corresponding distance, from least distance to greatest distance. 
        dists,labels = zip(*sorted(zip(dists, labels)))

        # Grab the k lowest values in dists
        low_val = []
        for i in range(k):
            low_val.append(labels[i])
        
        # Append the most frequent value to predictions.
        predictions.append(most_frequent(low_val))

        
    np.savetxt(pred_file, predictions, fmt='%1d', delimiter=",")

if __name__ == "__main__": 
    train_input_dir = 'train_data.csv'
    train_label_dir = 'train_labels.csv'
    test_input_dir = 'test_data.csv'
    pred_file = 'predictions.txt'
    run(train_input_dir, train_label_dir, test_input_dir, pred_file)
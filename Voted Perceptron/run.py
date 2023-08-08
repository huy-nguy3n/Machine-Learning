import numpy as np

from datetime import datetime
start_time = datetime.now()

def predict(weights, instance): 
    # weights needs to be a list of two-element tuples, where the first element is w_vector, and the second is c_labels
    # Initialize label to be an empty array
    label = []
    # For each weight, append the product of the weight label times the sign of the dot product of the corresponding weight vector and the instance

    for i in range(len(weights)):
        label.append(weights[i][1] * np.sign(np.dot(weights[i][0], instance)))

    # If the sign of the sum of this label is 1, return 1
    if np.sign(sum(label)) == 1:
        return 1
    # Else, return 0
    else:
        return 0
    
def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    train_data = np.loadtxt(Xtrain_file, delimiter=",", dtype=float)
    train_labels = np.loadtxt(Ytrain_file, dtype=float)

    # Correct the train labels by changing all the 0s to -1s
    labels = []
    for l in train_labels: 
        if l == 1: labels.append(1)
        else: labels.append(-1)
    train_labels = np.array(labels, dtype=object)
 
    # Initialize all parameters
    epochs = 1
    k = 0
    w_vector = np.zeros((len(train_data), len(train_data[0])))
    c_labels = np.zeros(len(train_labels))
    t=0

    while t < 1:
        for i in range(len(train_data)):
            dot = np.dot(w_vector[k], train_data[i])
            yHat = np.sign(dot)

            if not (yHat == train_labels[i] or t == 1):
                w_vector[k + 1] = w_vector[k] + (train_labels[i] * train_data[i])
                c_labels[k + 1] = 1
                k = k + 1 
            else:
                c_labels[k] += 1

        t = t + 1

    perceptrons = np.array([[v_i, c_i] for v_i,c_i in zip(w_vector,c_labels)], dtype=object)

    test_data = np.loadtxt(test_data_file, delimiter=",")
    predictions = [predict(perceptrons, data) for data in test_data]
    np.savetxt(pred_file, predictions, fmt = '%1d', delimiter = ",")

if __name__ == "__main__": 
    numbers = [1, 2, 5, 10, 20, 100]
    for n in numbers: 
        train_input_dir = 'data/train_data_%i.csv' % n 
        train_label_dir = 'data/train_labels_%i.csv' % n
        test_input_dir = 'test_data.csv'
        test_label_dir = 'test_labels.csv'
        pred_file = 'predictions_%i.txt' % n
        run(train_input_dir, train_label_dir, test_input_dir, pred_file)
        
        predicted = np.loadtxt(pred_file, skiprows=0)
        actual = np.loadtxt(test_label_dir, skiprows=0)

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for a,p in zip(actual, predicted): 
            if a == 1 and p == 1: tp += 1
            elif a == 1 and p == 0: fn += 1
            elif a == 0 and p == 1: fp += 1
            else: tn += 1
        
        accuracy = round(100 * (tp + tn) / (tp + fp + tn + fn), 4)
        f1_score = round(100 * (2 * tp) / (tp + fn + tp + fp),4)

        print("---- Dataset %i ----" % n)
        print("Accuracy: %s" % accuracy)
        print("F1 score: %s" % f1_score)
        print("TP : %i ; FP : %i" % (tp, fp))
        print("TN : %i ; FN : %i" % (tn, fn))
        print('')

    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))
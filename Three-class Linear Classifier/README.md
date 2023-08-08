# Three-class Linear Classifier

In this project, given training data was used to train a three-class linear classifier. A Python program named run.py implements a run(train_input_dir, train_label_dir, test_input_dir, pred_file) function: 

Parameters:
- train_input_dir: str - the path to the training dataset txt file.
- train_label_dir - the path to training dataset label txt file.
- test_input_dir: str - the path to the test dataset txt file.
- pred_file: str - the file name of your output prediction file.

The function should execute the following steps:
- 1. Load the training data examples located at train_dir.
- 2. Train a three-class linear classifier on the training data.
- 3. Load the test data from `test_dir` and make predictions using the classifier you
trained. The prediction should be a numpy array with shape (N, 1).
- 4. Write the predictions into a single file named `pred_file`. Hint: you may use the function numpy.savetxt(pred_file, prediction, fmt='%1d', delimiter=",") where pred_file is the file name of your prediction, prediction is the numpy array of your prediction.
You don’t need to know what the values of train_input_dir, train_label_dir, test_input_dir and pred_file will be. We will automatically import these variables on the Codalab platform. 

https://competitions.codalab.org/competitions/35876?secret_key=deb3fa0a-039f-4b6f-a6 36-ea18b11591c8

I was able to achieve an accuracy of 88.3333333333%.
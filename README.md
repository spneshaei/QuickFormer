# QuickFormer - SimpleTransformers, but even simpler

Quickly train and evaluate a Transformer classification model on a given dataset.

QuickFormer allows you to:

- Produce and evaluate a Transformer classification model by only one line of code
- Evaluate the model and find precision, recall, and F1 score for all classes
- Generate confusion matrices for the model

## How to run

1. Prepare your initial data in a CSV file with the columns `text` and `cat_label`. Name the file `model_name_input.csv` where `model_name` is an arbitrary name.

2. Call the `quickform()` function. The only mandatory argument is the model name (`model_name`). The function will automatically load the data from the CSV file, train the model, evaluate it and saves all the results in files and directories starting with `model_name`.

That's it!

## Automatically generated files after each run

1. `model_name_train_data.csv` - the data used for training
2. `model_name_test_data.csv` - the data used for testing
3. `model_name_confusion_matrix.png` - a visualization of the confusion matrix
3. `model_name_confusion_matrix_normalized.png` - a visualization of the confusion matrix with normalized values
4. `model_name_confusion_matrix.txt` - the raw data of the confusion matrix
5. `model_name_precision_recall_f1.txt` - the data for precision, recall, and F1 score for each of the classes
6. `model_name_codes.txt` - relation between the codes and the different classes in the dataset
7. `outputs_model_name` - saved model, which can be later reused for inference

Enjoy using QuickFormer!
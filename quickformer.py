from distutils.command.clean import clean
import os
import pandas as pd
import matplotlib.pyplot as plt
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, jaccard_score, hamming_loss, zero_one_loss
import numpy as np
import itertools
import numpy as np

# https://stackoverflow.com/a/50386871/4915882 with modifications
def save_confusion_matrix(cm,
                          target_names,
                          model_name,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize=False):

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('BuGn')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if normalize:
        plt.title(title + "for model '" + model_name + "' (Normalized)")
    else:
        plt.title(title + "for model '" + model_name + "'")
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if normalize:
        plt.savefig(model_name + "_confusion_matrix_normalized.png")
    else:
        plt.savefig(model_name + "_confusion_matrix.png")

def get_df_from_csv(csv_file_address, model_name):
    csv_file = csv_file_address
    if csv_file is None:
        csv_file = model_name + "_input.csv"
    df = pd.read_csv(csv_file)
    return df

def shuffled_df(df, random_state):
    new_df = None
    if random_state == None:
        new_df = df.sample(frac = 1)
    else:
        new_df = df.sample(frac = 1, random_state = random_state)
    return new_df

def categorize_df(df, model_name):
    df.cat_label = pd.Categorical(df.cat_label)
    classified_categories_raw = df.cat_label.cat.categories
    classified_categories = []
    codes_str = ""
    for i in range(len(classified_categories_raw)):
        classified_categories.append(str(classified_categories_raw[i]))
        codes_str += str(i) + " => " + str(classified_categories_raw[i]) + "\n"
    with open(model_name + '_codes.txt', 'w') as f:
        f.write(codes_str)
    df['label'] = df.cat_label.cat.codes
    df = df.drop(df.columns[[1]], axis = 1)
    return df, classified_categories

def split_train_test(df, train_percentage, model_name):
    last_index_of_training = int(len(df) * train_percentage)
    train_df = df[:last_index_of_training]
    test_df = df[last_index_of_training:]
    train_df.to_csv(model_name + '_train_data.csv', index = False)
    test_df.to_csv(model_name + '_test_data.csv', index = False)
    return train_df, test_df

def predict_model(model, test_df):
    all_texts = []
    for index, row in test_df.iterrows():
        all_texts.append(row["text"])
    predictions, _ = model.predict(list(all_texts))
    return predictions

def find_and_save_confusion_matrix(test_df, predictions, classified_categories, model_name):
    real = []
    for _, row in test_df.iterrows():
        real.append(row["label"])
    matrix = confusion_matrix(real, predictions)
    save_confusion_matrix(matrix, classified_categories, model_name, normalize=False)
    save_confusion_matrix(matrix, classified_categories, model_name, normalize=True)
    matrix_string = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in matrix) # https://stackoverflow.com/a/34349901/4915882
    with open(model_name + '_confusion_matrix.txt', 'w') as f:
        f.write(matrix_string)
    return matrix, real

def find_and_save_precision_recall_f1(num_labels, matrix, classified_categories, model_name, real, predictions):
    precision_recall_f1_string = ""
    for i in range(num_labels):
        sum_m_ji = 0
        sum_m_ij = 0
        for j in range(num_labels):
            sum_m_ji += matrix[j][i]
            sum_m_ij += matrix[i][j]
        precision = matrix[i][i] / sum_m_ji
        recall = matrix[i][i] / sum_m_ij
        f1 = (2 * precision * recall) / (precision + recall)
        precision_recall_f1_string += "Precision for class " + str(i) + " corresponding to " + str(classified_categories[i]) + " =>\t" + str(precision) + "\n" + "Recall for class " + str(i) + " corresponding to " + str(classified_categories[i]) + " =>\t" + str(recall) + "\n" + "F1 for class " + str(i) + " corresponding to " + str(classified_categories[i]) + " =>\t" + str(f1) + "\n"
    precision_recall_f1_string += "\nAccuracy of the model =>\t" + str(accuracy_score(real, predictions)) + "\n"
    precision_recall_f1_string += "Balanced Accuracy of the model =>\t" + str(balanced_accuracy_score(real, predictions)) + "\n"
    if len(classified_categories) == 2:
        precision_recall_f1_string += "Jaccard Similarity Coefficient of the model =>\t" + str(jaccard_score(real, predictions)) + "\n"
    else:
        precision_recall_f1_string += "Jaccard Similarity Coefficient (average = 'micro') of the model =>\t" + str(jaccard_score(real, predictions, average='micro')) + "\n"
    precision_recall_f1_string += "Hamming Loss of the model =>\t" + str(hamming_loss(real, predictions)) + "\n"
    precision_recall_f1_string += "Zero-one Loss of the model =>\t" + str(zero_one_loss(real, predictions)) + "\n"
    with open(model_name + '_precision_recall_f1.txt', 'w') as f:
        f.write(precision_recall_f1_string)

def evaluate_model(model, model_name, classified_categories, test_df, num_labels):
    predictions = predict_model(model, test_df)
    matrix, real = find_and_save_confusion_matrix(test_df, predictions, classified_categories, model_name)
    find_and_save_precision_recall_f1(num_labels, matrix, classified_categories, model_name, real, predictions)

def cleanup_directory_names(model_name):
    os.rename('outputs',  'outputs_' + model_name)
    os.rename('cache_dir',  'cache_dir_' + model_name)
    os.rename('runs',  'runs_' + model_name)

def quickform(model_name, model_type = "bert", model_huggingface_hub_name = "bert-base-german-cased", csv_file = None, min_sentence_length = 5, random_state = 1, train_percentage = 0.8, use_cuda = False):
    df = get_df_from_csv(csv_file, model_name)
    df = df[df['text'].str.len() >= min_sentence_length]
    df.columns = ['text', 'cat_label']
    df = shuffled_df(df, random_state)
    df, classified_categories = categorize_df(df, model_name)
    train_df, test_df = split_train_test(df, train_percentage, model_name)
    print("QuickFormer - Preparing the model...")
    num_labels = max(train_df.label) + 1
    model = ClassificationModel(model_type, model_huggingface_hub_name, num_labels = num_labels, use_cuda = use_cuda)
    print("QuickFormer - Training the model...")
    model.train_model(train_df, overwrite_output_dir = True)
    print("QuickFormer - Evaluating the model...")
    evaluate_model(model, model_name, classified_categories, test_df, num_labels)
    cleanup_directory_names(model_name)
    print("Thank you for using QuickFormer!")


if __name__ == "__main__":
    quickform("test")
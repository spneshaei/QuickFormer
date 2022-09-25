import os
import pandas as pd
import matplotlib.pyplot as plt
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

def quickform(model_name, model_type = "bert", model_huggingface_hub_name = "bert-base-german-cased", csv_file_address = None, min_sentence_length = 5, random_state = 1, train_percentage = 0.8, use_cuda = False):
    csv_file = csv_file_address
    if csv_file is None:
        csv_file = model_name + "_input.csv"
    df = pd.read_csv(csv_file)
    df = df[df['text'].str.len() >= min_sentence_length]
    df.columns = ['text', 'cat_label']
    if random_state == None:
        df = df.sample(frac=1)
    else:
        df = df.sample(frac=1, random_state=random_state)
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
    last_index_of_training = int(len(df) * train_percentage)
    train_df = df[:last_index_of_training]
    test_df = df[last_index_of_training:]
    train_df.to_csv(model_name + '_train_data.csv', index = False)
    test_df.to_csv(model_name + '_test_data.csv', index = False)
    print("QuickFormer - Preparing the model...")
    num_labels = max(train_df.label) + 1
    model = ClassificationModel(model_type, model_huggingface_hub_name, num_labels = num_labels, use_cuda = use_cuda)
    print("QuickFormer - Training the model...")
    model.train_model(train_df)
    print("QuickFormer - Evaluating the model...")
    all_texts = []
    for index, row in test_df.iterrows():
        all_texts.append(row["text"])
    predictions, raw_outputs = model.predict(list(all_texts))
    os.rename('outputs',  'outputs_' + model_name)
    all_result_pairs = [] # [predicted, real]
    real = []
    i = 0
    for index, row in test_df.iterrows():
        all_result_pairs.append([predictions[i], row["label"]])
        real.append(row["label"])
        i += 1
    matrix = confusion_matrix(real, predictions)
    save_confusion_matrix(matrix, classified_categories, model_name, normalize=False)
    save_confusion_matrix(matrix, classified_categories, model_name, normalize=True)
    matrix_string = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in matrix) # https://stackoverflow.com/a/34349901/4915882
    with open(model_name + '_confusion_matrix.txt', 'w') as f:
        f.write(matrix_string)
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
        precision_recall_f1_string += "Precision for class", i, "corresponding to", classified_categories[i], "=>\t", precision + "\n"
        precision_recall_f1_string += "Recall for class", i, "corresponding to", classified_categories[i], "=>\t", recall + "\n"
        precision_recall_f1_string += "F1 score for class", i, "corresponding to", classified_categories[i], "=>\t", f1 + "\n"
    with open(model_name + '_precision_recall_f1.txt', 'w') as f:
        f.write(precision_recall_f1_string)

    print("Thank you for using QuickFormer!")



import os
import pandas as pd
import matplotlib.pyplot as plt
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    disp = ConfusionMatrixDisplay(confusion_matrix = matrix)
    disp.plot()
    plt.savefig(model_name + '_confusion_matrix.png')
    plt.close()
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
        precision_recall_f1_string += "Precision for class", i, "=>", precision + "\n"
        precision_recall_f1_string += "Recall for class", i, "=>", recall + "\n"
        precision_recall_f1_string += "F1 for class", i, "=>", f1 + "\n"
    with open(model_name + '_precision_recall_f1.txt', 'w') as f:
        f.write(precision_recall_f1_string)

    print("Thank you for using QuickFormer!")



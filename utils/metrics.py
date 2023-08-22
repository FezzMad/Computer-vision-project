from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from numpy import nan


def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=nan)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=nan)
    return accuracy, precision, recall

    # print(acc_val/len(train_loader))
    # conf_matr = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matr, display_labels=dataset.classes)
    # disp.plot()
    # plt.show()


def accuracy(y_true, y_pred):
    return (y_pred == y_true).mean()

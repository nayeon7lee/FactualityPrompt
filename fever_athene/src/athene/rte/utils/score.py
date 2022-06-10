from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score, f1_score


def print_metrics(gold_labels, predictions, logger=None, average='macro'):
    info = logger.info if logger is not None else print
    info("Accuracy: " + str(accuracy_score(gold_labels, predictions)) + "\tRecall: " + str(
        recall_score(gold_labels, predictions, average=average)) + "\tPrecision: " + str(
        precision_score(gold_labels, predictions, average=average)) + "\tF1 " + average + ": " + str(
        f1_score(gold_labels, predictions, average=average)))
    info("Confusion Matrix:")
    info(confusion_matrix(gold_labels, predictions))

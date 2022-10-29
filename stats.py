def to_confusion_matrix(values):
    outFileName = f"results_{file_path}_{number_hidden_neurons}n_{learning_rate}r_{threshold}t_{training_set_percent}p_{randomSeed}.csv"
    # Writing Seciton
    outputFile = open("NN-results/" + outFileName, 'w')

    writer = csv.writer(outputFile)
    # Write Labels Row
    writer.writerow(possible_labels)

    confusion_m_result = []
    # Write Columns Row
    for label in possible_labels:
        confusion_row = []
        labelCounter = Counter(label_results_dict[label])
        for compareLabel in possible_labels:
            confusion_row.append(labelCounter[compareLabel])
        confusion_row.append(label)
        confusion_m_result.append(confusion_row)
        writer.writerow(confusion_row)
    return confusion_m_result

"""
    Print out all the stats from the confusion matrix such as accuracy, recall for each label, and their confidence interval
"""
def calculateStats(matrix):
    # calculate accuracy
    sum_diagnol = 0
    sum_of_cells = 0
    for x in range(len(matrix)):
        for y in range(len(matrix)):
            value = matrix[x][y]
            if x == y:
                sum_diagnol += matrix[x][y]
            sum_of_cells += matrix[x][y]

    accuracy = sum_diagnol / sum_of_cells
    print(f"Accuracy: {accuracy}")
    # calculate recall
    for recall_x in range(len(matrix)):
        sum_of_row_y = 0
        for recall_y in range(len(matrix)):
            if recall_y == recall_x:
                cellyy = matrix[recall_x][recall_y]
            sum_of_row_y +=  matrix[recall_y][recall_x]
        # Error catching for zero division errr
        if sum_of_row_y != 0:
            recall = cellyy / sum_of_row_y
        else:
            recall = 0
        print(f"Recall for {matrix[recall_x][-1]}: {recall}")

    print()
    # Calculate the confidence interval
    confidence_interval_positive = accuracy + (1.96 * math.sqrt((accuracy * (1-accuracy))/(len(testingSet))))
    confidence_interval_negative = accuracy - (1.96 * math.sqrt((accuracy * (1-accuracy))/(len(testingSet))))
    print(f"Confidence Interval: [{confidence_interval_negative}, {confidence_interval_positive}]")

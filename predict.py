from collections import defaultdict


def count_correct_predictions(file_path):
    correct_predictions = 0
    total_labels = 0

    with open(file_path, "r") as file:
        # Skip the header
        next(file)

        for line in file:
            y_pred, y_correct = map(int, line.strip().split("\t"))

            if y_pred == y_correct:
                correct_predictions += 1

            total_labels += 1

    return correct_predictions, total_labels


def count_correct_predictions2(file_path):
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    wrong_counts = defaultdict(int)

    with open(file_path, "r") as file:
        # Skip the header
        next(file)

        for line in file:
            y_pred, y_correct = map(int, line.strip().split("\t"))

            total_counts[y_correct] += 1

            if y_pred == y_correct:
                correct_counts[y_correct] += 1
            else:
                wrong_counts[y_pred] += 1

    return correct_counts, total_counts, wrong_counts


if __name__ == "__main__":
    file_path = "output.csv"
    correct_predictions, total_labels = count_correct_predictions(file_path)

    print(f"Total correct predictions: {correct_predictions}")
    print(f"Total labels: {total_labels}")
    print(f"Accuracy: {correct_predictions / total_labels * 100:.2f}%")

    correct_counts, total_counts, wrong_counts = count_correct_predictions2(file_path)

    for correct_label, correct_count in correct_counts.items():
        total_count = total_counts[correct_label]
        print(
            f"Correct label {correct_label}: {correct_count} out of {total_count} correctly classified. WRONG {wrong_counts[correct_label]}"
        )

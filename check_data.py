# check_data.py

import os
from pathlib import Path
import matplotlib.pyplot as plt


def count_images_per_class(base_dir):
    class_counts = {}
    base_dir = Path(base_dir)

    for class_dir in base_dir.iterdir():
        if class_dir.is_dir():
            num_images = len(list(class_dir.glob("*.jpeg")))
            class_counts[class_dir.name] = num_images

    return class_counts


def plot_distribution(counts_dict, title):
    labels = list(counts_dict.keys())
    values = list(counts_dict.values())

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def analyze_balance(data_root="data"):
    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_root, split)
        print(f"\n Analiza zbioru: {split}")
        class_counts = count_images_per_class(split_path)
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")
        plot_distribution(class_counts, f"Rozkład klas w {split}")


if __name__ == "__main__":
    analyze_balance()  # Utworzenie wykresów pokazujących rozłożenia się klas w zbiorach

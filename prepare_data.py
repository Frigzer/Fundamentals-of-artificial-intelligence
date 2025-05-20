# prepare_data.py

import os
import shutil
import kagglehub
from pathlib import Path
from PIL import Image
import random
from torchvision import transforms


def download_and_prepare_data(data_dir="data"):
    """Pobiera dane z repozytorium american-sign-language-dataset"""
    dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    print(f"Pobrano dane do cache: {dataset_path}")

    # Skopiuj wszystko z cache do ./data
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Znajdź folder ASL_Gestures_36_Classes w pobranych danych
    source_root = Path(dataset_path)
    asl_folder = None
    for item in source_root.iterdir():
        if item.is_dir() and "asl_alphabet_test" in item.name:
            asl_folder = item
            break

    if not asl_folder:
        raise RuntimeError("Nie znaleziono folderu ASL_Gestures_36_Classes w danych Kaggle!")

    # Przenieś zawartość tego folderu do ./data
    for item in asl_folder.iterdir():
        dest = Path(data_dir) / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"Skopiowano dane do: {data_dir}")


def merge_to_temp(data_dir="data", temp_dir="temp"):
    """Scala dane z folderów train i test do folderu temp/"""
    train_dir = Path(data_dir) / "train"
    test_dir = Path(data_dir) / "test"

    os.makedirs(temp_dir, exist_ok=True)

    for split_dir in [train_dir, test_dir]:
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                dest_class_dir = Path(temp_dir) / class_name
                dest_class_dir.mkdir(parents=True, exist_ok=True)

                for img_file in class_dir.iterdir():
                    if img_file.is_file():
                        dest_file = dest_class_dir / img_file.name
                        if dest_file.exists():
                            dest_file = dest_class_dir / f"{img_file.stem}_from_{split_dir.name}{img_file.suffix}"
                        shutil.copy(img_file, dest_file)

    print(f"Dane scalone do folderu: {temp_dir}/")

    # Usuń stare foldery train/test
    for old_split in ["train", "test"]:
        old_path = Path(data_dir) / old_split
        if old_path.exists():
            shutil.rmtree(old_path)


def augment_dataset(temp_dir="temp", target_count_per_class=140):
    """Augmentuje dane zwiększając ich ilość"""
    temp_dir = Path(temp_dir)

    # Przykładowe augmentacje (można rozszerzyć)
    augment_ops = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ])

    for class_dir in temp_dir.iterdir():
        if not class_dir.is_dir():
            continue

        image_files = list(class_dir.glob("*.jpeg"))
        current_count = len(image_files)
        needed = target_count_per_class - current_count

        if needed <= 0:
            continue  # ta klasa już ma wystarczająco

        print(f"Augmentuję klasę '{class_dir.name}': {current_count} -> {target_count_per_class}")

        i = 0
        while i < needed:
            # losowy obraz z oryginałów
            src_img_path = random.choice(image_files)
            with Image.open(src_img_path) as img:
                img = img.convert("RGB")  # na wszelki wypadek

                # wykonaj augmentację
                augmented = augment_ops(img)

                # nowa nazwa pliku
                new_filename = f"{src_img_path.stem}_aug{i}{src_img_path.suffix}"
                new_path = class_dir / new_filename

                augmented.save(new_path)
                i += 1

    print("Augmentacja zakończona.")


def split_dataset_from_temp(temp_dir="temp", output_dir="data", split=(0.8, 0.1, 0.1)):
    """Dzieli dane na zbiór treningowy, walidacyjny i testowy"""
    temp_dir = Path(temp_dir)
    output_dir = Path(output_dir)

    # Nowe foldery: train, val, test
    for split_name in ["train", "val", "test"]:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    for class_dir in temp_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        all_images = list(class_dir.glob("*.jpeg"))
        random.shuffle(all_images)

        total = len(all_images)
        train_end = int(split[0] * total)
        val_end = train_end + int(split[1] * total)

        splits = {
            "train": all_images[:train_end],
            "val": all_images[train_end:val_end],
            "test": all_images[val_end:]
        }

        for split_name, files in splits.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for file_path in files:
                shutil.copy(file_path, dest_dir)

        print(
            f"{class_name}: {total} obrazków -> train:{len(splits['train'])}, val:{len(splits['val'])}, test:{len(splits['test'])}")

    print("Podział danych zakończony.")

    # Usuń folder temp po zakończeniu
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"Usunięto folder tymczasowy: {temp_dir}")


if __name__ == "__main__":
    download_and_prepare_data()  # Pobranie danych
    merge_to_temp()  # Scalenie danych testowych i treningowych do folderu temp
    augment_dataset()  # Augmentacja danych - podwojenie liczby zdjęć
    split_dataset_from_temp()  # Podział danych na zbiór treningowy, testowy i walidacyjny

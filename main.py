import kagglehub
import pandas as pd
import cv2

from pathlib import Path

import src.preprocessing as pp

dataset_path_str = kagglehub.dataset_download("mariaherrerot/aptos2019")
dataset_path = Path(dataset_path_str)

train_csv = dataset_path / "train_1.csv"
df = pd.read_csv(train_csv)

first_image = df.iloc[0]["id_code"] + ".png"
first_image = pp.preprocess_dr_image(
    dataset_path / "train_images" / "train_images" / first_image
)

second_image = df.iloc[3]["id_code"] + ".png"
second_image = pp.preprocess_dr_image(
    dataset_path / "train_images" / "train_images" / second_image
)


cv2.imwrite("test1.png", first_image)
cv2.imwrite("test2.png", second_image)

import os
import pandas as pd
from instruments.dsloader import DatasetLoader
from instruments.preprocessor import SimplePreprocessor
from imutils import paths
from pathlib import Path

# Path to file that has the script:
project_dir = os.path.dirname(__file__)
img_folder = os.path.join(project_dir, "data/train_v2/")
imagePaths = sorted(list(paths.list_images("{}".format(img_folder))))
preprocesssor = SimplePreprocessor(32, 32)
dsl = DatasetLoader([preprocesssor])
loc_path = os.path.join(project_dir, "data/train_ship_segmentations_v2.csv")

# Get image locations
def get_img_loc(loc_path, labels):
    loc_data = pd.read_csv(loc_path)
    loc_data.set_index("ImageId", inplace=True)
    labels = labels.tolist()
    labels_data = loc_data.loc[labels]
    print(labels_data)


# A method, that combines all information into a structured dict
def get_info_dict(imagePaths, loc_path):
    info_dict = dict()
    images, labels = dsl.load(imagePaths)
    loc_dict = get_img_loc(loc_path, labels)


get_info_dict(imagePaths, loc_path)

import zipfile

path_to_zip_file = "nature_12K.zip"
directory = "dataset"

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory)
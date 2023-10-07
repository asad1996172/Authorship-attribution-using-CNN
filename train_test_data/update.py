import glob
import os
import pickle

# get current directory
dir_path = os.getcwd()

# find all .pkl files in directory
pkl_files = glob.glob(os.path.join(dir_path, "*.pkl"))

# load each .pkl file
for file in pkl_files:
    with open(file, "rb") as f:
        data = pickle.load(f)
    updated_data = []
    for file_path, filename, authorLabel, author, featureList, inputText in data:
        updated_data.append((inputText, authorLabel))
    with open(file, "wb") as f:
        pickle.dump(updated_data, f)

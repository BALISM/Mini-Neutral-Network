import numpy as np
import urllib.request
import gzip
import os

def load_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    // it is a dictionary in whcih there is the nickname for every key variable and also 
    // it just loop through it every single time 

    os.makedirs("mnist_data", exist_ok=True)
    data = {}

    for key, filename in files.items():
        filepath = f"mnist_data/{filename}"
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)

        with gzip.open(filepath, "rb") as f:
            if "images" in key:
                # first 16 bytes are header info, rest is pixel data
                data[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
            else:
                # first 8 bytes are header info, rest is labels
                data[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    return data["train_images"], data["train_labels"], data["test_images"], data["test_labels"]

X_train, y_train, X_test, y_test = load_mnist()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
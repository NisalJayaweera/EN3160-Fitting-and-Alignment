# EN3160 Assignment 2 — Colab setup
!git clone https://github.com/NisalJayaweera/EN3160-Fitting-and-Alignment.git
%cd EN3160-Fitting-and-Alignment
%pip install -q -r requirements.txt
from google.colab import files
from pathlib import Path
import zipfile
uploaded = files.upload()  # Select a2images.zip from the assignment materials
for filename in uploaded:
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(filename) as z:
            z.extractall("images")
        print("Extracted", filename)
!find images -type f | head -30
# The extraction may include nested folders. Check the printed paths and
# adjust the --image argument accordingly before running Q1.

from PIL import Image
from pytesseract import pytesseract
import pandas as pd
from pathlib import Path
import os


#image converter to txt,
path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# path changes depending on directory of user
path_to_images = r'images/'
pytesseract.tesseract_cmd = path_to_tesseract

for root, dirs, file_names in os.walk(path_to_images):
    for file_name in file_names:
        # Open image with PIL
        img = Image.open(path_to_images + file_name)

        # Extract text from image
        text = pytesseract.image_to_string(img)

        print(text)

#csv converter to text files

def export_csv(input_filepath, output_filepath):
    data = pd.read_csv(input_filepath)
    data.drop('symbol', inplace=True, axis=1)
    data['datetime'] = pd.to_datetime(data['datetime']).dt.strftime('%Y%m%d')
    data.to_csv(output_filepath, sep=';', header=None, index=False)


folderpath = Path('path/to/csv/files/folder').resolve()
new_suffix = '.txt'

# Convert all .csv files from text database.
for input_filepath in folderpath.glob('*.csv'):
    #convert file extension to txt
    output_filepath = input_filepath.with_suffix(new_suffix)
    export_csv(input_filepath, output_filepath)  # Convert the file.

# ideas for granularity
# sentence- either detect a period in text, or a specific word count
# paragraph- either detect a specific number of periods, or a specific word count

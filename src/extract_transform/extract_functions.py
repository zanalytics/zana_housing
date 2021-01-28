import requests, zipfile, io
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def raw_data(url, destination, file_name):
    """
        Read an open csv through http. and write a specified location.

        Parameters:
            - url : string
                HTTP url ending with the .csv extension.
            - destination : string
                Local directory 
            - file_name : string
                The file name you want to write out the csv.

        Returns:
            - writes the file to destination.
    """

    r = requests.get(url, stream=True)
    file_path = destination + file_name

    with open(file_path, "wb") as csv:
        for chunk in r.iter_content(chunk_size=10 ** 6):
            if chunk:
                csv.write(chunk)
    return logging.info("File written to destination")

def download_zip(url, destination):
    """
        Download zip file and extract all contents into destination

        Parameters:
            - url : string
                HTTP url ending with the .zip extension.
            - destination : string
                Local directory 

        Returns:
            - writes the zipped files in the destination
    """
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(destination)
    return logging.info("All files unzipped into destination")

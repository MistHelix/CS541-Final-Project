import os
import re

from google.cloud import storage


def download_public_file():
    bucket_name = "gcp-public-data-goes-16"
    prefix = "ABI-L1b-RadM/"
    destination_file_name = "goes_images_northeast/"

    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.get_bucket(bucket_name)
    for year in range(2017, 2026):
        for day in range(1,366):
            formatted_day = str(day).zfill(3)
            blobs = bucket.list_blobs(prefix=prefix+str(year)+"/"+formatted_day+"/12/")
            time = ""
            timeGotten = False
            for blob in blobs:
                if "C01" in str(blob.name) and not timeGotten:
                    time = re.search( r'_s(\d{14})', str(blob.name)).group(1)
                    print(time)
                    timeGotten = True
                    # It won't generate its own folders
                    local_folder = os.path.join(destination_file_name+"ABI-L1b-RadM/", str(year), formatted_day, "12/")
                    os.makedirs(local_folder, exist_ok=True)
                    blob.download_to_filename(destination_file_name+blob.name)
                if "C02" in str(blob.name) and time in str(blob.name):
                    print(blob.name)

                    # we only need one file
                    blob.download_to_filename(destination_file_name+blob.name)
                if "C03" in str(blob.name) and time in str(blob.name):
                    print("C03")
                    # we only need one file
                    blob.download_to_filename(destination_file_name+blob.name)


download_public_file()
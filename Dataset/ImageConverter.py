import os

from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

path = 'goes_images_northeast/ABI-L1b-RadM/'


def gen_image(path):
    print("Path:" + path)
    c01_file = ''
    c02_file = ''
    c03_file = ''
    if not os.path.isdir(path):
        return None
    for filename in os.listdir(path):
        if 'C01' in filename.upper():
            c01_file = os.path.join(path, filename)
        elif 'C02' in filename.upper():
            c02_file = os.path.join(path, filename)
        elif 'C03' in filename.upper():
            c03_file = os.path.join(path, filename)
    if '' in [c01_file, c02_file, c03_file]:
        print(c01_file)
        print(c02_file)
        print(c03_file)
        print("Error at" + path)
        return None
    # Red Band
    print('Co2' + c02_file)
    g16nc = Dataset(c02_file, 'r')
    radiance = g16nc.variables['Rad'][:]
    g16nc.close()

    # From paper that is gone
    Esun_Ch_01 = 726.721072
    Esun_Ch_02 = 663.274497
    Esun_Ch_03 = 441.868715
    d2 = 0.3

    ref = (radiance * np.pi * d2) / Esun_Ch_02
    ref = np.maximum(ref, 0.0)
    ref = np.minimum(ref, 1.0)

    ref_gamma = np.sqrt(ref)

    # Blue Band

    print(c01_file)
    g16nc = Dataset(c01_file, 'r')
    radiance_1 = g16nc.variables['Rad'][:]
    g16nc.close()
    g16nc = None

    ref_1 = (radiance_1 * np.pi * d2) / Esun_Ch_01
    # Make sure all data is in the valid data range
    ref_1 = np.maximum(ref_1, 0.0)
    ref_1 = np.minimum(ref_1, 1.0)
    ref_gamma_1 = np.sqrt(ref_1)

    # Green bang
    g16nc = Dataset(c03_file, 'r')
    radiance_3 = g16nc.variables['Rad'][:]
    g16nc.close()
    g16nc = None
    ref_3 = (radiance_3 * np.pi * d2) / Esun_Ch_03

    print(c03_file)
    ref_3 = np.maximum(ref_3, 0.0)
    ref_3 = np.minimum(ref_3, 1.0)
    ref_gamma_3 = np.sqrt(ref_3)

    def rebin(a, shape):
        sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    ref_gamma_2 = rebin(ref_gamma, [1000, 1000])

    ref_gamma_true_green = 0.48358168 * ref_gamma_2 + 0.45706946 * ref_gamma_1 + 0.06038137 * ref_gamma_3

    truecolor = np.stack([ref_gamma_2, ref_gamma_true_green, ref_gamma_1], axis=2)
    # fig = plt.figure(figsize=(6, 6), dpi=200)
    # im = plt.imshow(truecolor)
    # plt.title('TrueColor - Red - Psuedo-Green - Blue')
    # plt.show()

    return truecolor


def gen_images():
    for year in range(2017, 2026):
        for day in range(1, 366):
            formatted_day = str(day).zfill(3)
            local_folder = os.path.join("pre_crop/ABI-L1b-RadM/", str(year), formatted_day, "12/")
            os.makedirs(local_folder, exist_ok=True)
            truecolor = gen_image(path + str(year) + "/" + formatted_day + "/12/")
            if not truecolor is None:
                print("Making image")
                image = Image.fromarray((truecolor * 255).astype(np.uint8), 'RGB')
                print(("pre_crop/ABI-L1b-RadM/" + str(year) + "/" + formatted_day + "/12/" + "truecolor" + str(
                    year) + formatted_day + ".png"))
                image.save(("pre_crop/ABI-L1b-RadM/" + str(year) + "/" + formatted_day + "/12/" + "truecolor" + str(
                    year) + formatted_day + ".png"))


def crop_images():
    crop_box = (562, 126, 1000, 322)
    for year in range(2017, 2026):
        for day in range(1, 366):
            formatted_day = str(day).zfill(3)
            local_folder = os.path.join("post_crop/ABI-L1b-RadM/", str(year), formatted_day, "12/")
            os.makedirs(local_folder, exist_ok=True)
            if not os.path.isfile(("pre_crop/ABI-L1b-RadM/" + str(year) + "/" + formatted_day + "/12/" + "truecolor" + str(
                    year) + formatted_day + ".png")):
                print(("pre_crop/ABI-L1b-RadM/" + str(year) + "/" + formatted_day + "/12/" + "truecolor" + str(
                    year) + formatted_day + ".png"))
                continue
            img = Image.open(("pre_crop/ABI-L1b-RadM/" + str(year) + "/" + formatted_day + "/12/" + "truecolor" + str(
                year) + formatted_day + ".png"))
            cropped_img = img.crop(crop_box)
            cropped_img.save(("post_crop/ABI-L1b-RadM/" + str(year) + "/" + formatted_day + "/12/" + "truecolor" + str(
                year) + formatted_day + ".png"))


crop_images()

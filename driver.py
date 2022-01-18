import argparse
import os
import glob
from PlatePhoto import PlatePhoto

def get_image_paths(path):
    if not os.path.exists(path):
        raise Exception('Please enter a valid path to directory or image')
    
    if os.path.isdir(path):
        images = glob.glob(os.path.join(path, '*.*'))
    else:
        images = [path]
    return images

def open_images(path):
    paths = get_image_paths(path)

    plates = [PlatePhoto(path) for path in paths]

    return plates

def main(path):
    plates = open_images(path)

    for plate in plates:
        plate.split_align()
        plate.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorize single plate, or directory of plate images')
    parser.add_argument('--path',help='path to plate image(s)')

    args = parser.parse_args()
    main(args.path)
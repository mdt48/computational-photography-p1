import sys
from curses import window
from re import A, L, search

import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from tqdm import tqdm


class PlatePhoto:
    def __init__(self, path_to_plate) -> None:
        image = Image.open(path_to_plate)

        self.image_data = np.asarray(image)

    def split_align(self, crop_borders=False):
        B, G, R, = self.__split_photos(crop_borders)
        aligned_image = self.__align_photos(B, G, R)

        self.aligned_image, self.original = aligned_image

    def show(self):
        im = Image.fromarray(self.aligned_image)
        im.show()

        orig = Image.fromarray(self.original)
        orig.show()

    def __split_photos(self, crop_borders):
        """Handle the border cropping, splitting of channels, and padding
        of data that does not have enough rows"""
        # crop borders
        if crop_borders:
            # mask = self.image_data == 255
            pass

        # split image into the three separate channels
        # height of one image to nearest integer index
        height = self.image_data.shape[0] // 3

        B = self.image_data[:height, :]
        G = self.image_data[height:int(height*2), :]
        R = self.image_data[int(height*2):, :]

        plate_list = [B, G, R]

        # odd length images may need to be padded
        num_rows = np.array([B.shape[0], G.shape[0], R.shape[0]])
        max_rows = np.max(num_rows)

        differences = (num_rows == max_rows)

        if np.all(differences):
            return plate_list
        else:
            arrays_that_differ = np.argwhere(differences == True)
            for array in arrays_that_differ:
                array = int(array)
                for i in range(max_rows-height):
                    plate_list[array] = np.delete(plate_list[array], 0,0)
            return tuple(plate_list)

        

    def __align_photos(self, B, G, R):
        G_best_displacement, R_best_displacement = None, None
        G_best_error, R_best_error = sys.maxsize, sys.maxsize
        G_best_aligned, R_best_aligned = None, None

        
        search_area = 50

        for x in tqdm(list(range(-search_area, search_area))):
            for y in range(-search_area, search_area+1):

                G_x, G_y = -x,-y
                R_x, R_y = -x,-y

                aligned_G = np.roll(G, (x,y), axis=(1,0))
                aligned_R = np.roll(R, (x,y), axis=(1,0))


                G_error = self.__squared_error(B, aligned_G)
                R_error = self.__squared_error(B, aligned_R)

                if G_error < G_best_error:
                    G_best_error = G_error
                    G_best_displacement = (G_x, G_y)
                    G_best_aligned = aligned_G
                
                if R_error < R_best_error:
                    R_best_error = R_error
                    R_best_displacement = (R_x, R_y)
                    R_best_aligned = aligned_R
   
        print('G best alignment: {}'.format(G_best_displacement))
        print('R best alignment: {}'.format(R_best_displacement))


        return np.stack((R_best_aligned, G_best_aligned, B), axis=2), np.stack((R,G, B), axis=2)



    def __squared_error(self, A, B):
        return np.sum(np.square(A-B))

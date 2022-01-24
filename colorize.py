# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

from re import search
import numpy as np
import skimage as sk
import skimage.io as skio
import sys
import cv2
import os
from tqdm import tqdm
import glob

def main():
    # get all files
    fns = glob.glob('./data/*.*')
    # fns = ['/Users/mdt/projects/computational-photography-p1/data/00458u.jpg']
    for imname in tqdm(fns):

        # read in the image
        im =  skio.imread(imname)

        # split image
        b, g, r = split(im)

        # create unaligned iamge for comparison 
        orig = np.dstack([r,g,b])

        # crop borders 
        r, g, b = crop_borders(r,g,b,crop_percentage=0.12)

        # get edges
        # eb, eg, er = edge_detection(r,g,b)
        eb, eg, er = b, g ,r
        
        # only do the downsampling technique if its a larger image!
        if b.shape[0] > 1000:
            height = 3
            edge_p = pyramids(eb,er,eg, height)
        elif b.shape[0] < 1000:
            height = 1
            edge_p = pyramids(eb,er,eg, height)

        # displacement for r and g channels
        dr, dg = np.array([0,0]), np.array([0,0])

        # for each level of downsampling(height = height of pyramid)
        for i in range(height):
            # if its the most downsampled, search 7 percent of image 
            if i == 0:
                search_area = (int(0.07 * edge_p['blue'][0].shape[0]), int(0.07* edge_p['blue'][0].shape[1]))
            # if its the full size image roll the image and exit
            elif i == height - 1:
                edge_p['red'][i] = np.roll(edge_p['red'][i], tuple(dr), axis=(1,0))
                edge_p['green'][i] = np.roll(edge_p['green'][i], tuple(dg), axis=(1,0))
                break
            else:
                # any other sampled image just search in a smaller 2,2 area 
                search_area = (2,2)
                # adjust red and search in 2,2
                edge_p['red'][i] = np.roll(edge_p['red'][i], tuple(dr), axis=(1,0))
                # adjust green and search in 2,2
                edge_p['green'][i] = np.roll(edge_p['green'][i], tuple(dg), axis=(1,0))

            # align
            res = align(edge_p['red'][i], edge_p['green'][i], edge_p['blue'][i], search_area)

            # only scale displacement if we are downsampling images
            if height != 1:
                dr, dg = dr*2, dg*2

            # total displacement
            dr  = dr + res[0]
            dg = dg + res[1]

           

        print('Displacement for R: {}'.format(dr))
        print('Displacement for G: {}'.format(dg))

        # displace r,g by final displacements
        ar = np.roll(r, dr, axis=(1,0))
        ag = np.roll(g, dg, axis=(1,0))

        # create aligned image
        im_out = np.dstack([ar,ag,b])

        # save the image
        dirs = './auto_crop_no_edge/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        fname = dirs+os.path.basename(imname).split('.')[0] + '_' + str(height)
        skio.imsave(fname+'.jpg', im_out)
        skio.imsave(fname+'_original.jpg', orig)


def pyramids(eb, er, eg, height=4):
    """
    Takes in the three color channels and downsamples by half height times.
    Returns dictionary of the pyramid downsampling
    """
    pyramids = {'red':[], 'green': [],'blue': []}
    for i in range(height-1,-1, -1):
        factor = 0.5**i
        # factor = 0.10
        
        pyramids['blue'].append(cv2.resize(eb, (0,0), fx=factor, fy=factor))
        pyramids['red'].append(cv2.resize(er, (0,0), fx=factor, fy=factor))
        pyramids['green'].append(cv2.resize(eg, (0,0), fx=factor, fy=factor))

    return pyramids

def edge_detection(r,g,b):
    er = cv2.GaussianBlur(r, (5,5), sigmaX=1, sigmaY=1) - r 
    eg = cv2.GaussianBlur(g, (5,5), sigmaX=1, sigmaY=1) - g 
    eb = cv2.GaussianBlur(b, (5,5), sigmaX=1, sigmaY=1) - b

    return eb, eg, er 

def align(r,g,b, search_area):
    error_G, error_R = [], []
    displacement_G, displacement_R = [], []
    for x in range(-search_area[0], search_area[0]):
        for y in range(-search_area[1], search_area[1]):
            trans_R = np.roll(r, (x,y), axis=(1,0))
            trans_G = np.roll(g, (x,y), axis=(1,0))

            error_G.append(ssq(b, trans_G))
            error_R.append(ssq(b, trans_R))

            displacement_G.append([x,y])
            displacement_R.append([x,y])

    min_idx_G = np.argmin(error_G)
    min_idx_R = np.argmin(error_R)

    return np.array(displacement_R[min_idx_R]), np.array(displacement_G[min_idx_G])

def ssq(A, B):
    return np.mean(np.sum( np.square(A-B) ))

def split(im):

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = int(np.floor(im.shape[0] / 3.0))

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    return b,g,r

def crop_borders(r, g, b, crop_percentage=0.15):
    x,y = b.shape
    margin = 5
    
    # white border size
    white_width, white_height = crop_color(b, margin, color=200)

    total_width = white_width
    total_height = white_height

    r = r[total_width:x-total_width, total_height:y-total_height]
    g = g[total_width:x-total_width, total_height:y-total_height]
    b = b[total_width:x-total_width, total_height:y-total_height]
    return r, g, b

def crop_color(b, margin, color):
    middle_of_b = b.shape[0] // 2
    idxs = np.argwhere((b[middle_of_b]*255).astype(int) > color)
    
    white_width = get_size_of_border(margin, idxs)

    middle_of_b = b.shape[1] // 2
    idxs = np.argwhere((b[:, middle_of_b]*255).astype(int) > color)
    
    white_height = get_size_of_border(margin, idxs)
    return white_width, white_height

def get_size_of_border(margin, idxs):
    for idx, val in enumerate(idxs):
        if idx + 1 >= len(idxs):
            return len(idxs)
        if margin < np.abs(int(idxs[idx+1]) - val):
            return idx

if __name__ == '__main__':
    main()
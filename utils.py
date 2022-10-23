import imageio
import imageio.v3 as iio

# no need to import numpy, imageio has already imported it
np = iio.np

# a kind of film for recording gif videos
recorded_images = []

def load_island_image():
    image = iio.imread('pictures/color_islands.png')
    image = image[...,:3] # we don't need the alpha channel 
    return image

# helper consts for readability
black_colorvector = [ 0, 0, 0]
black_colorcode = 0

def make_black_islands_from_color_ones( image ):
    image_black_islands = image.copy()
    # trick: the very first pixel on the top left corner is surely water
    # set all pixels black (0,0,0) that is not water pixel 
    image_black_islands[(image != image[0,0]).all(-1)] = black_colorvector
    return image_black_islands

def get_rgb_colorvectors_from_image( image ):
    return np.unique(image.reshape(-1,3), axis = 0)

def make_colorvectors_from_color_dict( color_dict ):
    "make dictionary with given color names and simple 3 number RGB color vectors"
    colorvector_dict = {}
    for color_name, hex_color in color_dict.items():
        colorvector = [ int(hex_color[i:i+2], base = 16) for i in range(0, 6, 2) ]
        colorvector_dict[color_name] = colorvector
    return colorvector_dict

def make_colorcodes_from_color_dict( color_dict ):
    "assign ordinal numbers to color names, following their order"
    colorcode_dict = {}
    for i, color_name in enumerate(color_dict):
        colorcode_dict[color_name] = i
    return colorcode_dict

def get_colorvectors_from_colorvector_dict( colorvector_dict ):
    return np.array(list(colorvector_dict.values())).astype(np.uint8)

def convert_image_to_colormap_slow( image, colorvectors ):
    "convert image RGB vectors to simple integer numbers according to given color vectors"
    image_colormap = []
    for pixel_colorvector in image.reshape(-1,3):
        # don't ask, but in a nutshell: find the place of the colorvector of the current
        # pixel in the given colorvector list; np.argwhere return is a mess, tricks needed
        u, c = np.unique(
                   np.argwhere(colorvectors == pixel_colorvector)[:,0],
                   return_counts = True)
        image_colormap.append(u[c == 3])
    return np.array(image_colormap).reshape(image.shape[0],image.shape[1])

def convert_image_to_colormap( image, colorvectors):
    "convert image RGB vectors to simple integer numbers according to given color vectors"
    # trick: we can make 'hashes', or at least unique numbers for each pixel simultaneously
    # using functions like sum() or prod() on the last axis (axis = -1 or simply -1) 
    pixel_hashlist = (image.sum(-1) + image.prod(-1)).ravel()
    
    # we also make the same hashes for the colors in our color table
    colorvector_hashlist = colorvectors.sum(-1) + colorvectors.prod(-1)
    # then we can give each hash a number between 0 and the number of colors (hashes)
    # similariry as in make_colorcodes_from_color_dict()
    colorvector_hashmap = {}
    for i, colorvector_hash in enumerate(colorvector_hashlist):
        colorvector_hashmap[colorvector_hash] = i
    
    # now we can replace each pixel hash on the image to simple ordinal numbers
    image_colormap = [ colorvector_hashmap[pixel_hash] for pixel_hash in pixel_hashlist]
    return np.array(image_colormap).reshape(image.shape[0],image.shape[1]).astype(np.uint8)

def convert_colormap_to_image( image_colormap, colorvectors ):
    "convert integer number image colormap to image RGB vectors"
    # the other way around is much simpler
    return colorvectors[image_colormap]

# helper for parallel painting 
def convert_increasing_map_to_colormap( image_increasing_map, colorvectors ):
    
    image_colormap = np.copy( image_increasing_map )
    island_color_count = len(colorvectors) - 2
    huge_number = image_increasing_map.size
    image_colormap[ image_increasing_map == huge_number ]  = 1 # color code 1 is sea
    #image_colormap[ image_increasing_map != huge_number ] *= 257
    #image_colormap[ image_increasing_map != huge_number ] &= 255
    image_colormap[ image_increasing_map != huge_number ] %= island_color_count
    image_colormap[ image_increasing_map != huge_number ] += 2 # island color codes start at 2

    return image_colormap.astype(np.uint8)

# RGB image render, make gif, save

def reset_gif_frames( images = recorded_images):
    images.clear() # note, images = [] (rebind!) would be unnoticed for the outer world

def render_gif_frame( image_colormap, colorvectors, images = recorded_images ):
    image = convert_colormap_to_image( image_colormap, colorvectors)
    images.append(image)

def save_gif(filename, images = recorded_images, fps = 50):
    images += [images[-1]] * fps * 2 # hold last frame for 2 seconds
    print(f'Saving {filename}, it can take a while')
    imageio.mimwrite(filename, images, fps = fps, format = '.gif')
    print('Done')


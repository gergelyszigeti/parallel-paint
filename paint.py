import algorithms
import utils

# I played around a bit on trycolors.com (great site, worth to try)
color_dict = { # black islands
               'black'      : '000000',
               # water color
               'turquoise'  : '19ecef',
               # island colors
               'ocean blue' : '2b65ec',
               'red violet' : 'aa1872',
               'green'      : '44991e',
               'yellow'     : 'ffed00',
               'orange'     : 'ff7700',
               'blue'       : '1283a0',
               'supernova'  : 'f3c700',
               'sulu green' : '92ed6d',
               'cobalt'     : '044aab',
               'scarlet'    : 'ff3513',
               'dark green' : '44991e',
             }

colorvec_dict = utils.make_colorvectors_from_color_dict( color_dict )
colorvectors = utils.get_colorvectors_from_colorvector_dict( colorvec_dict )
colorcode_dict = utils.make_colorcodes_from_color_dict( color_dict )

im_islands = utils.load_island_image()
print(f'Image of color islands loaded, shape: {im_islands.shape}')
im_black_islands = utils.make_black_islands_from_color_ones( im_islands )
im_black_islands_colormap = utils.convert_image_to_colormap(im_black_islands, colorvectors)

# take the upper right corner only with 2 islands, as the serial
# painting process is really slow even for one island
# and it is easier to zoom out two islands, making the process more visible

im_black_islands_colormap_urc = im_black_islands_colormap[:165,95:248]
'''
level = 1

import sys
sys.setrecursionlimit(4000)

algorithms.paint( im_black_islands_colormap_urc, 124-95, 78, colorcode_dict['red violet'], colorvectors )
print('\nNumber of points:', algorithms.p_count)
utils.save_gif('island_paint_1st_of_2.gif.firstsep.gif')

# paint the second one, record, save

algorithms.p_count = 0
utils.reset_gif_frames()

algorithms.paint( im_black_islands_colormap_urc, 100, 100, colorcode_dict['yellow'], colorvectors )
print('\nNumber of points:', algorithms.p_count)
utils.save_gif('island_paint_2nd_of_2.gif.firstsep.gif')
'''
im_black_islands_colormap_all = utils.np.copy(im_black_islands_colormap)
print("painting all islands in parallel, also recording the process as a gif video")
final_colormap = algorithms.paint_parallel( im_black_islands_colormap_all, colorvectors )
print("\nDone")
print(algorithms.root_information)
print(set(final_colormap.flatten().tolist()))
utils.save_gif('island_paint_parallel.gif', fps = 1)

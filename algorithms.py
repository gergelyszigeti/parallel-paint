import utils

np = utils.np # numpy already imported in utils

#########################################################################################
#                                                                                       #
#                           Simple brute force serial painting                          #
#                                                                                       #
#########################################################################################

p_count = 0
recursion_level = 0

def paint( image_colormap, x, y, colorcode, colorvectors ):
    global recursion_level
    recursion_level += 1
    # for the sake of simplicity, no 0<=x<w and 0<=y<h check as water fills the borders
    if image_colormap[y,x] == utils.black_colorcode:
        image_colormap[y,x] = colorcode
        # let's count how many pixels we have in this island!
        global p_count
        p_count += 1
        # as we painted a pixel, let's render it as frame to see the process on video (gif)
        utils.render_gif_frame( image_colormap, colorvectors )
        paint(image_colormap, x - 1, y - 1, colorcode, colorvectors)
        paint(image_colormap, x    , y - 1, colorcode, colorvectors)
        paint(image_colormap, x + 1, y - 1, colorcode, colorvectors)
        paint(image_colormap, x - 1, y    , colorcode, colorvectors)
        paint(image_colormap, x + 1, y    , colorcode, colorvectors)
        paint(image_colormap, x - 1, y + 1, colorcode, colorvectors)
        paint(image_colormap, x    , y + 1, colorcode, colorvectors)
        paint(image_colormap, x + 1, y + 1, colorcode, colorvectors)
    # note: we return immediately on paint attempt of water or an already painted island pixel
    print(f'\rreturn from recursion level: {recursion_level} ', end = '')
    recursion_level -= 1
    return

#########################################################################################
#                                                                                       #
#                          Wiser, more complex parallel painting                        #
#                                                                                       #
#########################################################################################

root_information = set()

# helper functions for parallel painting
def increasing_value_islands( image_colormap ):
    "change island pixels to increasing values, sea pixels to one big number"
    # lets make an image_colormap sized array with increasing values only (0 <= v < width x height)
    increasing_values = np.arange(image_colormap.size).reshape(image_colormap.shape)
    # lets make a kind of mask map, island points are True (1), sea points are False (0)
    island_map = np.logical_not(image_colormap)
    # its inverse is also a mask map, sea points are False (0), island points are True (1)
    sea_map = ~island_map

    # now we make a mix of islands with increasing numbers and sea pixels as "huge numbers"
    # (the huge number is actually width x height, greater than any value in increasing_values)
    huge_number = image_colormap.size   # short for width x height
    return increasing_values * island_map + huge_number * sea_map

# retrieve neighbor value at x,y if applicable (applicable within map and in current area only)
def neighbor( image_map, x, y, current_area_x, current_area_y, area_size):
    "get neighbor values in the same area, otherwise a big number"
    h,w = image_map.shape
    huge_number = image_map.size   # = w*h
    area_x, area_y = x // area_size, y // area_size
    it_is_on_the_map = ( 0 <= x and x < w ) and ( 0 <= y and y < h )
    it_is_in_the_same_area =     area_x == current_area_x               \
                             and area_y == current_area_y
    it_is_a_neighbor = it_is_on_the_map and it_is_in_the_same_area
    return image_map[y,x] if it_is_a_neighbor else huge_number

def find_deepest_root( image_map, x, y ):

    # Current value of current x,y 'slot' points to a slot, which is the direct root of x,y
    # We update the current value with the direct root, as long as the (next) direct root is
    # in another slot (one or some steps only)
    while ( (direct_root := image_map.reshape(-1)[image_map[y,x]])
            != image_map[y,x] ):
        image_map[y,x] = direct_root
        root_information.add(direct_root)

    # note, image_map is mutable, we don't need to return anything

# parallel painting algorithm
def paint_parallel( image_colormap, colorvectors ):
    # first step: fill up black islands with increasing values
    image_increasing_map = increasing_value_islands( image_colormap )
    # record first frame of the video, this is the initial state
    image_colormap = utils.convert_increasing_map_to_colormap(image_increasing_map, colorvectors)
    utils.render_gif_frame( image_colormap, colorvectors )

    # init main area loop (one step is one frame in output gif, let me call one step as one round)
    area_size = 2
    h, w = image_increasing_map.shape
    huge_number = sea_code = image_increasing_map.size # sometimes sea_code says more,
                                                       # sometimes huge_number
    while area_size//2 <= w or area_size//2 <= h:

        # First part in each area-stretching round:
        # modify roots according to lower neighbors in area
        # Note: looks like a O(n2) part in a O(log2(n)) main loop, however,
        # these two loops are meant to be processed in paralell on GPUs
        for y in range(h):
            for x in range(w):

                current_xy_value = image_increasing_map[y,x]

                if current_xy_value != sea_code:
                    # note, this part is not written in numpy/python fashion,
                    # it more resembles the CUDA/C++ way
                    area_x, area_y = x // area_size, y // area_size
                    neighbor_min =                                          \
                      min( neighbor( image_increasing_map, x - 1, y - 1, area_x, area_y, area_size ),
                      min( neighbor( image_increasing_map,     x, y - 1, area_x, area_y, area_size ),
                      min( neighbor( image_increasing_map, x + 1, y - 1, area_x, area_y, area_size ),

                      min( neighbor( image_increasing_map, x - 1,     y, area_x, area_y, area_size ),
                      min( neighbor( image_increasing_map, x + 1,     y, area_x, area_y, area_size ),

                      min( neighbor( image_increasing_map, x - 1, y + 1, area_x, area_y, area_size ),
                      min( neighbor( image_increasing_map,     x, y + 1, area_x, area_y, area_size ),
                      min( neighbor( image_increasing_map, x + 1, y + 1, area_x, area_y, area_size ),

                           current_xy_value
                      )))))))) # these are nested mins, a strategy we are going to use in CUDA
                               # (on GPUs, min() with two parameters is a separate machine command,
                               #  executed in a blink of an eye)
                    if neighbor_min < current_xy_value:
                        image_increasing_map.reshape(-1)[current_xy_value] = neighbor_min
                                                   # set direct root to the value of lower neighbor

        # Second part in each round, excluding the first one:
        # replace all values with their deepest roots
        # In the first round, this is automatically done above as the area is too small,
        # with only 4 island values at most; if there are more than one island values,
        # one of them is always the root value for the other one(s)

        root_information.clear()
        if area_size > 2:
            for y in range(h):
                for x in range(w):
                    if image_increasing_map[y,x] != sea_code:
                        find_deepest_root( image_increasing_map, x, y )

        # make the area bigger for the next step
        area_size *= 2  # so, that is why I call it a 'log2n' algorithm
                        # taking power of twos, we reach the end soon
                        # (e.g in 9 steps only for the island example)
                        # now it is worth to note, the x and y for loops are meant to be
                        # computed in parallel on GPU
                        # ( actually all pixels are computed in parallel if you have a GPU from
                        #   somewhere in the distant future with at least w*h = 282 x 373 = 104,440
                        #   small processing units )

        # record what is happening, in a gif video
        # coloring? let's modulo increasing values for mapping onto our color vector list
        # Some islands will have the same color, but we can not do it much better
        # with just a few colors
        image_colormap = utils.convert_increasing_map_to_colormap(image_increasing_map, colorvectors)
        utils.render_gif_frame( image_colormap, colorvectors )

        # print some status, we are alive
        print('.', end='', flush=True)

    # let's return the result map, for statistics, e.g. the number of islands and their values
    # (the other output, the gif animation is already handled above)
    return image_increasing_map

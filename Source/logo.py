""" The following file scales and adds the logo image to any required plots."""

import matplotlib.pyplot as plt


def scale(im, nr, nc):
    """ Scales the logo to the desired length and width

       Parameters:
       - im: The image
       - nr: The row size
       - nc: The column size

       Returns:
       - scaled image
       """
    number_rows = len(im)
    number_columns = len(im[0])
    return [[im[int(number_rows * r / nr)][int(number_columns * c / nc)]
             for c in range(nr)] for r in range(nc)]


def watermark(ax, x0, y0):
    """ Adds logo and positions it on the plot

        Parameters:
        - ax: figure object
        - x0: adds x location
        - y0: adds y location
    """

    logo = plt.imread('jmann logo.png')
    # scale Image
    logo = scale(logo, 150, 150)

    ax.figure.figimage(logo, x0, y0, zorder=2, origin="upper")

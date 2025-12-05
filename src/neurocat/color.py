from matplotlib.colors import ListedColormap as clc
import pylab as plt
import yaml
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
from .util import __base__

# load all colors
with open(__base__ / "color.yml", 'r') as f:
    colors = yaml.load(f, Loader=yaml.FullLoader)

globals().update(colors)


def get_transparent_cm(n_colors, cm_name):
    """
    Creates a fully transparent colormap.

    Parameters:
        n_colors (int): Number of colors in the colormap.
        cm_name (str): Name of the colormap.

    Returns:
        LinearSegmentedColormap: A colormap with fully transparent colors.
    """
    transparent_color = (0, 0, 0, 0)  # RGBA for fully transparent black
    colors = [transparent_color] * n_colors
    return mcolors.LinearSegmentedColormap.from_list(cm_name, colors)


# color concerned
def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [np.append((1 - mix) * c1_rgb + (mix * c2_rgb), 1.) for mix in mix_pcts]
    return rgb_colors


def get_cm(color_list, n_fine, cm_name):
    color_n = len(color_list)

    this_color = color_list.pop()
    colors = []

    while len(color_list) >= 1:
        next_color = color_list.pop()
        # print(f"this color:{this_color}, next color:{next_color}")
        colors = colors + get_color_gradient(this_color, next_color, n_fine)
        this_color = next_color
    colors = np.array(colors)
    return clc(colors, name=cm_name)


def save_cm(cm, name):
    """save color map"""
    a = np.array([[0, 1]])
    plt.figure(figsize=(9, 1.5))
    img = plt.imshow(a, cmap=cm)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    plt.colorbar(orientation="horizontal", cax=cax)
    plt.savefig(f"output/{name}.pdf")


# classic color bar outside of matplotlib
class cmap:
    @staticmethod
    def gradient():
        """return the functional principal's classic color bar"""
        first = int((128 * 2) - np.round(255 * (1. - 0.90)))
        second = (256 - first)

        colors1 = plt.cm.viridis(np.linspace(0.1, .98, first * 50))  # amplify 50 times to make the color more fine
        colors2 = plt.cm.YlOrBr(np.linspace(0.25, 1, second * 50))

        # combine them and build a new colormap
        cols = np.vstack((colors1, colors2))
        return clc(cols, name='gradient')

    @staticmethod
    def myelin():
        color_list = [red, orange, oran_yell, yellow, limegreen, green, blue_videen7, blue_videen9, blue_videen11,
                      purple2]
        n_fine = 1000

        return get_cm(color_list, n_fine, 'myelin')

    @staticmethod
    def psych_no_none():
        color_list = [yellow, pyell_oran, orange, poran_red, pblue, pltblue1, pltblue2, pbluecyan]
        n_fine = 1000
        return get_cm(color_list, n_fine, 'psych_no_none')

    @staticmethod
    def hot_no_black():
        return clc(plt.cm.hot(np.linspace(0.15, 1., 5000)), name='hot_no_black')

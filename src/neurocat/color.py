"""
Color utilities module for NeuroCat.

This module provides functions and classes for color manipulation, including loading predefined colors from a YAML file, creating color gradients, generating custom colormaps, and saving colormaps as images. It also includes a class for predefined colormaps inspired by classic color schemes.
"""

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

COLORS = colors  # Store colors in a dictionary instead of polluting globals


def get_transparent_cm(n_colors, cm_name):
    """
    Creates a fully transparent colormap.

    Args:
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
    """
    Converts a hex color string to RGB list.

    Args:
        hex_str (str): Hex color string, e.g., '#FFFFFF'.

    Returns:
        list: RGB values as [R, G, B].
    """
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]


def rgb_to_hex(rgb):
    """
    Converts RGB values to hex color string.

    Args:
        rgb (list): RGB values as [R, G, B].

    Returns:
        str: Hex color string, e.g., '#FFFFFF'.
    """
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient with n colors.

    Args:
        c1 (str): First hex color.
        c2 (str): Second hex color.
        n (int): Number of colors in the gradient.

    Returns:
        list: List of RGB colors in the gradient.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [np.append((1 - mix) * c1_rgb + (mix * c2_rgb), 1.) for mix in mix_pcts]
    return rgb_colors


def get_cm(color_list, n_fine, cm_name):
    """
    Creates a custom colormap from a list of colors.

    Args:
        color_list (list): List of hex colors.
        n_fine (int): Number of fine steps between colors.
        cm_name (str): Name of the colormap.

    Returns:
        ListedColormap: The custom colormap.
    """
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
    """
    Saves a colormap as an image file.

    Args:
        cm: The colormap to save.
        name (str): Name of the output file.
    """
    a = np.array([[0, 1]])
    plt.figure(figsize=(9, 1.5))
    img = plt.imshow(a, cmap=cm)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    plt.colorbar(orientation="horizontal", cax=cax)
    plt.savefig(f"{name}.pdf")


# classic color bar outside of matplotlib
class cmap:
    @staticmethod
    def gradient():
        """
        Returns the functional principal's classic color bar.

        Returns:
            ListedColormap: The gradient colormap.
        """
        first = int((128 * 2) - np.round(255 * (1. - 0.90)))
        second = (256 - first)

        colors1 = plt.cm.viridis(np.linspace(0.1, .98, first * 50))  # amplify 50 times to make the color more fine
        colors2 = plt.cm.YlOrBr(np.linspace(0.25, 1, second * 50))

        # combine them and build a new colormap
        cols = np.vstack((colors1, colors2))
        return clc(cols, name='gradient')

    @staticmethod
    def myelin():
        """
        Returns a myelin-inspired colormap.

        Returns:
            ListedColormap: The myelin colormap.
        """
        color_list = [COLORS['red'], COLORS['orange'], COLORS['oran_yell'], COLORS['yellow'], COLORS['limegreen'], COLORS['green'], COLORS['blue_videen7'], COLORS['blue_videen9'], COLORS['blue_videen11'],
                      COLORS['purple2']]
        n_fine = 1000

        return get_cm(color_list, n_fine, 'myelin')

    @staticmethod
    def psych_no_none():
        """
        Returns a psychology-inspired colormap without neutral tones.

        Returns:
            ListedColormap: The psych_no_none colormap.
        """
        color_list = [COLORS['yellow'], COLORS['pyell_oran'], COLORS['orange'], COLORS['poran_red'], COLORS['pblue'], COLORS['pltblue1'], COLORS['pltblue2'], COLORS['pbluecyan']]
        n_fine = 1000
        return get_cm(color_list, n_fine, 'psych_no_none')

    @staticmethod
    def hot_no_black():
        """
        Returns a hot colormap without black.

        Returns:
            ListedColormap: The hot_no_black colormap.
        """
        return clc(plt.cm.hot(np.linspace(0.15, 1., 5000)), name='hot_no_black')

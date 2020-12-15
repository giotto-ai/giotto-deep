"""
Created on Thu Feb 28 14:28:50 2013

Produces colormaps by curving through Lch color space, inspired by
"Lab color scale" by Steve Eddins 09 May 2006 (Updated 10 May 2006)
(BSD license)
http://www.mathworks.co.uk/matlabcentral/fileexchange/11037-lab-color-scale

Chroma should probably be limited to the RGB cube in a way that never
produces sharp angles in the curve, which appear as bands in the gradient.

Examples:

This is similar to the 'hsv' map, but isoluminant:
    lab_color_scale(hue=30, rot=1, l=75, chroma=41)

This is similar to 'cubehelix', except in Lch space instead of RGB space:
    lab_color_scale(hue=0, rot=-1.5, chroma=30)
"""

from numpy import asarray, cos, sin, vstack, array, ones, radians, linspace
from matplotlib import cm

"""
Both libraries produce similar, but not identical, results.
colormath fails more gracefully when exceeding the RGB color cube.

https://code.google.com/p/python-colormath/
http://markkness.net/colorpy/ColorPy.html
"""
try:
    from colormath.color_conversions import convert_color
    from colormath.color_objects import LabColor, sRGBColor

    def irgb_from_lab(L, a, b):
        rgb = array(convert_color(LabColor(L, a, b),
                                  sRGBColor).get_value_tuple())
        return rgb
except ImportError:
    from colorpy.colormodels import irgb_from_xyz, xyz_from_lab

    def irgb_from_lab(L, a, b):
        rgb = irgb_from_xyz(xyz_from_lab((L, a, b)))
        return rgb / 255.0


def lch_to_lab(L, c, h):
    """
    L is lightness, 0 (black) to 100 (white)
    c is chroma, 0-100 or more
    h is hue, in degrees, 0 = red, 90 = yellow, 180 = green, 270 = blue
    """
    a = c * cos(radians(h))
    b = c * sin(radians(h))
    return L, a, b


def lab_color_scale(lutsize=256, hue=0, chroma=50, rot=1/4, l=None):
    """
    Color map created by drawing arcs through the L*c*h*
    (lightness, chroma, hue) cylindrical coordinate color space, also called
    the L*a*b* color space when using Cartesian coordinates.

    Parameters
    ----------
    lutsize : int
        The number of elements in the colormap lookup table. (Default is 256.)
    hue : float
        Hue angle at which the colormap starts, in degrees.  Default 0 is
        reddish.
    chroma : float
        Chroma radius for the colormap path.  If chroma is 0, the colormap is
        grayscale.  If chroma is too large, the colors will exceed the RGB
        gamut and produce ugly bands.  Since the RGB cube is pointy at the
        black and white ends, this always clips somewhat.
    rot : float
        Number of hue rotations.  If 0, hue is constant and only lightness is
        varied. If 1, all hues are passed through once.  If 2, circle through
        all hues twice, etc.
        For counterclockwise rotation, make the value negative
    l : float
        Lightness value for constant-lightness (isoluminant) colormaps. If
        not specified, lightness is varied from 0 at minimum to 100 at maximum.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        The resulting colormap object
    """
    
    hue = linspace(hue, hue + rot * 360, lutsize)

    # or use atleast1d
    if l is None:
        L = linspace(0, 100, lutsize)
    elif hasattr(l, "__len__"):
        if len(l) == 2:
            L = linspace(l[0], l[1], lutsize)
        elif len(l) == 1:
            L = l * ones(lutsize)
        elif len(l) == lutsize:
            L = asarray(l)
        else:
            raise ValueError('lightness argument not understood')
    else:
        L = l * ones(lutsize)

    L, a, b = lch_to_lab(L, chroma, hue)

    rgbs = []
    Lab = vstack([L, a, b])
    for L, a, b in Lab.T:
        R, G, B = irgb_from_lab(L, a, b)
        rgbs.append((R, G, B))

    return cm.colors.LinearSegmentedColormap.from_list('lab_color_scale', rgbs,
                                                       lutsize)


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    dx, dy = 0.01, 0.01

    x = np.arange(-2.0, 2.0001, dx)
    y = np.arange(-2.0, 2.0001, dy)
    X, Y = np.meshgrid(x, y)
    # Matplotlib's 2-bump example shows banding and other problems clearly
    Z = X * np.exp(-X**2 - Y**2)

    plt.figure()
    plt.imshow(Z, vmax=abs(Z).max(), vmin=-abs(Z).max(),
               cmap=lab_color_scale(hue=200, chroma=30, l=[20, 90], rot=-.3)
               )
    plt.colorbar()
    plt.show()

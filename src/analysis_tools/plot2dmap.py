import aplpy
import os
import glob
import numpy
import matplotlib as mpl
from astropy.io import fits
import matplotlib.pyplot as plt
from analysis_tools.functions import sbr2nhi
from analysis_tools.functions import get_info
from matplotlib import font_manager
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.units as unit
from analysis_tools import get_beam
from pathlib import Path

def find_folder(start_path, folder_name):
    for dirpath, dirnames, filenames in os.walk(start_path):
        if folder_name in dirnames:
            return os.path.join(dirpath, folder_name)

# Call function to find folder:
folder_path = find_folder('/', 'TeX-Gyre-Heros')  # '/' is the root directory
font_dirs = [folder_path]

font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
font = 'tex gyre heros'  
mpl.rcParams['font.sans-serif'] = font
mpl.rc('mathtext', fontset='custom', it=font + ':italic')
mpl.rc('font', size=17)

class PlotFigure:
    def __init__(self, f, f2, v_min, v_max, levels, figsize1=5, 
                figsize2 = 5, labeltext="", galname="", 
                coords=None,labels=None,basecontour=None,
                x0=None, y0=None, x0kin=None, y0kin=None, 
                x_el=None, y_el=None, x_el_center=None, 
                y_el_center=None, polypoints=None, fig=None,
                levcol="white", namecol="white", subplot=[0.1, 0.1, 0.8, 0.8]):

        self.f = f
        self.f2 = f2
        self.v_min = v_min
        self.v_max = v_max
        self.levels = levels
        self.labeltext = labeltext
        self.coords = coords
        self.labels = labels
        self.basecontour = basecontour
        self.x0 = x0
        self.y0 = y0
        self.x0kin = x0kin
        self.y0kin = y0kin
        self.x_el = x_el
        self.y_el = y_el
        self.x_el_center = x_el_center
        self.y_el_center = y_el_center
        self.polypoints = polypoints
        self.figsize1 = figsize1
        self.figsize2 = figsize2
        self.galname = galname
        self.levcol = levcol
        self.namecol = namecol
        self.subplot = subplot
        self.fig = fig

    def plot(self):
        if self.fig == None:
            fig = plt.figure(figsize=(self.figsize1, self.figsize2))
        else:
            fig = self.fig
        f1 = aplpy.FITSFigure(self.f, figure=fig, subplot=self.subplot)
        f1.show_colorscale(vmin=self.v_min, vmax=self.v_max, aspect='auto', cmap=plt.cm.gray_r)
        f1.show_contour(self.f2, levels=self.levels, colors=self.levcol, linewidths=1) 
        ax = plt.gca()
        if self.x0: 
            ax.plot(self.x0, self.y0, 'x', ms=14, color="yellow")
        if self.x0kin: 
            ax.plot(self.x0kin, self.y0kin, 'x', ms=14, color="black")
        if self.x_el is not None:
            ax.plot(self.x_el, self.y_el, '-', color="black", lw=1.3)
        if self.x_el_center is not None:
            ax.plot(self.x_el_center, self.y_el_center, 'x', color="#4d4d4d", ms=14)
        ax.text(0.1, 0.8, self.labeltext, color='black', 
            fontsize=15,transform=ax.transAxes, fontweight='light')
        ax.tick_params(direction='in', length=8.7, width=1.3, pad=10)
        ax.tick_params(which='minor', length=5)
        if self.coords:
            for k in range(len(self.coords)):
                ax.text(self.coords[k][0], self.coords[k][1], list(self.labels)[k], color="maroon")
        ax.text(0.1, 0.9, self.galname, transform=ax.transAxes, color=self.namecol)
        if self.basecontour:
            ax.text(0.01, 0.05, r'$N_{\mathrm{HI}}=$ '+ r'$2^n \times {0} \times 10^{1}$'.format(self.basecontour[0], 
                        self.basecontour[1]) + r'$^{0}$'.format(self.basecontour[2]) + r'$~\mathrm{cm}^{-2}~n=(0, 1, ...)$', 
                    transform=ax.transAxes, color="white", fontsize=16)
        if self.polypoints:
            ax.plot(self.polypoints[0], self.polypoints[1], color="black")
        # f1.add_colorbar()
        # f1.colorbar.show()
        # f1.colorbar.set_pad(0.0)
        # f1.colorbar.set_axis_label_text(r'$\mathrm{\Sigma_{HI}(M_{\odot}~pc^{-2})}$')
        # f1.colorbar.set_axis_label_pad(15)
        # f1.colorbar.set_location('top')
        try:
            f1.add_beam()
            f1.beam.set_color('white')
            f1.beam.set_edgecolor('black')
        except:
            pass

# The rest of your code


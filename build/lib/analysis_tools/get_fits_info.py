from astropy.wcs import WCS
import numpy
from astropy.io import fits

class Get_info():
    def __init__(self, f):
        self.hdu = fits.open(f)[0]
        self.img = numpy.squeeze(self.hdu.data)
        try:
            self.img[self.img<=0] = numpy.nan
        except:
            pass
        self.hdr = self.hdu.header 
        self.pixsize = abs(self.hdr["CDELT1"]*3600)
        try:
            self.wcs = WCS(self.hdr)
        except:
            pass
        try: 
            self.nz, self.ny, self.nx = self.img.shape
        except:
           self.ny, self.nx = self.img.shape
        self.X0 = int(self.hdr['CRPIX1']) - 1
        self.Y0 = int(self.hdr['CRPIX2']) - 1
        if "BMIN" in self.hdr:
            self.bmin = self.hdr["BMIN"]*3600 
            self.bmaj = self.hdr["BMAJ"]*3600
        if "BPA" in self.hdr:
            self.bpa = self.hdr["BPA"]
        else:
            self.bmin = None
            self.bmaj = None

from astropy.coordinates import SkyCoord
import argparse
from astropy.wcs.utils import skycoord_to_pixel
from astropy import constants as const
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import numpy as np
from astropy.nddata import NDData
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.nddata import Cutout2D
from astropy.wcs.utils import proj_plane_pixel_scales
from pvextractor import extract_pv_slice, PathFromCenter

def get_labels(labels, wcs_hdr):
    lab = labels.keys()
    lab_coords = list()
    for j in lab:
        lab_coords.append(skycoord_to_pixel(SkyCoord(ra=labels[j]["ra"]*u.deg, dec=labels[j]["dec"]*u.deg,
                                            frame='icrs', equinox='J2000'), wcs_hdr))
    return lab_coords

def multiply_cube_by_2d(cube_file, factor_file, output_file):
    # Read the FITS data cube
    hdus = fits.open(cube_file)
    cube_data = numpy.squeeze(hdus[0].data)
    cube_header = hdus[0].header.copy()  # Make a copy of the header
    # Read the 2D factor FITS file
    hdus_factor = fits.open(factor_file)
    factor_data = numpy.squeeze(hdus_factor[0].data)

    # Ensure that the shape of the factor matches the cube's shape
    factor_3d = numpy.repeat(factor_data[numpy.newaxis, :, :], cube_data.shape[0], axis=0)

    # Multiply each channel of the cube by the factor
    result_data = cube_data * factor_3d

    # Save the result to a new FITS file
    fits.writeto(output_file, numpy.squeeze(result_data), cube_header, overwrite=True)

    print("Multiplication completed. Result saved to", output_file)

class BoxDrawer:
    """
    Usage:
    drawer = BoxDrawer('image.fits')
    drawer.draw_box(('00h42m44.3s', '+41d16m9s'), 5.0) 
    """
    def __init__(self, fits_file):
        self.hdulist = fits.open(fits_file)
        self.data = self.hdulist[0].data
        self.header = self.hdulist[0].header
        self.wcs = WCS(self.header)

    def draw_box(self, center_coords, diameter_arcmin):
        # Convert the center coordinates to pixel coordinates
        center = SkyCoord(center_coords[0], center_coords[1])
        center_pix = self.wcs.world_to_pixel(center)

        # Convert diameter to pixels
        diameter_deg = diameter_arcmin / 60.0
        diameter_pix = diameter_deg / abs(self.header['CDELT1'])

        # Calculate the corners of the box
        ll_x = center_pix[0] - diameter_pix / 2.0
        ll_y = center_pix[1] - diameter_pix / 2.0
        ur_x = center_pix[0] + diameter_pix / 2.0
        ur_y = center_pix[1] + diameter_pix / 2.0

        # Create a new figure and show the FITS image
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=self.wcs)
        ax.imshow(self.data, origin='lower')

        # Draw the box
        rect = plt.Rectangle((ll_x, ll_y), diameter_pix, diameter_pix, 
                             edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        plt.show()

class EllipseDrawer:
    """
    drawer = EllipseDrawer('image.fits')
    drawer.draw_ellipse(('00h42m44.3s', '+41d16m9s'), 5.0, 3.0, 45.0) 
    """
    def __init__(self, fits_file):
        self.hdulist = fits.open(fits_file)
        self.data = self.hdulist[0].data
        self.header = self.hdulist[0].header
        self.wcs = WCS(self.header)

    def draw_ellipse(self, center_coords, major_axis_arcmin, minor_axis_arcmin, pos_angle_deg):
        # Convert the center coordinates to pixel coordinates
        center = SkyCoord(center_coords[0], center_coords[1])
        center_pix = self.wcs.world_to_pixel(center)

        # Convert major and minor axes to pixels
        major_axis_deg = major_axis_arcmin / 60.0
        minor_axis_deg = minor_axis_arcmin / 60.0
        major_axis_pix = major_axis_deg / abs(self.header['CDELT1'])
        minor_axis_pix = minor_axis_deg / abs(self.header['CDELT1'])

        # Create a new figure and show the FITS image
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=self.wcs)
        ax.imshow(self.data, origin='lower')

        # Draw the ellipse
        ellipse = Ellipse(center_pix, major_axis_pix, minor_axis_pix, angle=pos_angle_deg, 
                          edgecolor='red', facecolor='none')
        ax.add_patch(ellipse)

        plt.show()

class LineDrawer:
    """
    drawer = LineDrawer('image.fits')
    drawer.draw_line(('00h42m44.3s', '+41d16m9s'), 5.0, 45.0) 
    """
    def __init__(self, fits_file):
        self.hdulist = fits.open(fits_file)
        self.data = self.hdulist[0].data
        self.header = self.hdulist[0].header
        self.wcs = WCS(self.header)

    def draw_line(self, center_coords, diameter_arcmin, pos_angle_deg):
        # Convert the center coordinates to pixel coordinates
        center = SkyCoord(center_coords[0], center_coords[1])
        center_pix = self.wcs.world_to_pixel(center)

        # Convert diameter to pixels
        diameter_deg = diameter_arcmin / 60.0
        radius_pix = diameter_deg / (2 * abs(self.header['CDELT1']))

        # Calculate the line endpoints
        angle_rad = np.deg2rad(pos_angle_deg)
        dx = radius_pix * np.cos(angle_rad)
        dy = radius_pix * np.sin(angle_rad)
        x1 = center_pix[0] - dx
        y1 = center_pix[1] - dy
        x2 = center_pix[0] + dx
        y2 = center_pix[1] + dy

        # Create a new figure and show the FITS image
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=self.wcs)
        ax.imshow(self.data, origin='lower')

        # Draw the line
        line = Line2D([x1, x2], [y1, y2], color='red')
        ax.add_line(line)

        plt.show()

def delheader(hdr, op="4"):
    delhdr4 = ["CUNIT4", "CDELT4", "CRVAL4", "CRPIX4", "CTYPE4", "CDELT4", 
               "CRVAL4", "CRPIX4", "CTYPE4", "NAXIS4"]
    delhdr3 = ["CUNIT3", "CDELT3", "CRVAL3", "CRPIX3", "CTYPE3", "CDELT3", 
               "CRVAL3", "CRPIX3", "CTYPE3", "NAXIS3"]
    delhdr = {"4": delhdr4, "3": delhdr3}
    hdrc = hdr.copy()
    for key in delhdr[op]:
        if key in hdrc.keys():
            del hdrc[key]
    if op=="4":
        hdrc["NAXIS"] = 3
    else:
        hdrc["NAXIS"] = 2
    return hdrc
            
class FitsCutter:
    """ 
    fopt = fitsfile.fits
    cutter = FitsCutter(fopt)
    cutter.cut(radec_hcg90["ra"], radec_hcg90["dec"], 10, fopt[:-5]+'_cut.fits')   
    """
    def __init__(self, fits_file):
        # Load the FITS file
        self.fits_fitle = fits_file
        self.hdulist = fits.open(fits_file)
        self.data = np.squeeze(self.hdulist[0].data)
        self.header = self.hdulist[0].header
        self.wcs = WCS(self.header)

    def cut2D(self, ra, dec, size_arcmin, output_fits):
        # Create a SkyCoord object for the center
        center = SkyCoord(ra, dec, unit=(u.deg, u.deg))

        # Convert the size from arcmin to pixels
        size_deg = size_arcmin / 60.0
        size_pixels = size_deg / abs(self.header['CDELT1'])
        size_pixels = int(size_pixels)
        # Create a Cutout2D object
        cutout = Cutout2D(self.data, center, (size_pixels, size_pixels), wcs=self.wcs)

        # Create a new FITS HDU
        hdu = fits.PrimaryHDU(data=cutout.data, header=cutout.wcs.to_header())

        # Write the cutout to a new FITS file
        hdu.writeto(output_fits, overwrite=True)

    def cut_fits2D(self, center_ra, center_dec, radius_arcmin, output_file):
        # Open the FITS file and retrieve the data and WCS information
        data = np.squeeze(self.hdulist[0].data)
        wcs = WCS(self.header)

        # Define the center position
        center = SkyCoord(center_ra, center_dec, unit='deg', frame='icrs')

        # Calculate the cutout size in pixels based on the given radius
        pixel_scale = wcs.pixel_scale_matrix[1, 1] * 3600  # Convert degrees/pixel to arcseconds/pixel
        size_pixels = int(radius_arcmin * 60 / pixel_scale)

        # Create the cutout using the Cutout2D function
        cutout = Cutout2D(data, center, size_pixels, wcs=wcs)

        # Update the FITS header with the cutout WCS information
        new_header = cutout.wcs.to_header()
        new_header.update(cutout.wcs.to_header())

        # Create a new HDU with the cutout data and header
        hdu = fits.PrimaryHDU(cutout.data, new_header)

        # Save the cutout to a new FITS file
        hdu.writeto(output_file, overwrite=True)

        # Close the original FITS file
        self.hdulist.close()

    def cut3D(self, center_ra, center_dec, radius_arcmin, chan_start, chan_end, output_file):
        # Drop the third axis
        """
        fopt = 'fitsfile.fits'
        cutter = FitsCutter(fopt)
        cutter.cut3D(radec_hcg90["ra"], radec_hcg90["dec"], 10, 100, 200, fopt[:-5]+'_cut.fits')
        """
        # Define the center position
        center = SkyCoord(center_ra, center_dec, unit='deg', frame='icrs')

        # Calculate the cutout size in pixels based on the given radius
        pixel_scale = self.wcs.pixel_scale_matrix[1, 1] * 3600  # Convert degrees/pixel to arcseconds/pixel
        size_pixels = int(radius_arcmin * 60 / pixel_scale)

        # Create an empty array to hold the cutout data
        cutout_data = []
        newhed = delheader(self.header)
        # Iterate over the image planes in the 3D data within the channel range
        for i in range(chan_start, chan_end):
            wcs = WCS(newhed)
            wcs_2d = wcs.dropaxis(dropax=2)  # Drop the third axis
            # Create the cutout using the Cutout2D function
            cutout = Cutout2D(self.data[i, :, :], center, size_pixels, wcs=wcs_2d)
            # Append the cutout data to the list
            cutout_data.append(cutout.data)

        # Update the FITS header with the cutout WCS information
        new_header = cutout.wcs.to_header()
        new_header.update(self.header)
        new_header['CRPIX1'] = cutout.wcs.wcs.crpix[0]
        new_header['CRPIX2'] = cutout.wcs.wcs.crpix[1]
        if self.header["CDELT3"] < 0 :
            hdrlist = np.arange(self.header["CRVAL3"], -3.21E+09, self.header["CDELT3"])
        else:
            hdrlist = np.arange(self.header["CRVAL3"], 3.21E+09, self.header["CDELT3"])
        new_header['CRVAL3'] = hdrlist[chan_start]
        print(new_header['CRVAL3'])
        # Create a new HDU with the cutout data and header
        hdu = fits.PrimaryHDU(data=cutout_data, header=new_header)

        # Save the cutout to a new FITS file
        hdu.writeto(output_file, overwrite=True)

        # Close the original FITS file
        self.hdulist.close()

    def cut_circle3D(self, center_ra, center_dec, radius_arcmin, output_file):
        # Open the FITS file and retrieve the data and WCS information
        data = np.squeeze(self.hdulist[0].data)
        header = self.hdulist[0].header
        hdrc = delheader(header)
        wcs = WCS(hdrc)
    
        # Define the center position
        center = SkyCoord(center_ra, center_dec, unit='deg', frame='icrs')
    
        # Calculate the cutout size in pixels based on the given radius
        pixel_scale = wcs.pixel_scale_matrix[1, 1] * 3600  # Convert degrees/pixel to arcseconds/pixel
        size_pixels = int((radius_arcmin) * 60 / pixel_scale)
    
        # Create an empty array to hold the cutout data
        cutout_data = []
        # Iterate over the image planes in the 3D data
        for i, plane in enumerate(data):
            wcs_2d = wcs.dropaxis(dropax=2)  # Drop the third axis
            # Create the cutout using the Cutout2D function
            cutout = Cutout2D(plane, center, size_pixels, wcs=wcs_2d)
    
            # Generate a mask for pixels outside of the desired radius
            y, x = np.ogrid[0:cutout.shape[0], 0:cutout.shape[1]]
            mask = (x - cutout.shape[1]/2)**2 + (y - cutout.shape[0]/2)**2 > (size_pixels/2)**2
            cutout.data[mask] = np.nan
    
            # Append the cutout data to the list
            cutout_data.append(cutout.data)
        new_header = cutout.wcs.to_header()
        new_header.update(self.header)
        new_header['CRPIX1'] = cutout.wcs.wcs.crpix[0]
        new_header['CRPIX2'] = cutout.wcs.wcs.crpix[1]
        # Update the FITS header with the cutout WCS information
        #newheader.update(cutout.wcs.to_header())
    
        # Create a new HDU with the cutout data and header
        hdu = fits.PrimaryHDU(data=cutout_data, header=new_header)
    
        # Save the cutout to a new FITS file
        hdu.writeto(output_file, overwrite=True)
    
        # Close the original FITS file
        self.hdulist.close()

    def cut_circle2D(self, center_ra, center_dec, radius_arcmin, output_file):
        # Define the center position
        center = SkyCoord(center_ra, center_dec, unit='deg', frame='fk5')

        # Calculate the cutout size in pixels based on the given radius
        #pixel_scale = self.wcs.pixel_scale_matrix[1, 1] * 3600  # Convert degrees/pixel to arcseconds/pixel
        pixel_scale = np.mean(proj_plane_pixel_scales(self.wcs))*3600.0  # in arcsec
        size_pixels = int(radius_arcmin * 60 / pixel_scale)

        # Create the cutout using the Cutout2D function
        cutout = Cutout2D(self.data, center, size_pixels, wcs=self.wcs)

        # Generate a mask for pixels outside of the desired radius
        y, x = np.ogrid[0:cutout.shape[0], 0:cutout.shape[1]]
        mask = (x - cutout.shape[1]/2)**2 + (y - cutout.shape[0]/2)**2 > (size_pixels/2)**2
        cutout.data[mask] = np.nan

        # Update the FITS header with the cutout WCS information
        new_header = cutout.wcs.to_header()
        new_header.update(self.header)
        new_header['CRPIX1'] = cutout.wcs.wcs.crpix[0]
        new_header['CRPIX2'] = cutout.wcs.wcs.crpix[1]
        # Create a new HDU with the cutout data and header
        hdu = fits.PrimaryHDU(data=cutout.data, header=new_header)

        # Save the cutout to a new FITS file
        hdu.writeto(output_file, overwrite=True)

        # Close the original FITS file
        self.hdulist.close()

def sky2pix(input_file, ra_str, dec_str):
    # Open the FITS file and retrieve the WCS information
    hdulist = fits.open(input_file)
    wcs = WCS(hdulist[0].header)

    # Create a SkyCoord object for the input sky coordinates
    sky_coord = SkyCoord(ra_str, dec_str, unit='deg', frame='icrs')

    # Convert sky coordinates to pixel coordinates
    pixel_coord = sky_coord.to_pixel(wcs)

    # Print the pixel coordinates
    print(f"RA: {pixel_coord[0]:.2f}")
    print(f"Dec: {pixel_coord[1]:.2f}")

    # Close the FITS file
    hdulist.close()

if __name__ == '__main__':
    # Create the command-line argument parser
    parser = argparse.ArgumentParser(description='Convert sky coordinates to pixel coordinates.')
    parser.add_argument('-f', '--input_file', type=str, help='Input FITS file')
    parser.add_argument('-ra', '--ra', type=str, help='Right Ascension in format "hh:mm:ss.s"')
    parser.add_argument('-dec', '--dec', type=str, help='Declination in format "dd:mm:ss.s"')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the sky2pix function with the provided arguments
    sky2pix(args.input_file, args.ra, args.dec)


class GraphSettings:
    """ gs = GraphSettings(ax, 8.7)
        gs.apply_settings() 
    """
    def __init__(self, ax, xlab=" ", ylab=" ", major_length=8.7, 
            minor_length=5, tickslabelsize=12, xlabelpad=20, ylabelpad=25, xfontsize=14, yfontsize=14):
        self.ax = ax
        self.major_length = major_length
        self.minor_length = minor_length
        self.labelsize = tickslabelsize
        self.secax = self.ax.secondary_xaxis('top')
        self.secax.tick_params(labeltop=False)
        self.secaxr = self.ax.secondary_yaxis('right')
        self.secaxr.tick_params(labelright=False)
        self.ax.set_xlabel(xlab, labelpad=xlabelpad, fontsize=xfontsize)
        self.ax.set_ylabel(ylab, labelpad=ylabelpad, fontsize=yfontsize)

    def apply_settings(self):
        for axes in [self.ax, self.secax, self.secaxr]:
            axes.minorticks_on()
            axes.tick_params(which='both', direction='in', length=self.major_length, width=1, pad=10)
            axes.tick_params(which='minor', length=self.minor_length)
            axes.tick_params(labelsize=self.labelsize)

def radec2deg(ra, dec):
    """
    Usage:
    pos_ngc7173 = ("22:02:03.38", "-31:58:26.92")
    ra_ngc7173 = radec2deg(pos_ngc7173[0], pos_ngc7173[1]) 
    """
    # Create a SkyCoord object
    sky_coord = SkyCoord(ra, dec, unit=('hourangle', 'deg'))

    # Extract the RA and DEC in degrees
    ra_deg = sky_coord.ra.deg
    dec_deg = sky_coord.dec.deg

    return {"ra":ra_deg, "dec":dec_deg}

def delkeys(todel, hdr):
    for j in todel:
        if j in hdr.keys():
            del hdr[j]

class Get_infomom():
    def __init__(self, f):
        self.hdu = fits.open(f)[0]
        self.img = np.squeeze(self.hdu.data)
        try:
            self.img[self.img<=0] = np.nan
        except:
            pass
        self.hdr = self.hdu.header
        try:
            self.pixsize = abs(self.hdr["CDELT1"]*3600)
        except:
            pass
        self.wcs = WCS(self.hdr)
        try: 
            self.nz, self.ny, self.nx = self.img.shape
        except:
            self.ny, self.nx = self.img.shape
        if 'CRPIX1' in self.hdr:
            self.X0 = int(self.hdr['CRPIX1']) - 1
            self.Y0 = int(self.hdr['CRPIX2']) - 1
        if "BMIN" in self.hdr:
            self.bmin = self.hdr["BMIN"]*3600 
            self.bmaj = self.hdr["BMAJ"]*3600
        else:
            self.bmin = None
            self.bmaj = None

def chan2freq(channels, fits_name):
    """Convert channels to frequencies.

    :param channels: which channels to convert
    :type channels: Iterable[int]
    :param fits_name: name of the FITS file
    :type fits_name: str
    :return: frequencies
    :rtype: Iterable[float]
    """
    header = fits.getheader(fits_name)
    # Don't know how to deal with cubelets having diff CRPIX3 from orig data; catalog data is in ref to orig base 0
    frequencies = (header['CDELT3'] * (channels - (header['CRPIX3'] - 1)) + header['CRVAL3']) * u.Hz
    # frequencies = (header['CDELT3'] * channels + header['CRVAL3']) * u.Hz
    return frequencies


def chan2vel(channels, fits_name):
    """Convert channels to velocities.

    N.B.: This assumes the channels have uniform width in velocity space,
          which may not be the case!

    :param channels: the channels to convert
    :type channels: Iterable[int]
    :param fits_name: name of the FITS file
    :type fits_name: str
    :return: calculated velocities
    :rtype: Iterable[float]
    """
    print("\tWARNING: Assuming channels are uniform width in velocity.")
    header = fits.getheader(fits_name)
    # Don't know how to deal with cubelets having diff CRPIX3 from orig data; catalog data is in ref to orig base 0
    velocities = (header['CDELT3'] * (channels - (header['CRPIX3'] - 1)) + header['CRVAL3']) * u.m / u.s
    # velocities = (header['CDELT3'] * channels + header['CRVAL3']) * u.m / u.s
    return velocities


def felo2vel(channels, fits_name):
    """Converts channels to velocities for a cube with non-linear channels.

    N.B.: This conversion differs from the output of SoFiA-2 which uses wcslib and therefore may not be strictly correct.

    :param channels:
    :type channels: Iterable[int]
    :param fits_name:
    :type fits_name: str
    :return: calculated velocities
    :rtype: Iterable[float]
    """
    # Formula taken from here: https://www.astro.rug.nl/software/kapteyn/spectralbackground.html#aips-axis-type-felo
    print("\tWARNING: Axis type FELO...this conversion may not be precise (may be off by ~10 km/s).")
    c = const.c.to(u.m/u.s).value
    header = fits.getheader(fits_name)
    fr = header['RESTFREQ'] / (1 + header['CRVAL3'] / c)
    df = -1 * header['RESTFREQ'] * header['CDELT3'] * c / ((c + header['CRVAL3']) * (c + header['CRVAL3']))
    velocities = header['CRVAL3'] + c * header['RESTFREQ'] * (1 / (fr + (channels - header['CRPIX3']) * df) - 1 / fr)
    return velocities


def sbr2nhi(sbr, bunit, bmaj, bmin):
    """Get the HI column density from sbr.

    :param sbr: SBR
    :type sbr: float
    :param bunit: unit in which sbr is measured
    :type bunit: str
    :param bmaj: major axis of the beam
    :type bmaj: float
    :param bmin: minor axis of the bea,
    :type bmin: float
    :return: column density
    :rtype: float
    """
    # NEED TO ADD UNITS THAT ANNOYINGLY COME OUT OF SPECTRAL CUBE! DONE?
    if (bunit == 'Jy/beam*m/s') or (bunit == 'Jy/beam*M/S'):
      nhi = 1.104e+21 * sbr / bmaj / bmin
    elif (bunit == 'Jy/beam*Hz') or (bunit == 'beam-1 Jy*Hz'):
      nhi = 2.330e+20 * sbr / bmaj / bmin
    else:
      print("\tWARNING: Mom0 imag units are not Jy/beam*m/s or Jy/beam*Hz. Cannot convert to HI column density.")
      nhi = sbr
    nhi_ofm = np.int(np.floor(np.log10(np.abs(nhi))))
    nhi_label = '$N_\mathrm{{HI}}$ = {0:.1f} x $10^{{ {1:d} }}$ cm$^{{-2}}$'.format(nhi/10**nhi_ofm, nhi_ofm)
    nhi_labels = '$N_\mathrm{{HI}}$ = $2^n$ x {0:.1f} x $10^{{ {1:d} }}$ cm$^{{-2}}$ ($n$=0,1,...)'.format(nhi/10**nhi_ofm, nhi_ofm)
    return nhi, nhi_label, nhi_labels


def get_info(fits_name, beam=None):
    """Get the beam info from a FITS file.

    :param fits_name: name of the FITS file
    :type fits_name: str
    :param beam: beam specifications, defaults to None. Specifications are
        given in arcsec (axes) and degrees (position_angle), and formatted as
        {[major_axis, minor_axis, position_angle]|[major_axis, minor_axis]|
        [position_angle]}
    :type beam: Iterable[float], optional
    :return: The characteristics of the beam and coordinate system of the image.
    :rtype: dict
    """

    # For FITS conventions on the equinox, see:
    # https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf

    header = fits.getheader(fits_name)
    hdu = fits.open(fits_name)[0]
    img = hdu.data
    try:
        nz, ny, nx = np.squeeze(img).shape
    except:
        ny, nx = np.squeeze(img).shape
        nz = 0
    cellsize = header['CDELT2'] * 3600. * u.arcsec

    try:
       bmaj = header['BMAJ'] * 3600. * u.arcsec
       bmin = header['BMIN'] * 3600. * u.arcsec
       bpa = header['BPA']
       print(f"\tFound {bmaj:.1f} by {bmin:.1f} beam with PA={bpa:.1f} deg in primary header.")
    except:
       print("\tWARNING: Couldn't find beam in primary header information; in other extension? "
             "Assuming beam is 3.5x3.5 pixels"
             "\n\t\tColumn density and beam plotted as order of magnitude estimate ONLY. "
             "\n\t\tRerun with -b and provide beam info to remove red strikethroughs on plots.")
       bmaj, bmin, bpa = 3.5 * cellsize, 3.5 * cellsize, 0
       default_beam = True
    default_beam = False
    beam = (bmaj, bmin, bpa)
    if len(beam) == 3:
        print(f"\tUsing user specified beam: {beam[0]} arcsec by {beam[1]} arcsec; PA: {beam[2]} deg")
        bmaj = beam[0] * u.arcsec
        bmin = beam[1] * u.arcsec
        bpa = beam[2]
    elif len(beam) == 2:
        print(f"\tWARNING: assuming PA = 0. Using user specified beam: {beam[0]} arcsec by {beam[1]} arcsec.")
        bmaj = beam[0] * u.arcsec
        bmin = beam[1] * u.arcsec
        bpa = 0
    elif len(beam) == 1:
        print(f"\tWARNING: using user specified circular beam size of {beam[0]} arcsec.")
        bmaj = bmin = beam[0] * u.arcsec
        bpa = 0

    pix_per_beam = bmaj / cellsize * bmin / cellsize * np.pi / (4 * np.log(2))

    # Try catching cubes in Galactic coordinates first
    if 'GLON' in header['CTYPE1']:
        print("\tFound data is in Galactic spatial frame.")
        equinox = None
        frame = 'galactic'
    # If not Galacticc, try to determine the equinox of the observations
    else:
        try:
            equinox = header['EQUINOX']
            if equinox < 1984.0:
                equinox = 'B' + str(equinox)
                frame = 'fk4'
            else:
                equinox = 'J' + str(equinox)
                frame = 'fk5'
            print("\tFound {} equinox in header.".format(equinox))
        except KeyError:
            try:
                equinox = header['EPOCH']
                if equinox < 1984.0:
                    equinox = 'B' + str(equinox)
                    frame = 'fk4'
                else:
                    equinox = 'J' + str(equinox)
                    frame = 'fk5'
                print("\tWARNING: Using deprecated EPOCH in header for equinox: {}.".format(equinox))
            except KeyError:
                print("\tWARNING: No equinox information in header; assuming ICRS frame.")
                equinox = None
                frame = 'icrs'

    # Try to determine the reference frame.  AIPS conventions use VELREF: http://parac.eu/AIPSMEM117.pdf
    spec_sys = False
    try:
        spec_sys = header['SPECSYS']
        print("\tFound {} reference frame specified in SPECSYS in header.".format(spec_sys))
    except:
        try:
            velref = header['VELREF']
            if velref == 1: spec_sys = 'LSR'
            if velref == 2: spec_sys = 'HELIOCEN'
            if velref == 3: spec_sys = 'TOPOCENT'
            print("\tDerived {} reference frame from VELREF in header using AIPS convention.".format(spec_sys))
        except:
            # Comment this message out for now...program checks later.
            # print("\tNo SPECSYS or VELREF in header to define reference frame, checking CTYPE3.")
            pass

    # Try to determine the spectral properties
    if fits_name[-9:] != 'cube.fits':
        print("\tWARNING: Retrieving info from a moment map or other 2D image?")
        chan_width = None
        spec_axis = None

    else:
        spec_axis = header['CTYPE3']
        chan_width = header['CDELT3']
        if 'FREQ' in spec_axis:
            units = u.Hz
        else:
            units = u.m / u.s
        chan_width = chan_width * units

        print("\tFound CTYPE3 spectral axis type {} in header.".format(spec_axis))
        if ("-" in spec_axis) and spec_sys:
            print("\tWARNING: dropping end of spectral axis type. Using SPECSYS/VELREF for reference frame.")
            spec_axis = spec_axis.split ("-")[0]
        elif ("-" in spec_axis) and (not spec_sys):
            spec_sys = spec_axis.split("-")[1]
            spec_axis = spec_axis.split("-")[0]
            if spec_sys == 'HEL': spec_sys = 'HELIOCEN'
            print("\tWARNING: attempting to use end of CTYPE3 for reference frame: {}".format(spec_sys))

    if not spec_sys:
        print("\tNo SPECSYS, VELREF, or reference frame in CTYPE3, assuming data in TOPOCENT reference frame.")
        spec_sys = 'TOPOCENT'

    return {'bmaj': bmaj, 'bmin': bmin, 'bpa': bpa, 'pix_per_beam': pix_per_beam, 'default_beam': default_beam,
            'chan_width': chan_width, 'equinox': equinox, 'frame': frame, 'cellsize': cellsize, 'spec_sys': spec_sys,
            'spec_axis': spec_axis, 'img': img, 'hdr': header, 'nz': nz, 'ny': ny, 'nx': nx}


def get_radecfreq(catalog, original):
    """Get the right ascension, declination, and frequeny of a catalog object.

    :param catalog: catalog object header
    :type catalog: astropy.Header? TODO check in function calls
    :param original: name of the original file
    :type original: str
    :return: right ascension, declination, and frequency
    :rtype: tuple
    """

    header = fits.getheader(original)
    wcs = WCS(header)
    # Get the x,y-position of the catalog object
    Xc = catalog['x']
    Yc = catalog['y']
    if header['NAXIS'] == 3:
        subcoords = wcs.wcs_pix2world(Xc, Yc, 1, 0)   # origin follows: spatial, spectral, stokes?
    if header['NAXIS'] == 4:
        subcoords = wcs.wcs_pix2world(Xc, Yc, 1, 0, 0)
    ra, dec, freq = subcoords[0], subcoords[1], subcoords[2]

    return ra, dec, freq

def get_subcube(source, original):
    """Retrieve a subcube from a datacube

    :param source: source object
    :type source: Astropy table
    :param original: original data file
    :type original: str
    :return: subcube of data
    :rtype: NDArray
    """

    hdu_orig = fits.open(original)

    if hdu_orig[0].header['NAXIS'] == 4:
        stokes_dim, z_dim, y_dim, x_dim = 0, 1, 2, 3
    if hdu_orig[0].header['NAXIS'] == 3:
        z_dim, y_dim, x_dim = 0, 1, 2

    # Some lines stolen from cubelets in  SoFiA:
    # Could consider allowing a user specified range in z.
    cubeDim = hdu_orig[0].data.shape
    Xc = source['x']
    Yc = source['y']
    Xmin = source['x_min']
    Ymin = source['y_min']
    Xmax = source['x_max']
    Ymax = source['y_max']
    cPixXNew = int(Xc)
    cPixYNew = int(Yc)
    maxX = 2 * max(abs(cPixXNew - Xmin), abs(cPixXNew - Xmax))
    maxY = 2 * max(abs(cPixYNew - Ymin), abs(cPixYNew - Ymax))
    XminNew = cPixXNew - maxX
    if XminNew < 0: XminNew = 0
    YminNew = cPixYNew - maxY
    if YminNew < 0: YminNew = 0
    XmaxNew = cPixXNew + maxX
    if XmaxNew > cubeDim[x_dim] - 1: XmaxNew = cubeDim[x_dim] - 1
    YmaxNew = cPixYNew + maxY
    if YmaxNew > cubeDim[y_dim] - 1: YmaxNew = cubeDim[y_dim] - 1

    if len(cubeDim) == 4:
        subcube = hdu_orig[0].data[0, :, int(YminNew):int(YmaxNew) + 1, int(XminNew):int(XmaxNew) + 1]
    elif len(cubeDim) == 3:
        subcube = hdu_orig[0].data[:, int(YminNew):int(YmaxNew) + 1, int(XminNew):int(XmaxNew) + 1]
    else:
        print("WARNING: Original cube does not have 3-4 dimensions.")
        subcube = None

    hdu_orig.close()

    return subcube


def create_pv(source, filename, opt_view=6*u.arcmin, min_axis=False):
    """

    :param source: source object
    :type source: Astropy table
    :param filename: name of FITS file
    :type filename: str
    :param opt_view: requested size of the image for regriding
    :type opt_view: quantity
    :param min_axis: flag for extracting major or minor axis
    :type min_axis: boolean
    :return: position-velocity slice of the mask cube
    :rtype: FITS HDU
    """

    pos_angle = source['kin_pa']
    if min_axis == True:
        pos_angle += 90.
    slice = PathFromCenter(center=SkyCoord(ra=source['pos_x'], dec=source['pos_y'], unit='deg'),
                           length=opt_view, angle=pos_angle*u.deg, width=6*u.arcsec)
    mask = fits.open(filename)
    try:
        mask_pv = extract_pv_slice(mask[0].data, slice, wcs=WCS(mask[0].header, fix=True, translate_units='shd'))
    except ValueError:
        print('\tWARNING: pvextractor is complaining about non-square pixels, try with assert_square = False')
        try:
            mask_pv = extract_pv_slice(mask[0].data, slice, wcs=WCS(mask[0].header, fix=True, translate_units='shd'),
                                       assert_square=False)
        except:
            print('\tERROR: Cannot extract pv slice of mask. Try upgrading to latest version of pvextractor (v>=0.4) from github:\n'
                  '\t\t"python3 -m pip install git+https://github.com/radio-astro-tools/pvextractor"')
            mask_pv = None
    mask.close()

    return mask_pv


def plot_labels(source, ax, default_beam, x_color='k'):
    """Plot labels on spatial plots depending on the coordinate frame.

    :param source: source object
    :type source: Astropy table
    :param ax: matplotlib axes instance
    :type ax: axes object
    :param default_beam: whether the synthesized beam is known from data/user or not
    :type default_beam: bool
    :param x_color: color of galaxy position marker
    :type x_color: str
    :return:
    """

    if 'l' in source.colnames:
        x_coord, y_coord = 'glon', 'glat'
        # x_label, y_label = 'Galactic Longitude [deg]', 'Galactic Latitude [deg]'
        x_label, y_label = '$\it{{l}}$ [deg]', '$\it{{b}}$ [deg]'
    else:
        x_coord, y_coord = 'ra', 'dec'
        x_label, y_label = 'RA (ICRS)', 'Dec (ICRS)'

    ax.scatter(source['pos_x'], source['pos_y'], marker='x', c=x_color, linewidth=0.75,
               transform=ax.get_transform('world'))
    ax.set_title(source['name'], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.coords[x_coord].set_axislabel(x_label, fontsize=20)
    ax.coords[y_coord].set_axislabel(y_label, fontsize=20)
    if default_beam:
        ax.scatter(0.92, 0.9, marker='x', c='red', s=500, linewidth=5, transform=ax.transAxes, zorder=99)
        ax.plot([0.1, 0.9], [0.05, 0.05], c='red', linewidth=3, transform=ax.transAxes, zorder=100)
        ax.text(0.5, 0.5, 'Not calculated with correct beam', transform=ax.transAxes, fontsize=40, color='gray',
                alpha=0.5, ha='center', va='center', rotation='30', zorder=101)

    return

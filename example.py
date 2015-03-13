from compute_gvirial import compute_gvirial
from astropy.io import fits
import numpy as np


# openging the file
file_in = "ngc1333.fits"
file_out = 'ngc1333_gvirial.fits'
hdulist = fits.open(file_in)
header = hdulist[0].header
data = hdulist[0].data


# parameters
distance = 250     # in pc
c0 = 1e5           # in cm
npad_fft = 3


dx = abs(float(header['CDELT1']) * 1.4959787e13 * 3600.0 * distance)
dv = abs(float(header['CDELT3'])) * 1e2

# the factor 1.0e5 comes from the conversion between km/s and cm/s
# the factor 5e20 converts Tb into column density
k_fco = 1.0 / 1.0e5 * 5.0e20 * 3.34e-24


# compute the density distribution. for each vorxel, the value at
#       data_dens[v, y, x]
# stands for column density at position (x, y) and within the velocity interval
# (v - 0.5 * dv, v + 0.5 * dv) where dv is the velocity resolution of the cube.
# the unit is g cm^{-2}.
# this can be achieved (in the case of CO(1-0)) by multiplying Tb with a
# conversion factor)
data_dens = data * k_fco

# compute the G-virial
data_gvirial = compute_gvirial(data_dens, dx, dv, c0, npad_fft)

# updating the header
hdulist[0].data = data_gvirial
hdulist[0].header.update('BUNIT', 'G M/(v^2 r) == 1')
hdulist[0].header.update('DATAMIN', np.nanmin(hdulist[0].data))
hdulist[0].header.update('DATAMAX', np.nanmax(hdulist[0].data))

# writing the output
hdulist[0].writeto(file_out)

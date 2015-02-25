#!/opt/local/bin/python
import numpy as np
from math import pi
from numpy import fft
import sys
import itertools
from astropy.io import fits

G = 6.67e-8


def compute_gvirial(data_dens, dxy, dv, c0=1e5, npad_fft=3):
    """
    Compute the g-virial based on mass distribution

    input:
           data_mass: column density distribution in the
               position-position-velocity space.
             The input data is 3D array,
               and the axes are arranged as (v, y, x).

           dxy, dv: the spearation of vorxels in the spatial and the velocity
             direction.

           c0: a parameter in the calculation. Larger c0 leads to results that
             are more smoothed in the v direction. c0 should be chosen to be
             comparable with the sound speed. Default, 1e5 (cm/s).


           npad_fft: zero-padding for the FFT. Larger value leads to better
             behaviors at the edges, but requires more computational time and
             memory. Default, 3.


    output:
           G-virial in position-velocity space.

    units: data_dens has the unit of mass (g).
           c0 has the unit of cm/s.
    """
    data_dens[np.where(np.isnan(data_dens))] = 0.0
    data_dens[np.where(np.isinf(data_dens))] = 0.0

    nv = len(data_dens)

    data_dens_result = data_dens.copy().astype(complex)

    omagex = np.fft.fftfreq(len(data_dens[0][0])
                            * npad_fft, d=dxy).reshape(1,
                                                       len(data_dens[0][0])
                                                       * npad_fft)
    omegay = np.fft.fftfreq(len(data_dens[0])
                            * npad_fft, d=dxy).reshape(len(data_dens[0])
                                                       * npad_fft, 1)

    omagexy = np.sqrt(omagex * omagex + omegay * omegay)
    omagexy[0, 0] = 0.5 * (omagexy.min() + omagexy.max())  # to avoid infinity

    print "computing G-virial"
    for i in range(len(data_dens)):
        percentage = int(i * 100 / len(data_dens))
        sys.stdout.write("\r%d%%" % percentage)
        sys.stdout.flush()

        shape_orig = np.array(data_dens[0].shape)
        shape_padded = np.array(data_dens[0].shape) * npad_fft
        padded_map = np.zeros(shape_padded)

        padded_map[0: shape_orig[0], 0: shape_orig[1]] = data_dens[i]
        t_xy = fft.fftn(padded_map)
        t_xy_p = t_xy / omagexy
        phi_xy = fft.ifftn(t_xy_p)

        data_dens_result[i]\
            = phi_xy[0: shape_orig[0], 0: shape_orig[1]]

    omegav = np.fft.fftfreq(npad_fft * nv, d=dv)
    omegavf = pi * np.exp(-c0 * np.abs(omegav) * 2 * pi) / c0
    for x, y in itertools.product(xrange(len(data_dens_result[0][0])),
                                  xrange(len(data_dens_result[0]))):
        spec_line = omegav.copy() * 0
        spec_line[0: nv] = data_dens_result[:, y, x].real
        spec_k = fft.fft(spec_line) * omegavf
        line = fft.ifft(spec_k)
        data_dens_result[:, y, x] = line[0: 0 + nv]

    data_gvirial = data_dens_result * G

    return data_gvirial.real


if __name__ == "__main__":
    print """
    Function:
        Calculate the G-virial for a given data cube.
    Usage:
        python cal_gvirial_pub.py file_in, file_out, distance, cs, npad_fft

        file_in : input file, a PPV data cube in FITs format.
        file_out: name of the output file.
        distance: distance of the object in pc.
        c0      : smoothing velocity.
        npad_fft: a parameter which determines how much padding is to be done
            while doing the FFT calculations.
    Units:
        all in cgs units.
    """

    try:
        file_in = sys.argv[1]
        file_out = sys.argv[2]
        distance = float(sys.argv[3])
        c0 = float(sys.argv[4])
        npad_fft = int(sys.argv[5])
    except:
        print "check the input."

    # opening the file
    hdulist = fits.open(file_in)
    data = hdulist[0].data.copy()

    try:
        dx = abs(float(hdulist[0].header['CDELT1'])
                 * 1.4959787e13 * 3600.0 * distance)
    except:
        print "using CD1_1, check later"
        dx = abs(float(hdulist[0].header['CD1_1'])
                 * 1.4959787e13 * 3600.0 * distance)
    dy = dx
    dv = abs(float(hdulist[0].header['CDELT3'])) * 1e2

    # convering the flux to column density
    k_fco = 1.0 / 1.0e5 * 5.0e20 * 3.33e-24

    # obtainting the density distribution
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

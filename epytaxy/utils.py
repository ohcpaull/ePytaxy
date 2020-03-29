import numpy as np
from scipy import signal
from scipy.interpolate import UnivariateSpline


def atomic_scattering_factor(QQ, a1, b1, a2, b2, a3, b3, a4, b4, c):
    return (
        a1 * np.exp(-b1 * QQ * np.pi ** 2 / 16)
        + a2 * np.exp(-b2 * QQ * np.pi ** 2 / 16)
        + a3 * np.exp(-b3 * QQ * np.pi ** 2 / 16)
        + a4 * np.exp(-b4 * QQ * np.pi ** 2 / 16)
        + c
    )


def structure_factor(fA, fB, fO, Q, c, z):
    # print(np.shape(z))
    f = (
        fA * np.exp(1j * Q * z[0] * c)
        + fB * np.exp(1j * Q * z[1] * c)
        + fO * np.exp(1j * Q * z[2] * c)
        + fO * np.exp(1j * Q * z[3] * c)
        + fO * np.exp(1j * Q * z[4] * c)
    )
    return f


def calc_FWHM(x, y, clr=False, ax=None):

    # create a spline of x and blue-np.max(blue)/2
    spline = UnivariateSpline(x, y - np.max(y) / 2, s=0)
    r1, r2 = spline.roots()  # find the roots
    if clr == True:
        ax.axvspan(r1, r2, facecolor="g", alpha=0.5)

    return r2 - r1


def convolute(ampl, FWHM, ttrange=(15, 115)):

    # Normalise amplitude values
    maxVal = np.max(np.abs(ampl))
    print(maxVal)
    normAmp = ampl / np.abs(np.max(ampl))

    # Convolute film G with gaussian
    filt = signal.gaussian(len(ampl), FWHM / (ttrange[1] - ttrange[0]) * len(ampl))
    ampc = signal.convolve(normAmp, filt, mode="same", method="auto")
    ampcn = ampc / np.max(np.abs(ampc))
    # print(np.max(np.abs(amp1cn)))

    # Restore to initial intensity
    ampc = ampcn * maxVal
    return ampc


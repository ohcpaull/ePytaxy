import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.io
from scipy.spatial.transform import Rotation as R
from scipy import signal
from tqdm import tqdm
import time

from .utils import atomic_scattering_factor, structure_factor, calc_FWHM, convolute
from .mat_data import element_factors, c_axis, AB_atoms


class Substrate(object):
    def __init__(self, sub_mat, orientation, ttrange=(15, 115)):
        self.ttheta = np.linspace(ttrange[0], ttrange[1], 20000)
        # if len(ttrange) == 2:
        #
        # else:
        #    self.ttheta = np.linspace(15, 115, 20000)

        self.Q = 4 * np.pi / 1.5406 * np.sin(np.radians(self.ttheta / 2))
        self.QQ = self.Q * self.Q

        self.atom_posns = []  # z-positions of atoms in a cubic perovskite structure
        self.atom_posns.append(0)  # A
        self.atom_posns.append(0.5)  # B
        self.atom_posns.append(0)  # O1
        self.atom_posns.append(0.5)  # O2
        self.atom_posns.append(0.5)  # O3

        self.sub = sub_mat
        # self.ori = orientation
        # self.prefix = 'gsubstrate'
        # substrates = scipy.io.loadmat('Substrates.mat')

        # if ( self.prefix + self.sub + self.ori ) in substrates:
        #    self.g = substrates[ self.prefix + self.sub + self.ori ]
        # else:
        #    print('Invalid substrate!')

        self.c = c_axis[self.sub]
        self.N001 = 2e4
        self.d111 = self.c / np.sqrt(3)
        self.N111 = self.N001 * np.sqrt(3)

        A = element_factors[AB_atoms[self.sub][0]]
        B = element_factors[AB_atoms[self.sub][1]]
        O = element_factors["O"]

        self.fA = atomic_scattering_factor(
            self.QQ, A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]
        )
        self.fB = atomic_scattering_factor(
            self.QQ, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8]
        )
        self.fO = atomic_scattering_factor(
            self.QQ, O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7], O[8]
        )

        SF_sub = structure_factor(
            self.fA, self.fB, self.fO, self.Q, c_axis[self.sub], self.atom_posns
        )
        ampl = 0
        mu = 10000
        tot_thickness = 80000

        for i in tqdm(range(0, 10000)):

            ampl = ampl + SF_sub * np.exp(1j * self.Q * i * c_axis[self.sub]) * np.exp(
                -(tot_thickness - (i * c_axis[self.sub])) / mu
            )

        self.g = ampl


class Film(object):
    def __init__(self, film_mat, orientation, N, ttrange=(15, 115)):

        self.ttheta = np.linspace(ttrange[0], ttrange[1], 20000)
        # if len(ttrange) != (15, 115):
        #    self.ttheta = np.linspace(ttrange[0], ttrange[1], 20000)
        # else:
        #    print(ttrange)
        #    self.ttheta = np.linspace(15, 115, 20000)

        self.Q = 4 * np.pi / 1.5406 * np.sin(np.radians(self.ttheta / 2))
        self.QQ = self.Q * self.Q
        self.film = film_mat
        self.ori = orientation
        self.cz = []
        self.N = N
        self.NN = np.linspace(0, self.N, self.N)
        self.A_pos3D = (0, 0, 0)
        self.B_pos3D = (0.5, 0.5, 0.5)
        self.O1_pos3D = (0.5, 0.5, 0)
        self.O2_pos3D = (0.5, 0, 0.5)
        self.O3_pos3D = (0, 0.5, 0.5)

        # volume = self.c**3

        self.atom_posns = []  # z-positions of atoms in a cubic perovskite structure
        self.atom_posns.append(0)  # A
        self.atom_posns.append(0.5)  # B
        self.atom_posns.append(0)  # O1
        self.atom_posns.append(0.5)  # O2
        self.atom_posns.append(0.5)  # O3

        if orientation == ("001"):
            self.d = c_axis[self.film]
        elif orientation == ("111"):
            self.d = c_axis[self.film] / (np.sqrt(3))
        elif self.ori == ("110"):
            self.d = c_axis[self.film] / (np.sqrt(2))

        self.volume = self.d ** 3
        print("d-spacing is " + str(self.d))
        print("unit cell volume is " + str(self.volume))

    def SF_mono_comp(self, c, x):

        if self.film == "(Pb_x,Sr_{1-x})TiO3":
            A1 = element_factors[AB_atoms[self.film][0]]
            A2 = element_factors[AB_atoms[self.film][1]]
            B = element_factors[AB_atoms[self.film][2]]
            O = element_factors["O"]

            self.fA = atomic_scattering_factor(
                self.QQ, A1[0], A1[1], A1[2], A1[3], A1[4], A1[5], A1[6], A1[7], A1[8]
            ) * x + atomic_scattering_factor(
                self.QQ, A2[0], A2[1], A2[2], A2[3], A2[4], A2[5], A2[6], A2[7], A2[8]
            ) * (
                1 - x
            )
            self.fB = atomic_scattering_factor(
                self.QQ, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8]
            )
            self.fO = atomic_scattering_factor(
                self.QQ, O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7], O[8]
            )

            f_ML = structure_factor(
                self.fA, self.fB, self.fO, self.Q, self.atom_posns, c
            )
            return f_ML
        elif self.film == "Pb(Zr_x,Ti_{1-x})O3":
            A = element_factors[AB_atoms[self.film][0]]
            B1 = element_factors[AB_atoms[self.film][1]]
            B2 = element_factors[AB_atoms[self.film][2]]
            O = element_factors["O"]

            self.fA = atomic_scattering_factor(
                self.QQ, A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]
            )
            self.fB = atomic_scattering_factor(
                self.QQ, B1[0], B1[1], B1[2], B1[3], B1[4], B1[5], B1[6], B1[7], B1[8]
            ) * x + atomic_scattering_factor(
                self.QQ, B2[0], B2[1], B2[2], B2[3], B2[4], B2[5], B2[6], B2[7], B2[8]
            ) * (
                1 - x
            )
            self.fO = atomic_scattering_factor(
                self.QQ, O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7], O[8]
            )

            f_ML = structure_factor(
                self.fA, self.fB, self.fO, self.Q, self.atom_posns, c
            )
            return f_ML
        else:
            print(
                "This function ONLY calculates the structure factor for a monolayer with a compositional spread. Use SF_monolayer"
            )
        return

    def SF_monolayer(self, c, FWHM=0):
        if len(AB_atoms[self.film]) == 2:
            A = element_factors[AB_atoms[self.film][0]]
            B = element_factors[AB_atoms[self.film][1]]
            O = element_factors["O"]

            self.fA = atomic_scattering_factor(
                self.QQ, A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]
            )
            self.fB = atomic_scattering_factor(
                self.QQ, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8]
            )
            self.fO = atomic_scattering_factor(
                self.QQ, O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7], O[8]
            )

            if FWHM == 0:
                f_ML = structure_factor(
                    self.fA, self.fB, self.fO, self.Q, c, self.atom_posns
                )
                return f_ML
            else:

                self.z_mosaicity = self.rotate_posns(
                    FWHM,
                    self.A_pos3D,
                    self.B_pos3D,
                    self.O1_pos3D,
                    self.O2_pos3D,
                    self.O3_pos3D,
                )
                # print(ML_mosaicity)
                # f_ML_dist = []
                # for i in range(0, len(self.z_mosaicity) ):
                #    f_ML_dist.append( structure_factor( self.fA, self.fB, self.fO, self.Q, c, self.z_mosaicity[i] ) )
                # f_ML = np.average( f_ML_dist )

                f_ML = structure_factor(
                    self.fA, self.fB, self.fO, self.Q, c, self.atom_posns
                )
                # f_ML _dist = [ structure_factor( self.fA, self.fB, self.fO, self.Q, val, c ) for val in ML_mosaicity ]
                # f_ML = np.average(f_ML_dist)
                # f_ML = f_ML_dist
                return f_ML
        else:
            print(
                "This function ONLY calculates the structure factor for a monolayer without a compositional spread. Use SF_mono_comp"
            )

    def determine_relaxation(
        self, substrate, relaxation="None", power=2, A=-0.07, B=200, D=4.15
    ):
        rel_strain = (self.d - self.volume) / self.volume
        delta = rel_strain * self.volume
        # print(delta)

        if relaxation == "exponential":
            # power = arg
            c = 0
            cz = []
            for uc in self.NN:
                c = c + self.d + delta * np.exp(-power * uc / self.N)
                cz.append(c)
            self.cz = cz
            return self.cz
        elif relaxation == "power":
            # power = arg
            c = 0
            cz = []
            for uc in self.NN:
                c = c + self.d + delta * (uc / (self.N)) ** power
                cz.append(c)
            self.cz = cz
            return self.cz
        elif relaxation == "XRDFit":

            self.cz = [A * np.exp(i / B) + D for i in self.NN]
            return self.cz
        elif relaxation == "None":
            cz = []
            for uc in self.NN:
                cz.append(self.d)
            self.cz = cz
            return self.cz
            # print('Invalid relaxation method')

    def scattering_amplitude(self, FWHM=0):

        mu = 2e4  # attenuation factor to correct for absorption
        uc = 0
        ampl = 0

        # NOTE: Calculate the penetration depth of xrays in the thin film + substrate
        #       and use this as the "total thickness" introduced by
        tot_thickness = sum(self.cz) + 10000
        for i in range(0, self.N):
            # uc = uc + self.cz[i]
            # for i in range(0, len(self.z_mosaicity) ):
            #    c_mosaicity = self.rotate_posns( )
            #    struct_fact_mosaic.append( structure_factor( self.fA, self.fB, self.fO, self.Q, cax[i], self.z_mosaicity[i] ) )

            # Calculate total thickness/current thickness of layer for each mosaicity value
            # calculate amplitude for each total thickness z_mosaic
            # average the complex amplitudes

            struct_fact = self.SF_monolayer(self.cz[i], FWHM)

            ampl = ampl + struct_fact * np.exp(
                1j * self.Q * sum(self.cz[0:i])
            ) * np.exp(-(tot_thickness - sum(self.cz[0:i])) / mu)

        # print(np.shape(filt))
        # conv_ampl = [ signal.convolve( amp, filt) for amp in ampl ]

        return ampl

    def rotate_posns(self, FWHM=0.5, *posn):

        """
        Create gaussian distribution of z positions rocking around equilibrium with standard deviation equal to the FWHM
        FWHM = 2 * sqrt( 2 * ln( 2 ) ) * sigma
        Gaussian = 1/(sigma * sqrt(2*pi)) * exp( - (( x - x0 )/( 2*sigma**2 )) )
        """

        # print(deg)
        # posns = [A_pos, B_pos, O1_pos, O2_pos, O3_pos]
        theta_dist = np.random.normal(loc=0, scale=FWHM, size=50)
        rotations = [R.from_rotvec([np.radians(angle), 0, 0]) for angle in theta_dist]
        rot_posns = [rotation.apply(posn) for rotation in rotations]

        return rot_posns

    # def conv_scattering_amplitude(self, FWHM):
    #        """
    #        This function calculates the scattering amplitude for a c-axis profile that has a given mosaicity for each unit cell.
    #
    #        Parameters
    #        ----------
    #        FWHM    :   array
    #            array of FWHMs for each unit cell in the thin film. FWHM[0] is the FWHM of the unit cell closest to the substrate.
    #        #
    #
    #        Returns
    #        -------
    #        ampl : array
    #            Scattering amplitude for thin film
    #        """
    #
    #        mu = 2e4  # attenuation factor to correct for absorption
    #        uc = 0
    #        monolayer_ampl = 0
    #        ampl = 0

    def test_scattering_amplitude(self, FWHM):
        """
        This test function calculates the scattering amplitude for an array of different c-axis values that are consistent with mosaicity values determined from the FWHM
        
        These amplitudes are averaged, then the intensity is calculated from the square modulus. There is no discernible result on the XRD pattern (i.e. no peak broadening or washing out of laue oscillations)
        
        """

        mu = 2e4  # attenuation factor to correct for absorption
        uc = 0
        monolayer_ampl = 0
        ampl = 0
        # print(FWHM)
        # NOTE: Calculate the penetration depth of xrays in the thin film + substrate
        #       and use this as the "total thickness" introduced by
        tot_thickness = sum(self.cz) + 10000
        for i in tqdm(range(0, self.N)):
            # uc = uc + self.cz[i]
            # for i in range(0, len(self.z_mosaicity) ):
            #    c_mosaicity = self.rotate_posns( )
            #    struct_fact_mosaic.append( structure_factor( self.fA, self.fB, self.fO, self.Q, cax[i], self.z_mosaicity[i] ) )

            # Calculate total thickness/current thickness of layer for each mosaicity value
            # calculate amplitude for each total thickness z_mosaic
            # average the complex amplitudes

            struct_fact = self.SF_monolayer(self.cz[i], FWHM)
            # print((struct_fact))
            # print(type(struct_fact))
            # print(type(self.cz))
            cz_3D = np.array([0, 0, self.cz[i]])
            # print(cz_3D[2])
            self.c_spread = self.rotate_posns(FWHM, cz_3D)
            c_spread = np.array(self.c_spread)
            # print(np.shape(c_spread))
            # print(c_spread[:][0][0][2])
            for j in range(0, len(c_spread[:, 0, 2])):
                if j == 0:
                    # print(c_spread[j,0,2])

                    monolayer_ampl = (
                        struct_fact
                        * np.exp(1j * self.Q * (c_spread[j, 0, 2]))
                        * np.exp(-(tot_thickness - (c_spread[j, 0, 2])) / mu)
                    )
                    # print('done')
                else:
                    monolayer_ampl = monolayer_ampl + struct_fact * np.exp(
                        1j * self.Q * (sum(self.cz[0 : j - 1]) + c_spread[j, 0, 2])
                    ) * np.exp(
                        -(tot_thickness - (sum(self.cz[0 : j - 1]) + c_spread[j, 0, 2]))
                        / mu
                    )
                    # print(monolayer_ampl)
            monolayer_ampl = monolayer_ampl / len(c_spread[:, 0, 2])

            ampl = ampl + monolayer_ampl
            # print(np.shape((ampl)))

        # print(np.shape(filt))
        # conv_ampl = [ signal.convolve( amp, filt) for amp in ampl ]

        return ampl


import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.io
from scipy.spatial.transform import Rotation as R
from scipy import signal
from tqdm import tqdm
import time
import mat_data

class Substrate(object): 
    def __init__(self, sub_mat, orientation, **ttrange):
        self.c_axis = {
            "DyScO3" : 3.9403,
            "GdScO3" : 3.9636,
            "KTaO3"  : 3.989,
            "LaAlO3" : 3.789,
            "LSAT"   : 3.868,
            "NdAlO3" : 3.74,
            "NdGaO3" : 3.864,
            "SrTiO3" : 3.905,
            "YAlO3"  : 3.71
        }
        
        self.AB_atoms = {
            "DyScO3" : ["Dy", "Sc"],
            "GdScO3" : ["Gd", "Sc"],
            "KTaO3"  : ["K", "Ta"],
            "LaAlO3" : ["La", "Al"],
            "LSAT"   : ["La", "Al", "Sr", "Ta"], 
            "NdAlO3" : ["Nd", "Al"],
            "NdGaO3" : ["Nd", "Gd"],
            "SrTiO3" : ["Sr", "Ti"],
            "YAlO3"  : ["Y", "Al"]          
        }
        
        
        if len(ttrange) == 2:
            self.ttheta = np.linspace(ttrange[0], ttrange[1], 20000)
        else:
            self.ttheta = np.linspace(15, 115, 20000)

        self.Q = 4 * np.pi / 1.5406 * np.sin( np.radians( self.ttheta/2 ) )
        self.QQ = self.Q * self.Q
        
        
        self.atom_posns = []     # z-positions of atoms in a cubic perovskite structure
        self.atom_posns.append(0)     #A
        self.atom_posns.append(0.5)   #B
        self.atom_posns.append(0)     #O1
        self.atom_posns.append(0.5)   #O2
        self.atom_posns.append(0.5)   #O3
        
        self.sub = sub_mat
        #self.ori = orientation
        #self.prefix = 'gsubstrate'
        #substrates = scipy.io.loadmat('Substrates.mat')
        
        #if ( self.prefix + self.sub + self.ori ) in substrates:
        #    self.g = substrates[ self.prefix + self.sub + self.ori ]
        #else:
        #    print('Invalid substrate!')
            
        self.c = self.c_axis[self.sub] 
        self.N001 = 2e+4
        self.d111 = self.c / np.sqrt(3)
        self.N111 = self.N001 * np.sqrt(3)
        
                
        A = element_factors[self.AB_atoms[self.sub][0]]
        B = element_factors[self.AB_atoms[self.sub][1]]
        O = element_factors["O"]
            
        self.fA = self.atomic_scattering_factor( self.QQ, A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8] )
        self.fB = self.atomic_scattering_factor( self.QQ, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8] )
        self.fO = self.atomic_scattering_factor( self.QQ, O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7], O[8] )
            
        SF_sub = self.structure_factor( self.fA, self.fB, self.fO, self.Q, self.c_axis[self.sub], self.atom_posns )
        ampl=0
        mu= 10000
        tot_thickness = 80000
        
        for i in tqdm( range(0, 10000) ):
                
            ampl = ampl + SF_sub * np.exp( 1j * self.Q * i * self.c_axis[self.sub] ) * np.exp( - (tot_thickness - ( i * self.c_axis[self.sub] ) ) / mu )
            
        self.g = ampl
        
    
    
        

    
class Film(object):
    def __init__(self, film_mat, orientation ):


        self.c_axes = {
            "BaTiO3" : 4.036,
            "BiFeO3" : 3.965,
            "BiFeO3-T" : 4.66,
            "LaAlO3" : 3.79,
            #"LaFeO3" : 4,
            #"LaMnO3" : 4,
            "LaNiO3" : ["La", "Ni"],
            "LSMO"   : ["La", "Sr", "Mn"],
            #"MnTiO3" : ["Mn", "Ti"],
            #"NdNiO3" : ["Nd", "Ni"],
            "PbTiO3" : 4.14,
            #"(Pb_x,Sr_{1-x})TiO3" : ["Pb", "Sr", "Ti"],
            #"Pb(Zr_x,Ti_{1-x})O3" : ["Pb", "Zr", "Ti"],
            #"SmNiO3" : ["Sm", "Ni"],
            #"SrRuO3" : ,
            "SrTiO3" : 3.905 
        }
        
        self.AB_atoms = {
            "BaTiO3" : ["Ba", "Ti"],
            "BiFeO3" : ["Bi", "Fe"],
            "LaAlO3" : ["La", "Al"],
            "LaFeO3" : ["La", "Fe"],
            "LaMnO3" : ["La", "Mn"],
            "LaNiO3" : ["La", "Ni"],
            "LSMO"   : ["La", "Sr", "Mn"],
            "MnTiO3" : ["Mn", "Ti"],
            "NdNiO3" : ["Nd", "Ni"],
            "PbTiO3" : ["Pb", "Ti"],
            "(Pb_x,Sr_{1-x})TiO3" : ["Pb", "Sr", "Ti"],
            "Pb(Zr_x,Ti_{1-x})O3" : ["Pb", "Zr", "Ti"],
            "SmNiO3" : ["Sm", "Ni"],
            "SrRuO3" : ["Sr", "Ru"],
            "SrTiO3" : ["Sr", "Ti"] 
        }       
        
        self.ttheta = np.linspace(0, 180.01, 180.01/0.01)
        self.Q = 4 * np.pi / 1.5406 * np.sin( np.radians( self.ttheta/2 ) )
        self.QQ = self.Q * self.Q
        self.film = film_mat
        self.ori = orientation
        self.cz = []
        self.N = 125
        self.NN = np.linspace( 0, self.N, self.N )
        self.A_pos3D = (0,0,0)
        self.B_pos3D = (0.5, 0.5, 0.5)
        self.O1_pos3D = (0.5, 0.5, 0)
        self.O2_pos3D = (0.5, 0, 0.5)
        self.O3_pos3D = (0, 0.5, 0.5)
        
        #volume = self.c**3       
        
        self.atom_posns = []     # z-positions of atoms in a cubic perovskite structure
        self.atom_posns.append(0)     #A
        self.atom_posns.append(0.5)   #B
        self.atom_posns.append(0)     #O1
        self.atom_posns.append(0.5)   #O2
        self.atom_posns.append(0.5)   #O3
        

        
        if orientation == ('001'):
            self.d = self.c_axes[self.film]
        elif orientation == ('111'):
            self.d = self.c_axes[self.film]/(np.sqrt(3))
        elif self.ori == ('110'):
            self.d = self.c_axes[self.film]/(np.sqrt(2))
        
        self.volume = self.d**3
        print('d-spacing is ' + str(self.d) )
        print('unit cell volume is ' + str( self.volume))
        
    def atomic_scattering_factor( self, QQ, a1, b1, a2, b2, a3, b3, a4, b4, c ):
        return ( a1 * np.exp( -b1 * QQ * np.pi**2/16 ) + a2 * np.exp( -b2 * QQ * np.pi**2/16 ) + \
                a3 * np.exp( -b3 * QQ * np.pi**2/16 )+ a4 * np.exp( -b4 * QQ * np.pi**2/16) + c )
    
    def structure_factor( self, fA, fB, fO, Q, c, z):
        #print(np.shape(z))
        f = fA * np.exp( 1j*Q*z[0]*c ) + fB * np.exp( 1j*Q*z[1]*c ) + fO * np.exp( 1j * Q * z[2] * c ) + \
                                fO * np.exp( 1j * Q * z[3] * c ) + fO * np.exp( 1j * Q * z[4] * c )
        return f   
    
    def SF_mono_comp( self, c, x ):

        if self.film == "(Pb_x,Sr_{1-x})TiO3":
            A1 = self.element_factors[self.AB_atoms[self.film][0]] 
            A2 = self.element_factors[self.AB_atoms[self.film][1]]
            B = self.element_factors[self.AB_atoms[self.film][2]]
            O = self.element_factors["O"]
            
            self.fA = self.atomic_scattering_factor( self.QQ, A1[0], A1[1], A1[2], A1[3], \
                                               A1[4], A1[5], A1[6], A1[7], A1[8] ) * x + \
                      self.atomic_scattering_factor( self.QQ, A2[0], A2[1], A2[2], A2[3], \
                                               A2[4], A2[5], A2[6], A2[7], A2[8] ) * (1-x)
            self.fB = self.atomic_scattering_factor( self.QQ, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8] )
            self.fO = self.atomic_scattering_factor( self.QQ, O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7], O[8] )
                
            f_ML = self.structure_factor( self.fA, self.fB, self.fO, self.Q, self.atom_posns, c )
            return f_ML
        elif self.film == "Pb(Zr_x,Ti_{1-x})O3":
            A = self.element_factors[self.AB_atoms[self.film][0]] 
            B1 = self.element_factors[self.AB_atoms[self.film][1]] 
            B2 = self.element_factors[self.AB_atoms[self.film][2]]            
            O = self.element_factors["O"]                
        
            self.fA = self.atomic_scattering_factor( self.QQ, A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8] )
            self.fB = self.atomic_scattering_factor( self.QQ, B1[0], B1[1], B1[2], B1[3], \
                                               B1[4], B1[5], B1[6], B1[7], B1[8] ) * x + \
                      self.atomic_scattering_factor( self.QQ, B2[0], B2[1], B2[2], B2[3], \
                                               B2[4], B2[5], B2[6], B2[7], B2[8] ) * (1-x)
            self.fO = self.atomic_scattering_factor( self.QQ, O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7], O[8] )
                
            f_ML = self.structure_factor( self.fA, self.fB, self.fO, self.Q, self.atom_posns, c )
            return f_ML
        else:
            print('This function ONLY calculates the structure factor for a monolayer with a compositional spread. Use SF_monolayer')
        return
        
    def SF_monolayer( self, c, FWHM=0 ):
        if (len(self.AB_atoms[self.film]) == 2):
            A = self.element_factors[self.AB_atoms[self.film][0]]
            B = self.element_factors[self.AB_atoms[self.film][1]]
            O = self.element_factors["O"]
            
            self.fA = self.atomic_scattering_factor( self.QQ, A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8] )
            self.fB = self.atomic_scattering_factor( self.QQ, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8] )
            self.fO = self.atomic_scattering_factor( self.QQ, O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7], O[8] )
            
            
            if FWHM == 0:
                f_ML = self.structure_factor( self.fA, self.fB, self.fO, self.Q, c, self.atom_posns )
                return f_ML
            else:
                
                self.z_mosaicity = self.rotate_posns( FWHM, self.A_pos3D, self.B_pos3D, self.O1_pos3D, self.O2_pos3D, self.O3_pos3D ) 
                #print(ML_mosaicity)
                #f_ML_dist = []
                #for i in range(0, len(self.z_mosaicity) ):
                #    f_ML_dist.append( self.structure_factor( self.fA, self.fB, self.fO, self.Q, c, self.z_mosaicity[i] ) )
                #f_ML = np.average( f_ML_dist )
                    
                f_ML = self.structure_factor( self.fA, self.fB, self.fO, self.Q, c, self.atom_posns )
                #f_ML _dist = [ self.structure_factor( self.fA, self.fB, self.fO, self.Q, val, c ) for val in ML_mosaicity ]
                #f_ML = np.average(f_ML_dist)
                #f_ML = f_ML_dist
                return f_ML
        else:
            print('This function ONLY calculates the structure factor for a monolayer without a compositional spread. Use SF_mono_comp')
    
    def determine_relaxation( self, substrate, power, relaxation='exponential' ):
        rel_strain = ( self.d - self.volume ) / self.volume
        delta = rel_strain * self.volume
        print(delta)
        
        if relaxation == 'exponential':
            c = 0
            cz = []
            for uc in self.NN:
                c = c + self.d + delta * np.exp( - power * uc/self.N)
                cz.append( c )
            self.cz = cz
        elif relaxation == 'power':
            c = 0
            cz = []
            for uc in self.NN:
                c = c + self.d + delta*( uc / ( self.N ) )**power 
                cz.append( c )
            self.cz = cz
        elif relaxation == 'XRDFit':
            A = -0.07
            B = 200
            D = 4.15
            self.cz = [ A*np.exp(i/B) + D for i in self.NN ]
            #print('Invalid relaxation method')
       
    def scattering_amplitude( self, power, FWHM=0 ):
            
        mu = 2e4 #attenuation factor to correct for absorption
        uc = 0
        ampl = 0
                
        # NOTE: Calculate the penetration depth of xrays in the thin film + substrate
        #       and use this as the "total thickness" introduced by 
        tot_thickness = sum(self.cz) + 10000
        for i in range(0,N):
            #uc = uc + self.cz[i]
            #for i in range(0, len(self.z_mosaicity) ):
            #    c_mosaicity = self.rotate_posns( )
            #    struct_fact_mosaic.append( self.structure_factor( self.fA, self.fB, self.fO, self.Q, cax[i], self.z_mosaicity[i] ) )
            
            # Calculate total thickness/current thickness of layer for each mosaicity value
            # calculate amplitude for each total thickness z_mosaic
            # average the complex amplitudes
            
            struct_fact = self.SF_monolayer( self.cz[i], FWHM )
            #print((struct_fact))
            #print(type(struct_fact))
            #print(type(self.cz))
            
            
            ampl = ampl + struct_fact * np.exp( 1j * self.Q * sum(self.cz[0:i]) ) * np.exp( - (tot_thickness - sum(self.cz[0:i])) / mu )
            #print(np.shape((ampl)))
        
        #print(np.shape(filt))
        #conv_ampl = [ signal.convolve( amp, filt) for amp in ampl ]
        
            
        return ampl/(self.N)
        
    # Create gaussian distribution of z positions rocking around equilibrium with standard deviation equal to the FWHM
    # FWHM = 2 * sqrt( 2 * ln( 2 ) ) * sigma
    # Gaussian = 1/(sigma * sqrt(2*pi)) * exp( - (( x - x0 )/( 2*sigma**2 )) )
    
    def rotate_posns( self, FWHM=0.5, *posn ):
        #print(deg)
        #posns = [A_pos, B_pos, O1_pos, O2_pos, O3_pos]
        theta_dist = np.random.normal( loc=0, scale=FWHM, size=50 )
        rotations = [ R.from_rotvec([np.radians(angle),0,0]) for angle in theta_dist ]
        rot_posns = [ rotation.apply(posn) for rotation in rotations ] 
        
        return rot_posns
    
    def test_scattering_amplitude( self, FWHM ):
        '''
        This test function calculates the scattering amplitude for an array of different c-axis values that are consistent with mosaicity values determined from the FWHM
        
        These amplitudes are averaged, then the intensity is calculated from the square modulus. There is no discernible result on the XRD pattern (i.e. no peak broadening or washing out of laue oscillations)
        
        '''
            
        mu = 2e4 #attenuation factor to correct for absorption
        uc = 0
        monolayer_ampl = 0
        ampl = 0
        #print(FWHM)
    # NOTE: Calculate the penetration depth of xrays in the thin film + substrate
    #       and use this as the "total thickness" introduced by 
        tot_thickness = sum(self.cz) + 10000
        for i in tqdm(range(0,N)):
        #uc = uc + self.cz[i]
        #for i in range(0, len(self.z_mosaicity) ):
            #    c_mosaicity = self.rotate_posns( )
            #    struct_fact_mosaic.append( self.structure_factor( self.fA, self.fB, self.fO, self.Q, cax[i], self.z_mosaicity[i] ) )
            
            # Calculate total thickness/current thickness of layer for each mosaicity value
            # calculate amplitude for each total thickness z_mosaic
            # average the complex amplitudes
            
            struct_fact = self.SF_monolayer( self.cz[i], FWHM )
            #print((struct_fact))
            #print(type(struct_fact))
            #print(type(self.cz))
            cz_3D = np.array([0,0,self.cz[i]])
            #print(cz_3D[2])
            self.c_spread = self.rotate_posns( FWHM, cz_3D )
            c_spread = np.array( self.c_spread )
            #print(np.shape(c_spread))
            #print(c_spread[:][0][0][2])
            for j in range(0, len(c_spread[:,0,2])):
                if j == 0:
                    #print(c_spread[j,0,2])
                    
                    monolayer_ampl = np.exp( 1j * self.Q * ( c_spread[j,0,2] ) ) * np.exp( - ( tot_thickness - ( c_spread[j,0,2] ) )/mu )  
                    #print('done')
                else:
                    monolayer_ampl =  monolayer_ampl + np.exp( 1j * self.Q * ( sum(self.cz[0:j-1]) + c_spread[j,0,2] ) ) * np.exp( - ( tot_thickness - ( sum( self.cz[0:j-1] ) + c_spread[j,0,2] ) )/ mu )  
                    #print(monolayer_ampl)
            monolayer_ampl = monolayer_ampl / len(c_spread[:,0,2])
        
            ampl = ampl + monolayer_ampl
            #print(np.shape((ampl)))
        
        #print(np.shape(filt))
        #conv_ampl = [ signal.convolve( amp, filt) for amp in ampl ]
        
            
        return ampl    

     
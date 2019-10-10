import numpy as np
import scipy.interpolate as spint
import astropy.units as u
import astropy.constants as const
import scipy.ndimage.filters as spfilt
import sncosmo


def get_color_band(band_name):
    color_band = {'A':'grey',
            'U': 'blue',
            'B': 'royalblue',
            'V':  'limegreen',
            'R':  'red',
            'I':  'mediumvioletred',
            'g':    'darkgreen',
            'i':    'purple',
            'r':    'darkred',
            'z':    'sienna',
            'u': 'darkblue',
            'UVW1':'purple',
            'UVW2':'pink',
            'UVM2':'m',
            'H':'brown',
            'J':'salmon',
            'K':'sienna'}
    if band_name in color_band.keys():
        return color_band[band_name]
    else:
        return 'grey'

#load vega spectrum
from astropy.io import fits
vega = fits.open('/Users/mariavincenzi/PhD/Photometry_Utils/alpha_lyr_stis_006.fits')
w_Vega_full, f_Vega_full = vega[1].data['WAVELENGTH'], vega[1].data['FLUX']
mask_Vega = (w_Vega_full>500.)&(w_Vega_full<25000)
w_Vega = w_Vega_full[mask_Vega]
f_Vega = f_Vega_full[mask_Vega]


def loadFilter(dir, natural=True):
    '''In this function, the files cotaining the filter
    transmission functions are loaded and returns the wavelengths and
    transmission functions

    NATURAL = TRUE means 'This filter has NOT been multiplied for lambda. So what im reading from the file is just the transmission of the filter'
    NATURAL = FALSE means 'This filter has already been multiplied for lambda. So what im reading from the file is not just the transmission of the filter itself but transmission*lambda'
    '''
    w, t = np.loadtxt(dir,usecols=(0,1), dtype=np.float64, unpack=True)

    if max(t)>1.:
        t = t/100.

    if natural: return w, (t)/max(t)
    else: return w, (t/w)/max(t/w)

def fluxLamFilter(wave, filt_wave, fil_trans, F_lam):
    filt_int = np.interp(wave, filt_wave, fil_trans, left=0, right=0)
    filtspec = filt_int*F_lam
    fluxFilt = np.trapz(filtspec,wave)
    #if type(fluxFilt)==u.quantity.Quantity:
    #    return fluxFilt.value
    #else: return float(fluxFilt)
    return fluxFilt

def countFilter(wave, filt_wave, fil_trans, F_cnt):
    filt_int = np.interp(wave, filt_wave, fil_trans)
    filtspec = filt_int*F_cnt
    countFilt = np.trapz(filtspec,wave)
    if type(countFilt)==u.quantity.Quantity:
        return countFilt.value
    else: return float(countFilt)

def mag2flux(band,magnitude, perArea=True):
    return band.zpFlux(perArea=perArea)*10.**(magnitude/-2.5)

def ERRmag2ERRflux(band, magnitude, err_magnitude, perArea=True):
    #  fZP * 10**(-0.4x)* ln(10) * (-0.4) *err_mag
    return band.zpFlux(perArea=perArea)* np.abs(10**(-0.4*magnitude) * np.log(10.)*(-0.4)*err_magnitude)

def flux2mag(band,flux, perArea=True):
    return -2.5*np.log10(flux/ band.zpFlux(perArea=perArea))

def ERRflux2ERRmag(band,flux, err_flux, perArea=True):
    return 2.5/np.log(10.) * err_flux/flux

def mag2Counts(band, magnitude,totalThrough=True):
    if totalThrough==True:
        return band.zpCnt(perArea=True)*10.**(magnitude/-2.5) * band.areaA()
    else:
        return band.zpCnt(perArea=False)*10.**(magnitude/-2.5)


class Band_AB(object):
    """The transmission function of a filter.
    The transmission is a fraction of 1, unless fraction = False
    The wavelengths are in Angstrom

    Parameters
    -----------
    wave: array

    transmission: array

    Examples
    -----------

    import pyfilter as wtf
    w, t = wtf.loadFilter('/path/to/Bessell_B.dat')
    B=wtf.Band(w,t)

    r.meanWave()
    4413.083819849874

    r.effWave()
    4357.412565527486

    r.zpFlux()
    5.69885539058255e-09

    """


    def __init__(self, wave, transmission):
        super(Band_AB, self).__init__()
        if wave.shape != transmission.shape:
            raise ValueError('The wave array and transmission array\
             are different sizes!')

        self.wave = wave*u.AA
        self.transmission = transmission

    def _Fnu(self):
        return 3631e-23*(u.erg/u.s/u.cm**2/u.Hz)

    def filterFLambdaBins(self):
        return np.arange(min(self.wave.value),max(self.wave.value)+1.,1.)*u.AA

    def _zpFluxesFlam(self):
        wavelength_array = self.filterFLambdaBins()
        F_nu = self._Fnu()
        F_lam = (F_nu*const.c.cgs/((wavelength_array)**2.)).to(u.erg/u.s/u.cm**2/u.AA)
        return wavelength_array, F_lam

    def _zpFluxesFcnt(self):
        wavelength_array = self.filterFLambdaBins()
        F_nu = self._Fnu()

        F_lam = (F_nu*const.c.cgs/((wavelength_array)**2.)).to(u.erg/u.s/u.cm**2/u.AA)

        F_cnt = ((F_lam.value*wavelength_array.value)/(const.c.cgs.value*const.h.cgs.value))*1e-8
        return wavelength_array, F_cnt

    def areaA(self):
        return ((np.trapz(self.wave,self.transmission))**2)**0.5

    def zpAngstrom(self,perArea=True):
        zp_dum = self._zpFluxesFlam()
        if perArea==True:
            return -2.5*np.log10(fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])/self.areaA())
        else:
            return -2.5*np.log10(fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1]))

    def zpFlux(self,perArea=True):
        zp_dum = self._zpFluxesFlam()
        if perArea==True:
            return fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])/self.areaA()
        else:
            return fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])
    
    def zpFluxFnu(self,perArea=True):
        F0lam = self.zpFlux(perArea=perArea)
        return (F0lam / const.c.cgs *((self.effWave())**2.)).to(u.erg/u.s/u.cm**2/u.Hz)

    def zpCnt(self,perArea=True):
        zp_dum = self._zpFluxesFcnt()
        if perArea==True:
            return countFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])/self.areaA()
        else:
            return countFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])

    def meanWave(self):
        return np.average(self.wave,weights=self.transmission)

    def effWave(self):
        fw = np.interp(self.wave, self.filterFLambdaBins().value, self._zpFluxesFlam()[1])
        n1 = self.wave*self.transmission*fw
        d1 = self.transmission*fw
        return np.trapz(n1)/np.trapz(d1)

    def pivot(self):
        lt = np.trapz(self.wave*self.transmission, self.wave)
        return np.sqrt(lt/np.trapz(self.transmission/self.wave,self.wave))

    def peakWave(self):
        return self.wave[self.transmission==max(self.transmission)][0]

    def minWave(self):
        return min(self.wave)

    def maxWave(self):
        return max(self.wave)

    def wiredfunc(self):
        int1 = np.trapz(self.transmission/self.wave, self.wave)
        int2 = np.trapz(self.transmission/(self.wave)**2, self.wave)
        int3 = np.trapz(self.transmission, self.wave)
        return int1**2 / (int2*int3)

    def extinction(self, eb_v, dust_law, r_v=3.1):
        """ Given eb-v and r-v return the dust extinction in a particular BAND 
         It can use:
         Cardelli dust law (CCM),
         ODonneal (OD94)
         Fitzpatrick (F99)

        KEEP IN MIND:
        _minwave = 909.09
        _maxwave = 33333.33
         """
        if dust_law=='OD94':
            dust = sncosmo.OD94Dust()
        elif dust_law=='CCM':
            dust = sncosmo.CCM89Dust()
        elif dust_law=='F99':
            dust = sncosmo.F99Dust()
        else: print ('Add this dust law! I dont know it')

        dust.parameters = [eb_v, r_v]
        w = self.wave
        t = self.transmission
        ext = dust.propagate(w,t)
        correction_extinction = np.trapz((ext)[1:-1], w[1:-1])/np.trapz((t)[1:-1], w[1:-1])
        return correction_extinction


class Band_Vega(object):
    """The transmission function of a filter.
    The transmission is a fraction of 1, unless fraction = False
    The wavelengths are in Angstrom

    Parameters
    -----------
    wave: array

    transmission: array

    Examples
    -----------

    import pyfilter as wtf
    w, t = wtf.loadFilter('/path/to/Bessell_B.dat')
    B=wtf.Band(w,t)

    r.meanWave()
    4413.083819849874

    r.effWave()
    4357.412565527486

    r.zpFlux()
    5.69885539058255e-09

    """


    def __init__(self, wave, transmission):
        super(Band_Vega, self).__init__()
        if wave.shape != transmission.shape:
            raise ValueError('The wave array and transmission array\
             are different sizes!')

        self.wave = wave*u.AA
        self.transmission = transmission


    def filterFLambdaBins(self):
        return np.arange(min(self.wave.value),max(self.wave.value)+1.,1.)*u.AA

    def _zpFluxesFlam(self):
        wavelength_array = self.filterFLambdaBins()

        #intepolate Vega to the same wavelength_array
        F_Vega_interp = np.interp(wavelength_array, w_Vega, f_Vega)*u.erg/u.s/u.cm**2/u.AA

        return wavelength_array, F_Vega_interp

    def _zpFluxesFcnt(self):
        wavelength_array = self.filterFLambdaBins()

        F_lam = self._zpFluxesFlam()

        F_cnt = ((F_lam.value*wavelength_array.value)/(const.c.cgs *const.h.cgs ))*1e-8
        return wavelength_array, F_cnt

    def _zpFluxesFnu(self):
        wavelength_array = self.filterFLambdaBins()
        
        F_lam = self._zpFluxesFlam()[1]

        F_nu = (F_lam / const.c.cgs *((wavelength_array)**2.)).to(u.erg/u.s/u.cm**2/u.Hz)
        return wavelength_array, F_nu

    def areaA(self):
        return ((np.trapz(self.wave,self.transmission))**2)**0.5

    def zpAngstrom(self,perArea=True):
        zp_dum = self._zpFluxesFlam()
        if perArea==True:
            return -2.5*np.log10(fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])/self.areaA())
        else:
            return -2.5*np.log10(fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1]))

    def zpFlux(self,perArea=True):
        zp_dum = self._zpFluxesFlam()
        if perArea==True:
            return fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])/self.areaA()
        else:
            return fluxLamFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])

    def zpFluxFnu(self,perArea=True):
        F0lam = self.zpFlux(perArea=perArea)
        return (F0lam / const.c.cgs *((self.effWave())**2.)).to(u.erg/u.s/u.cm**2/u.Hz)

    def zpCnt(self,perArea=True):
        zp_dum = self._zpFluxesFcnt()
        if perArea==True:
            return countFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])/self.areaA()
        else:
            return countFilter(zp_dum[0], self.wave, self.transmission, zp_dum[1])

    def meanWave(self):
        return np.average(self.wave,weights=self.transmission)

    def effWave(self):
        fw = np.interp(self.wave, self.filterFLambdaBins().value, self._zpFluxesFlam()[1])
        n1 = self.wave*self.transmission*fw
        d1 = self.transmission*fw
        return np.trapz(n1)/np.trapz(d1)

    def pivot(self):
        lt = np.trapz(self.wave*self.transmission, self.wave)
        return np.sqrt(lt/np.trapz(self.transmission/self.wave,self.wave))

    def peakWave(self):
        return self.wave[self.transmission==max(self.transmission)][0]

    def minWave(self):
        return min(self.wave)

    def maxWave(self):
        return max(self.wave)

    def extinction(self, eb_v, dust_law, r_v=3.1):
        """ Given eb-v and r-v return the dust extinction in a particular BAND 
         It can use:
         Cardelli dust law (CCM),
         ODonneal (OD94)
         Fitzpatrick (F99)
        
        KEEP IN MIND:
        _minwave = 909.09
        _maxwave = 33333.33
         """
        if dust_law=='OD94':
            dust = sncosmo.OD94Dust()
        elif dust_law=='CCM':
            dust = sncosmo.CCM89Dust()
        elif dust_law=='F99':
            dust = sncosmo.F99Dust()
        else: print ('Add this dust law! I dont know it')

        dust.parameters = [eb_v, r_v]
        w = self.wave
        t = self.transmission
        ext = dust.propagate(w,t)
        correction_extinction = np.trapz((ext)[1:-1], w[1:-1])/np.trapz((t)[1:-1], w[1:-1])
        return correction_extinction





class Spectrum(object):
    """docstring for Spectrum."""
    def __init__(self,wave,flux):
        super(Spectrum, self).__init__()
        self.wave = wave
        self.flux = flux

    def bandflux(self, band):
        """Do some integration through a filter in this function
        """
        filt_int = np.interp(self.wave, band.wave,band.transmission)
        filtspec = filt_int*self.flux
        flux1 = np.trapz(filtspec,self.wave)/band.areaA()
        return flux1

    def runningSmooth(self, AAbin=None, NumBin=None):
        if AAbin==NumBin==None:
            raise ValueError('You must provide a binning.\
             Either AAbin to provide the angstrom bins (rounded to\
              nearest integer) or NumBin for number of bins')
        if NumBin is not None and NumBin.is_integer()==False:
            raise ValueError('NumBin must be an integer')

        if AAbin is not None:
            medDiff = np.median(np.diff(self.wave))
            nbins = int(AAbin/medDiff)
            self.wave = np.convolve(self.wave, np.ones((nbins,))/nbins, mode='valid')
            self.flux = np.convolve(self.flux, np.ones((nbins,))/nbins, mode='valid')
        elif AAbin is None:
            self.wave = np.convolve(self.wave, np.ones((NumBin,))/NumBin, mode='valid')
            self.flux = np.convolve(self.flux, np.ones((NumBin,))/NumBin, mode='valid')

    def binSpec(self, nbins, median=True):
        """Write this function at some point
        median or mean bin the spectrum
        """
        def medianFootprint(values):
            return np.median(values)
        def meanFootprint(values):
            return np.mean(values)
        footprint = np.ones((nbins,))
        if median==True:
            self.wave = spfilt.generic_filter(self.wave,medianFootprint,footprint=footprint,mode='reflect',cval=0.)
            self.flux = spfilt.generic_filter(self.flux,medianFootprint,footprint=footprint,mode='reflect',cval=0.)
        else:
            self.wave = spfilt.generic_filter(self.wave,meanFootprint,footprint=footprint,mode='reflect',cval=0.)
            self.flux = spfilt.generic_filter(self.flux,meanFootprint,footprint=footprint,mode='reflect',cval=0.)

    def filterOverlap(self, band):
        """This function checks to see if there is complete, partial, or no overlap of the spectrum
        with the desired band"""
        low1 = min(band.wave[band.transmission>1e-3])
        hi1 = max(band.wave[band.transmission>1e-3])

        low2 = min(self.wave)
        hi2 = max(self.wave)
        inarr = low1>low2 and hi1<hi2
        return inarr

#def polyFitPhot()
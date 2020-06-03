import numpy as np
import sncosmo
import os
import matplotlib.pyplot as plt

name2type = {'ASASSN14jb':'II',
   'ASASSN15oz':'II', 'SN1999em':'II', 'SN2004et':'II', 'SN2005cs':'II', 
   'SN2007od':'II', 'SN2008bj':'II', 'SN2008in':'II', 'SN2009N':'II', 
   'SN2009bw':'II', 'SN2009dd':'II', 'SN2009ib':'II', 'SN2009kr':'II', 
    'SN2012A':'II', 'SN2012aw':'II', 'SN2013ab':'II', 'SN2013am':'II', 
   'SN2013by':'II', 'SN2013ej':'II', 'SN2013fs':'II', 'SN2014G':'II', 
    'SN2016X':'II', 'SN2016bkv':'II', 'SN1987A':'II',  'SN1993J':'IIb',
    'SN2006T':'IIb', 'SN2008aq':'IIb', 'SN2008ax':'IIb', 'SN2008bo':'IIb',
   'SN2011dh':'IIb', 'SN2011ei':'IIb', 'SN2011fu':'IIb', 'SN2011hs':'IIb',
   'SN2013df':'IIb', 'SN2016gkg':'IIb', 'SN1999dn':'IIb', 'SN2006aa':'IIn', 'SN2007pk':'IIn',
   'SN2008fq':'IIn', 'SN2009ip':'IIn', 'SN2010al':'IIn', 'SN2011ht':'IIn',
   'SN2004gq':'Ib', 'SN2004gv':'Ib', 'SN2005bf':'Ib', 'SN2005hg':'Ib',
   'SN2006ep':'Ib', 'SN2007Y':'Ib', 'SN2007uy':'Ib', 'SN2008D':'Ib',
   'SN2009iz':'Ib', 'SN2009jf':'Ib', 'SN2012au':'Ib', 'iPTF13bvn':'Ib',
   'SN1994I': 'Ic', 'SN2004aw': 'Ic', 'SN2004fe': 'Ic', 'SN2004gt': 'Ic',
   'SN2007gr': 'Ic', 'SN2011bm': 'Ic', 'SN2013ge': 'Ic',
   'SN1998bw': 'Ic-BL', 'SN2002ap': 'Ic-BL', 'SN2006aj': 'Ic-BL', 'SN2007ru': 'Ic-BL',
   'SN2009bb': 'Ic-BL', 'SN2012ap': 'Ic-BL'}

pycoco_dir = '/home/mvincenzi/PyCOCO_templates/'
tmpl_host_extinction_corrected = [f for f in os.listdir(pycoco_dir+'Templates_HostCorrected') if 'SN2005cs' not in f]
tmpl_host_extinction_NOTcorrected = [f for f in os.listdir(pycoco_dir+'Templates_noHostCorr') if 'SN2005cs' not in f]

## the structure of the PYCOCO files is:
#   phase0 wls0 flux
#   phase0 wls1 flux
#   phase0 wls2 flux
# ....
#   phaseN wlsN-1 flux
#   phaseN wlsN flux
## We need to adapt the format to build the sncosmo class TimeSeriesSource

for tmpl in tmpl_host_extinction_corrected[:]:
    tmpl_name = tmpl.replace('.SED', '')
    phase_raw, wls_raw, flux_raw = np.genfromtxt(pycoco_dir+'Templates_HostCorrected/'+tmpl, unpack=True)
    wls = np.unique(wls_raw)
    phase = phase_raw[::len(wls)]
    flux_reshaped = flux_raw.reshape(len(phase),len(wls))
    source = sncosmo.TimeSeriesSource(phase, wls, flux_reshaped, zero_before=True, name=tmpl_name)
    sncosmo.register(source, name=tmpl_name)
    print (tmpl_name, '(type %s) registered!'%name2type[tmpl_name.replace('pycoco_','')])

for tmpl in tmpl_host_extinction_NOTcorrected[:]:
    tmpl_name = tmpl.replace('.SED', '')
    phase_raw, wls_raw, flux_raw = np.genfromtxt(pycoco_dir+'Templates_noHostCorr/'+tmpl, unpack=True)
    wls = np.unique(wls_raw)
    phase = phase_raw[::len(wls)]
    flux_reshaped = flux_raw.reshape(len(phase),len(wls))
    source = sncosmo.TimeSeriesSource(phase, wls, flux_reshaped, zero_before=True, name=tmpl_name)
    print (tmpl_name, '(type %s) registered!'%name2type[tmpl_name.replace('pycoco_','').replace('_noHostCorr','')])

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia
import numpy as np
from functools import reduce
from scipy import stats
from astropy.modeling import models, fitting
from astropy.table import Table, join, MaskedColumn, vstack

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm


class GaiaClusterMembers(object):
    '''
    This Class will grab data from the Gaia archive, and attempt to determine members using the 
    proper motions, radial velocities and parallaxes.  
    
    The user must provide the RA and Dec values, and the Class will return the full catalog and 
    the indices of the members.
    
    Note: in this code, each membership check builds on the previous one, using only the stars that 
    have passed the previous membership selection.  This may not be ideal, but appears to work better 
    than if I simply take all stars in each of the membership calculations.
    '''
    
    def __init__(self, *args,**kwargs):
    
        #required inputs
        self.RA = None
        self.Dec = None

        #parameters that the user could change
        self.radius = 1 #in degrees
        self.minPMerror = 5
        self.minRVmembership = 0.5
        self.minPMmembership = 0.5
        self.minPamembership = 0.3
        self.PaPolyD = 6
        self.verbosity = 1
        self.showPlots = False
        self.RVmin = -100. #km/s
        self.RVmax = 100. #km/s
        self.RVbins = 100
        self.dmin = 0. #parsecs
        self.dmax = 3000. #parsecs
        self.dbins = 200
        self.PMxmin = -200 #mas/yr
        self.PMxmax = 200 #mas/yr
        self.PMxbins = 400
        self.PMymin = -200 #mas/yr
        self.PMymax = 200 #mas/yr
        self.PMybins = 400  
        self.CMDxmin = 0.5
        self.CMDxmax = 2.5
        self.CMDymin = 18
        self.CMDymax = 8
        self.RVmean = None
        self.distance = None
        self.PMmean = [None, None]
        
        #outputs
        self.catalog = None
        self.members = None
        self.RVmembers = None
        self.PMmembers = None
        self.PAmembers = None
        self.dist = None
        
    def getGaiaData(self):
        if (self.verbosity > 0):
            print("Retrieving catalog ... ")
        cmd = f"SELECT * FROM gaiadr2.gaia_source \
        WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra, gaiadr2.gaia_source.dec),\
        CIRCLE('ICRS', {self.RA}, {self.Dec}, {self.radius}))=1\
        AND abs(pmra_error)<{self.minPMerror} \
        AND abs(pmdec_error)<{self.minPMerror} \
        AND pmra IS NOT NULL AND abs(pmra)>0 \
        AND pmdec IS NOT NULL AND abs(pmdec)>0;"
        if (self.verbosity > 1):
            print(cmd)
        job = Gaia.launch_job_async(cmd, dump_to_file=False) #could save this to a file
        self.catalog = job.get_results()
        self.members = range(len(self.catalog))
        self.dist = (self.catalog['parallax']).to(u.parsec, equivalencies=u.parallax())
        
    def getRVMembers(self):
        if (self.verbosity > 0):
            print("Finding radial-velocity members ... ")
        
        x = self.catalog['radial_velocity']
        ind = np.where(self.dist.to(u.parsec).value < self.dmax)
        mem = np.intersect1d(self.members, ind)                                                                      
        xm = x[mem]
        
        #1D histogram (use the members here)
        hrv, brv = np.histogram(xm, bins = self.RVbins, range=(self.RVmin, self.RVmax))
        
        #fit
        RVguess = brv[np.argmax(hrv)]
        if (self.RVmean != None):
            RVguess = self.RVmean
        p_init = models.Gaussian1D(np.max(hrv), RVguess, 1) \
                + models.Gaussian1D(5, brv[np.argmax(hrv)], 50)
        fit_p = fitting.LevMarLSQFitter()
        rvG1D = fit_p(p_init, brv[:-1], hrv)
        if (self.verbosity > 1):
            print(rvG1D)
            print(rvG1D.parameters)

        if (self.showPlots):
            hrv, brv = np.histogram(xm, bins = self.RVbins, range=(self.RVmin, self.RVmax))
            plt.step(brv[:-1],hrv)
            xf = np.linspace(self.RVmin, self.RVmax, self.RVbins*10)
            plt.plot(xf,rvG1D(xf), color='red')
            plt.xlabel(r'RV (km s$^{-1}$)', fontsize = 16)
            plt.show()
            
        #membership calculation
        Fc = models.Gaussian1D()
        Fc.parameters = rvG1D.parameters[0:3]
        PRV = Fc(x)/rvG1D(x)

        self.RVmembers = np.where(np.logical_and(PRV > self.minRVmembership, self.catalog['radial_velocity'].mask == False))
        membersRVAll = np.where(PRV > self.minRVmembership)
        self.members = np.intersect1d(self.members, membersRVAll)

    def getParallaxMembers(self):
        if (self.verbosity > 0):
            print("Finding parallax members ... ")
            
        x = self.dist.to(u.parsec).value
        ind = np.where(self.dist.to(u.parsec).value < self.dmax)
        mem = np.intersect1d(self.members, ind)                                                                      
        xm = x[mem]
        
        #1D histogram (use the members here)
        hpa, bpa = np.histogram(xm, bins = self.dbins, range=(self.dmin, self.dmax))

        #fit
        dguess = bpa[np.argmax(hpa)]
        if (self.distance != None):
            dguess = self.distance
        p_init = models.Gaussian1D(np.max(hpa), dguess, 10)\
                + models.Polynomial1D(degree=self.PaPolyD)
        fit_p = fitting.LevMarLSQFitter()
        pa1D = fit_p(p_init, bpa[:-1], hpa)
        if (self.verbosity > 1):
            print(pa1D)
            print(pa1D.parameters)

        if (self.showPlots):
            hpa, bpa = np.histogram(xm, bins = self.dbins, range=(self.dmin, self.dmax))
            plt.step(bpa[:-1],hpa)
            xf = np.linspace(self.dmin, self.dmax, self.dbins*10)
            plt.plot(xf,pa1D(xf), color='red')
            plt.xlabel('distance (pc)', fontsize = 16)
            plt.show()
            
        #membership calculation
        Fc = models.Gaussian1D()
        Fc.parameters = pa1D.parameters[0:3]
        Ppa = Fc(x)/pa1D(x)

        self.PAmembers = np.where(np.logical_and(Ppa > self.minPamembership, self.catalog['parallax'].mask == False))
        membersPaAll = np.where(Ppa > self.minPamembership)
        self.members = np.intersect1d(self.members, membersPaAll)
        
    def getPMMembers(self):
        if (self.verbosity > 0):
            print("finding proper-motion members ...")
        
        x = self.catalog['pmra']*np.cos(self.catalog['dec']*np.pi/180.)
        y = self.catalog['pmdec']
        ind = np.where(self.dist.to(u.parsec).value < self.dmax)
        mem = np.intersect1d(self.members, ind)                                                                      
        xm = x[mem]
        ym = y[mem]
        
        #1D histograms (use the members here)          
        pmRAbins = np.linspace(self.PMxmin, self.PMxmax, self.PMxbins)
        pmDecbins = np.linspace(self.PMymin, self.PMymax, self.PMybins)
        hx1D, x1D = np.histogram(xm, bins=pmRAbins)
        hy1D, y1D = np.histogram(ym, bins=pmDecbins)

        #2D histogram
        h2D, x2D, y2D = np.histogram2d(xm, ym, bins=[self.PMxbins, self.PMybins], \
                                       range=[[self.PMxmin, self.PMxmax], [self.PMymin, self.PMymax]])
        
        #fit
        PMxguess = x1D[np.argmax(hx1D)]
        PMyguess = y1D[np.argmax(hy1D)]
        if (self.PMmean[0] != None):
            PMxguess = self.PMmean[0]
        if (self.PMmean[1] != None):
            PMyguess = self.PMmean[1]
        p_init = models.Gaussian2D(np.max(h2D.flatten()), PMxguess, PMyguess, 1, 1)\
                + models.Gaussian2D(np.max(h2D.flatten()), 0, 0, 5, 5)
        fit_p = fitting.LevMarLSQFitter()
        xf, yf = np.meshgrid(x2D[:-1], y2D[:-1], indexing='ij')
        pmG2D = fit_p(p_init, xf, yf, h2D)
        if (self.verbosity > 1):
            print(pmG2D)
            print(pmG2D.parameters)
            
        if (self.showPlots):
            f = plt.figure(figsize=(8, 8)) 
            gs = gridspec.GridSpec(2, 2, height_ratios = [1, 3], width_ratios = [3, 1]) 
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[2])
            ax3 = plt.subplot(gs[3])

            #histograms
            hx1D, x1D = np.histogram(xm, bins=pmRAbins)
            hy1D, y1D = np.histogram(ym, bins=pmDecbins)
            ax1.step(x1D[:-1], hx1D)
            ax1.plot(x2D[:-1], np.sum(pmG2D(xf, yf), axis=1), color='red')
            ax3.step(hy1D, y1D[:-1])
            ax3.plot(np.sum(pmG2D(xf, yf), axis=0), y2D[:-1], color='red')

            #heatmap
            h2D, x2D, y2D, im = ax2.hist2d(xm, ym, bins=[self.PMxbins, self.PMybins],\
                                           range=[[self.PMxmin, self.PMxmax], [self.PMymin, self.PMymax]], \
                                           norm = mpl.colors.LogNorm(), cmap = cm.Blues)
#             ax2.contourf(x2D[:-1], y2D[:-1], pmG2D(xf, yf).T, cmap=cm.Reds, bins = 20, \
#                          norm=mpl.colors.LogNorm(), alpha = 0.3)

            ax1.set_xlim(self.PMxmin, self.PMxmax)
            ax2.set_xlim(self.PMxmin, self.PMxmax)
            ax2.set_ylim(self.PMymin, self.PMymax)
            ax3.set_ylim(self.PMymin, self.PMymax)
            ax1.set_yscale("log")
            ax1.set_ylim(1, 2*max(hx1D))
            ax3.set_xscale("log")
            ax3.set_xlim(1, 2*max(hy1D))
            ax2.set_xlabel(r'$\mu_\alpha \cos(\delta)$ (mas yr$^{-1}$)', fontsize=16)
            ax2.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)', fontsize=16)
            plt.setp(ax1.get_yticklabels()[0], visible=False)
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax3.get_yticklabels(), visible=False)
            plt.setp(ax3.get_xticklabels()[0], visible=False)
            f.subplots_adjust(hspace=0., wspace=0.)
            plt.show()

        #membership calculation
        Fc = models.Gaussian2D()
        Fc.parameters = pmG2D.parameters[0:6]
        PPM = Fc(x,y)/pmG2D(x,y)

        self.PMmembers = np.where(np.logical_and(PPM > self.minPMmembership, self.catalog['pmra'].mask == False))
        membersPMAll = np.where(PPM > self.minPMmembership)
        self.members = np.intersect1d(self.members, membersPMAll)
        
    def plotCMD(self):
        f = plt.figure(figsize=(5,6))
        plt.scatter(self.catalog['bp_rp'], self.catalog['phot_rp_mean_mag'], s = 1,  color='C0', alpha = 0.3, label='All')
        plt.scatter(self.catalog['bp_rp'][self.members], self.catalog['phot_rp_mean_mag'][self.members], s = 5, color='red',  label='Members')
        plt.legend()
        plt.xlim(self.CMDxmin, self.CMDxmax)
        plt.ylim(self.CMDymin, self.CMDymax)
        plt.xlabel('BP - RP (mag)', fontsize=16)
        plt.ylabel('RP (mag)', fontsize=16)
        plt.show()
        
        
    def runAll(self):
        self.getGaiaData()
        self.getRVMembers()
        self.getParallaxMembers()
        self.getPMMembers()
        if (self.showPlots):
            self.plotCMD()
        if (self.verbosity > 0):
            print("Done.")
            
            
            
            
            
            
            
#example
#M67
GM = GaiaClusterMembers()
GM.RA = 132.825
GM.Dec = 11.8167
GM.showPlots = True
GM.runAll()

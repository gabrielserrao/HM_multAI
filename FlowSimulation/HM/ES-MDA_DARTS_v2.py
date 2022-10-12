# %%
import os
from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from physics.physics_comp_sup import SuperPhysics
from physics.property_container import *
from physics.properties_dead_oil import *
from darts.engines import *
import numpy as np
import math
import os
import re
import shutil
import time
import matplotlib.pyplot as plt
import pandas as pd
#from main_darts import ModelLDA
import pickle


# %%
class ModelTrue(DartsModel):
    
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        # create reservoir
        self.nx = 60
        self.ny = 60
        self.nz = 1
        
        self.dx = 8
        self.dy = 8
        self.dz = 4

        # self.permx = np.ones(self.nx*self.ny)*1000
        # self.actnum = np.ones(self.nx*self.ny)
        
        self.permx = load_single_keyword('data/Egg/data.in','PERMX')
       
        self.actnum = load_single_keyword('data/Egg/data.in','ACTNUM')            
        
        self.permy = self.permx
        self.permz = 0.1 * self.permx
        self.poro = 0.2
        self.depth = 4000

        # run discretization
        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dy=self.dy,
                                         dz=self.dz, permx=self.permx, permy=self.permy, permz=self.permz,
                                         poro=self.poro, depth=self.depth,actnum=self.actnum)
        
        self.well_init()
               
        """Physical properties"""
        self.pvt = 'data/Egg/physics.in'
        self.zero = 1e-13
        self.property_container = model_properties(phases_name=['water', 'oil'], components_name=['w', 'o'], 
                                                   pvt=self.pvt, min_z=self.zero/10)

        # Define property evaluators based on custom properties
        self.flash_ev = []
        self.property_container.density_ev = dict([('water', DensityWat(self.pvt)),
                                                   ('oil', DensityOil(self.pvt))])
        self.property_container.viscosity_ev = dict([('water', ViscoWat(self.pvt)),
                                                     ('oil', ViscoOil(self.pvt))])
        self.property_container.rel_perm_ev = dict([('water', WatRelPerm(self.pvt)),
                                                    ('oil', OilRelPerm(self.pvt))])
        self.property_container.capillary_pressure_ev = CapillarypressurePcow(self.pvt)

        self.property_container.rock_compress_ev = RockCompactionEvaluator(self.pvt)

        # create physics
        self.thermal = 0
        self.physics = SuperPhysics(self.property_container, self.timer, n_points=400, min_p=0, max_p=1000,
                                     min_z=self.zero, max_z=1 - self.zero, thermal=self.thermal)

        self.params.first_ts = 0.01
        self.params.mult_ts = 2
        self.params.max_ts = 10
        self.params.tolerance_newton = 1e-2
        self.params.tolerance_linear = 1e-4

        self.inj = [0.999]

        self.runtime = 300

        self.timer.node["initialization"].stop()
        
    def well_init(self):
        
        # add two wells
        well_diam = 0.2
        well_rad = well_diam/2
           
        self.reservoir.add_well("INJ1")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 5, 57, 1, well_radius=well_rad, multi_segment=False)
        self.reservoir.inj_wells = [self.reservoir.wells[-1]]

        self.reservoir.add_well("INJ2")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 29, 53, 1, well_radius=well_rad, multi_segment=False)
        self.reservoir.inj_wells = [self.reservoir.wells[-1]]   

        self.reservoir.add_well("INJ3")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 4, 35, 1, well_radius=well_rad, multi_segment=False)
        self.reservoir.inj_wells = [self.reservoir.wells[-1]]   

        self.reservoir.add_well("INJ4")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 27, 29, 1, well_radius=well_rad, multi_segment=False)
          

        self.reservoir.add_well("INJ5")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 49, 35, 1, well_radius=well_rad, multi_segment=False)
           

        self.reservoir.add_well("INJ6")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 10, 9, 1, well_radius=well_rad, multi_segment=False)
          

        self.reservoir.add_well("INJ7")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 32, 3, 1, well_radius=well_rad, multi_segment=False)
        

        self.reservoir.add_well("INJ8")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 57, 7, 1, well_radius=well_rad, multi_segment=False)
        
        self.reservoir.inj_wells = [self.reservoir.wells[-1]]   

        self.reservoir.add_well("PRD1")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 16, 43, 1, well_radius=well_rad, multi_segment=False)
        

        self.reservoir.add_well("PRD2")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 35, 40, 1, well_radius=well_rad, multi_segment=False)
        

        self.reservoir.add_well("PRD3")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 23, 16, 1, well_radius=well_rad, multi_segment=False)
        

        self.reservoir.add_well("PRD4")
        self.reservoir.add_perforation(self.reservoir.wells[-1], 43, 18, 1, well_radius=well_rad, multi_segment=False)
        
        self.reservoir.prod_wells = [self.reservoir.wells[-1]]


    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=400,
                                                    uniform_composition=[2e-2])

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ" in w.name:
                w.control = self.physics.new_rate_inj(200, self.inj, 0)
                w.constraint = self.physics.new_bhp_inj(450, self.inj)
            else:
                w.control = self.physics.new_bhp_prod(390)


# evaluate all properties which are needed in the simulation
class model_properties(property_container):
    def __init__(self, phases_name, components_name, pvt, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, min_z)
        self.x = np.zeros((self.nph, self.nc))
        self.pvt = pvt
        self.surf_dens = get_table_keyword(self.pvt, 'DENSITY')[0]
        self.surf_oil_dens = self.surf_dens[0]
        self.surf_wat_dens = self.surf_dens[1]

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        ph = [0, 1]

        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(state)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(state)  # output in [cp]

        self.nu = zc
        self.compute_saturation(ph)
        
        # when evaluate rel-perm based on the table, we only need water saturation to interpolate both phases saturation
        for j in ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[0])
            self.pc[j] = 0

        return self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, ph

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        state = value_vector([1, 0])

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(state)

        self.dens_m = [self.surf_wat_dens, self.surf_oil_dens]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m


m_true = ModelTrue()
m_true.init();
time_range = np.arange(0,1000, 10)+1
m_true.run_python(time_range[-1]);


time_data = pd.DataFrame.from_dict(m_true.physics.engine.time_data)
# wirte timedata to output file
time_data.to_pickle("darts_time_data.pkl")
# write timedata to excel file
writer = pd.ExcelWriter('time_data.xlsx')
time_data.to_excel(writer, 'Sheet1')
writer.save()


#calculate the reference to compare models
def media_function(time_range, time_data, data_type):
    media =[]
    i= 1
    while i<=(len(time_range)-1):
        s=[]
        for j,tempo in enumerate(time_data['time']):
                if time_range[i-1]<time_data['time'][j]<=time_range[i]:
                    s.append(time_data[data_type][j])
                    np.mean(s)
        media.append(np.mean(s))
        i=i+1
    return media


#Create hard data from the true simulation response 

reference_dataSet = time_data #pd.read_pickle(f'{srcDir}')

#data columns
columnsNameList = ['PRD1 : oil rate (m3/day)', 'PRD2 : oil rate (m3/day)', 'PRD3 : oil rate (m3/day)', 'PRD4 : oil rate (m3/day)']

obsValues=[]
for i, data_type in enumerate(columnsNameList):
  obsValues.append(-1*np.array(media_function(time_range, reference_dataSet, data_type)))
dTime= np.array(time_range)
dObs=np.concatenate(obsValues, axis=0)
dObs = dObs.T.flatten()

#add noise to make the problem realistic
#np.random.seed(10)
#CeDiag = np.random.normal(0,0.1*dObs[:],len(dObs[:]))
CeDiag =np.array(0.05*dObs[:]) #diagonal of the covariance matrix of observed data

#Nd = len(dObs)

NTimesteps=len(dTime)-1
# Configure the wells list
wells = ['PRD1','PRD2','PRD3','PRD4']
wellDObs = np.repeat(wells, NTimesteps)


# ## Data Assimilation


# Define the problem dimensions

# %%
# problem dimensions
Ni = m_true.nx
Nj = m_true.ny
NGrid = Ni * Nj
NScalar = 0 #we are not considering any scalar parameters in the problem like kro, krw 
Nm = NGrid + NScalar
Nd = len(dObs)  #len(dTime)* obsValues.shape[0] #  timesteps * 4 well datas
Ne = 17
NTimesteps = len(dTime)
NWells = len(wells)

# Covariogram parameters 
L = (10,10) 
theta = 45 * np.pi/180 #degrees
sigmaPr2 = 1.0

# Localization parameters
locL = (10,10)
locTheta = 45 * np.pi/180

# svd truncation parameter for SVD 
csi = 0.99




# Rotate coordinates and flattens the matrix to an array
def CalcHL(x0, x1, L, theta):
    cosT = np.cos(theta)
    sinT = np.sin(theta)
    dx = x1[0] - x0[0]
    dy = x1[1] - x0[1]

    dxRot = np.array([[cosT, -sinT], [sinT, cosT]]) @ np.array([[dx], [dy]])
    dxFlat = dxRot.flatten()

    return np.sqrt((dxFlat[0]/L[0])**2 + (dxFlat[1]/L[1])**2)

# Calc covariance between two gridblocks
def SphereFunction(x0, x1, L, theta, sigmaPr2):
    hl = CalcHL(x0, x1, L, theta)

    if (hl > 1):
        return 0
    
    return sigmaPr2 * (1.0 - 3.0/2.0*hl + (hl**3)/2)

def GaspariCohnFunction(x0, x1, L, theta):
    hl = CalcHL(x0, x1, L, theta)

    if (hl < 1):
        return -(hl**5)/4. + (hl**4)/2. + (hl**3)*5./8. - (hl**2)*5./3. + 1.
    if (hl >= 1 and hl < 2):
        return (hl**5)/12. - (hl**4)/2. + (hl**3)*5./8. + (hl**2)*5./3. - hl*5 + 4 - (1/hl)*2./3.
    
    return 0

# convert index numeration to I J index
def IndexToIJ(index, ni, nj):
    return ((index % ni) + 1, (index // ni) + 1)

# Convert i J numeration to index
def IJToIndex(i,j,ni,nj):
    return (i-1) + (j-1)*ni

def BuildPermCovMatrix(Ni, Nj, L, theta, sigmaPr2):
    Nmatrix = Ni * Nj
    Cm = np.empty([Nmatrix, Nmatrix])
    for index0 in range(Nmatrix):
        I0 = IndexToIJ(index0,Ni,Nj)
        for index1 in range(Nmatrix):
            I1 = IndexToIJ(index1,Ni,Nj)
            Cm[index0, index1] = SphereFunction(I0, I1, L, theta, sigmaPr2)
    return Cm

# Builds the localization matrix. wellPos is a list with tuples, 
# each corresponding with the position of the data
def BuildLocalizationMatrix(Ni, Nj, wellPos, L, theta):
    Npos = Ni * Nj
    Nd = len(wellPos)
    Rmd = np.ones([Npos,Nd])
    for i in range(Npos):
        # Get the index of the cell
        Im = IndexToIJ(i, Ni, Nj)
        
        for j in range(Nd):
            Iw = wellPos[j]

            Rmd[i, j] = GaspariCohnFunction(Im, Iw, L, theta)
    return Rmd

def PlotModelRealization(m, title, axis, vmin=None, vmax=None):
    return PlotMatrix(m.reshape((Ni,Nj),order='F').T, title, axis, vmin, vmax)

def PlotMatrix(matrix, title, axis, vmin=None, vmax=None):
    axis.set_title(title)
    return axis.imshow(matrix, cmap='RdYlGn_r', vmin=vmin, vmax=vmax, aspect='auto')



# %%
# ## 1. Generate the prior ensemble of Grid parameters
# Generate the covariance matrix
Cgrid = BuildPermCovMatrix(Ni, Nj, L, theta, sigmaPr2) 

fig, ax = plt.subplots()
im = PlotMatrix(Cgrid, 'Matriz de covariancia (CGrid)', ax)
fig.colorbar(im, ax=ax)

# Generate the ensembles
mpr = np.full((NGrid,1),5.0)

lCholesky = np.linalg.cholesky(Cgrid)
mList = []
for i in range(Ne):
    z = np.random.normal(size=(NGrid,1))
    mList.append(mpr + lCholesky @ z)
MGridPrior = np.transpose(np.array(mList).reshape((Ne,NGrid)))

fig, ax = plt.subplots(nrows=1, ncols=2)
im = PlotModelRealization(MGridPrior[:,0], 'realization 0', ax[0], vmin=2.5, vmax=7.5)
PlotModelRealization(MGridPrior[:,1], 'realization 1', ax[1], vmin=2.5, vmax=7.5)
fig.colorbar(im, ax=ax)


# ## 2. Generate the prior ensemble of scalar parameters - here we are not
#krw = np.clip(np.random.normal(0.5, 0.1, Ne), 0.2, 0.8)
#ew = np.clip(np.random.normal(2.0, 0.3, Ne), 1.0, 3.0)
#eo = np.clip(np.random.normal(3.0, 0.3, Ne), 2.0, 4.0)
    
MScalarPrior = [] #np.stack((krw,ew,eo))

# %%
def RunModels(destDir, MGrid, MScalar):
    for j,mGridColumn in enumerate(MGrid.T):
        job_file = os.path.join(destDir,f'data_model{j}') 
        #grid = f'grid_{job_file}'
        print(job_file)

        mGridColumn = pd.DataFrame(mGridColumn)
        mGridColumn.to_pickle(f'{job_file}_grid.pkl')
        
        print(f'Rodando modelo {j}')
        with open(job_file, 'w+') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --job-name=%s.job\n" % j)
            fh.writelines("#SBATCH --output=%s.out\n" % job_file)
            fh.writelines("#SBATCH --error=%s.err\n" % job_file)
            fh.writelines("#SBATCH --time=2:00:00\n")
            fh.writelines("#SBATCH --mem=1G\n")
            fh.writelines("#SBATCH --partition=compute\n")
            fh.writelines("#SBATCH --account=research-ceg-gse\n")
            fh.writelines("module load 2022r2\n")
            fh.writelines(f'a={int(j)}\n')
            fh.writelines(f'b={str(destDir)}\n')
            fh.writelines(f'c={str(time_range[-1])}\n')
            fh.writelines(f'python main_darts.py $a $b $c\n')

        os.system(f'sbatch {job_file}') #{str(j)} {str(destDir)} {str(time_range[-1])} ') 

def check_job(filename):
    while not os.path.exists(filename):
        time.sleep(2)
        print('waiting for job last to finish')
    return filename        
    
#Read the result from the model
def ReadModels(destDir, columnsNameList, Nd, Ne):
    D = np.empty([Nd, Ne])
    for i in range(Ne):
        dataSet = pd.read_pickle(f'{destDir}/data_model'+str(i)+'.pkl') 
        model_value=[]
        d_models=[]
        for j, data_type in enumerate(columnsNameList):
            model_value.append(-1*np.array(media_function(time_range, dataSet, data_type)))
        
        d_models = np.concatenate(model_value, axis=0)
        d_models = d_models.T.flatten()    
           
        D[:,i] = d_models 

    return D


# Finds the truncation number
def FindTruncationNumber(Sigma, csi):
    temp = 0
    i = 0
    svSum = np.sum(Sigma)
    stopValue = svSum * csi
    for sv in np.nditer(Sigma):
        if (temp >= stopValue):
            break
        temp += sv
        i += 1
    return i

def CentralizeMatrix(M):
    meanMatrix = np.mean(M, axis=1)
    return M - meanMatrix[:,np.newaxis]

# Psi = X9 in (12.23)
def UpdateModelLocalized(M, Psi, R, DobsD):
    DeltaM = CentralizeMatrix(M)
   
    K = DeltaM @ Psi
    Kloc = R * K
    return M + Kloc @ DobsD

def UpdateModel(M, Psi, DobsD):
    DeltaM = CentralizeMatrix(M)

    X10 = Psi @ DobsD
    return M + DeltaM @ X10

def calcDataMismatchObjectiveFunction(dObs, D, CeInv):
    Ne = D.shape[1]
    Nd = D.shape[0]

    Od = np.empty(Ne)
    for i in range(Ne):
        dObsD = dObs - D[:,i].reshape(Nd,1)
        Od[i] = (dObsD.T) @ (CeInv[:,np.newaxis] * dObsD)/2
    return Od

# Replaces the pattern with the value in array cosrresponding its position.
# Only 1 group per line for now...
def ReplacePattern(matchobj, array):
    return f'{array[int(matchobj.group(1))]:.2f}' 



#%%
# Read the well from the file
wellDataFull = pd.read_csv('data/hard_data.txt', sep='\t')
wellDataDictionary = wellDataFull[['WELL','I','J']].set_index('WELL').T.to_dict('list')

wellPos = [wellDataDictionary[well] for well in wellDObs]

# Builds the localization function
Rmd = BuildLocalizationMatrix(Ni, Nj, wellPos, locL, locTheta)


Rmd.shape


# ## Generate posterior realizations using ES-MDA


def analyseFunction(R, M, D, Dobs, Ce, alpha):
    cNe = D.shape[1]
    cNd = D.shape[0]
    ccsi = 0.99

    SDiagFunc = np.sqrt(Ce)
    SInvDiagFunc = np.power(SDiagFunc, -1)

    DD = Dobs - D

    # 4. Analysis
    # 4.1 Invert matrix C

    # Calculates DeltaD (12.5)
    meanMatrix = np.mean(D, axis=1)
    DeltaD = D - meanMatrix[:,np.newaxis]

    # Calculates CHat (12.10)
    Ind = np.eye(cNd)
    CHat = SInvDiagFunc[:,np.newaxis] * ( DeltaD @ DeltaD.T ) * SInvDiagFunc[np.newaxis,:] + alpha * (cNe - 1) * Ind

    # Calculates Gamma and X (12.18)
    U, SigmaDiag, Vt = np.linalg.svd(CHat)
    Nr = FindTruncationNumber(SigmaDiag, ccsi)

    GammaDiag = np.power(SigmaDiag[0:Nr], -1)
    X = SInvDiagFunc[:,np.newaxis] * U[:,0:Nr]

    # Calculates M^a (12.21)
    X1 = GammaDiag[:,np.newaxis] * X.T
    X8 = DeltaD.T @ X
    X9 = X8 @ X1
    
    # subpart: for grid, use localization
    return UpdateModelLocalized(M, X9, R, DD)

# %% [markdown]
# Run ES-MDA

curDir = os.getcwd()
srcDir =  f'{curDir}'
srcDir


SDiag = np.sqrt(CeDiag)
SInvDiag = np.power(SDiag, -1)

INd = np.eye(Nd)

MGrid = MGridPrior
MScalar = MScalarPrior

alphas = [4., 4., 4., 4.]
l = 0
for alpha in alphas:
    # 2. Forecast

    # Generates the perturbed observations (10.27)
    z = np.random.normal(size=(Nd, Ne))
    DPObs = dObs[:, np.newaxis] + math.sqrt(alpha) * CeDiag[:, np.newaxis] * z
    

    # Run the simulations g(M) (12.4)
    destDir = f'/tudelft.net/staff-umbrella/gabrielserrao/FlowSimulation/HM/data/simulations/it{l}'
    RunModels(destDir, MGrid, MScalar)
    time.sleep(10)
    filename = f'/tudelft.net/staff-umbrella/gabrielserrao/FlowSimulation/HM/data/simulations/it{l}/data_model'+str(Ne-1)+'.pkl'
    check_job(filename)
    print(f'Finished iteration {l} runs')
    D = ReadModels(destDir, columnsNameList, Nd, Ne)
    print(f'Finished iteration {l} read')

    if (l == 0):
        DPrior = D

    DobsD = DPObs - D

    # 4. Analysis
    # 4.1 Invert matrix C

    # Calculates DeltaD (12.5)
    meanMatrix = np.mean(D, axis=1)
    DeltaD = D - meanMatrix[:, np.newaxis]

    # Calculates CHat (12.10)
    CHat = SInvDiag[:, np.newaxis] * \
        (DeltaD @ DeltaD.T) * \
        SInvDiag[np.newaxis, :] + alpha * (Ne - 1) * INd

    # Calculates Gamma and X (12.18)
    U, SigmaDiag, Vt = np.linalg.svd(CHat)
    Nr = FindTruncationNumber(SigmaDiag, csi)

    GammaDiag = np.power(SigmaDiag[0:Nr], -1)
    X = SInvDiag[:, np.newaxis] * U[:, 0:Nr]

    # Calculates M^a (12.21)
    X1 = GammaDiag[:, np.newaxis] * X.T
    X8 = DeltaD.T @ X
    X9 = X8 @ X1

    # subpart: for grid, use localization
    MGrid = UpdateModelLocalized(MGrid, X9, Rmd, DobsD)
    #don't have scalars
    # subpart: for scalars, don't use localization
    ##MScalar = UpdateModel(MScalar, X9, DobsD)
    ### clip values
    ##MScalar[0, :] = np.clip(MScalar[0, :], 0.2, 0.8)
    ##MScalar[1, :] = np.clip(MScalar[1, :], 1.0, 3.0)
    ##MScalar[2, :] = np.clip(MScalar[2, :], 2.0, 4.0)
    l += 1

MGridPost = MGrid
MScalarPost = MScalar
DPost = D


# %%
plt.plot(dObs, marker='o')
plt.plot(DPrior[:,2])

# %% [markdown]
# ## Report
# 

# %%
# ## 6. Report

#%%
# Comparison between prior and posterior grid values
# mean
priorMean = np.mean(MGridPrior, axis=1)
postMean = np.mean(MGridPost, axis=1)

# std
priorStd = np.std(MGridPrior, axis=1, ddof=1)
postStd = np.std(MGridPost, axis=1, ddof=1)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7,10))
im0 = PlotModelRealization(priorMean, 'Prior mean', ax[0,0], vmin=2.5, vmax=7.5)
im1 = PlotModelRealization(postMean, 'Posterior mean', ax[1,0], vmin=2.5, vmax=7.5)
im2 = PlotModelRealization(priorStd, 'Prior std', ax[0,1], vmin=0.0, vmax=1.5)
im3 = PlotModelRealization(postStd, 'Posterior std', ax[1,1], vmin=0.0, vmax=1.5)
fig.colorbar(im0, ax=ax[0,0])
fig.colorbar(im1, ax=ax[1,0])
fig.colorbar(im2, ax=ax[0,1])
fig.colorbar(im3, ax=ax[1,1])

# Plot of the first three realizations
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,10))
im = PlotModelRealization(MGridPrior[:,0], 'Prior 0', ax[0,0], vmin=2.5, vmax=7.5)
PlotModelRealization(MGridPrior[:,1], 'Prior 1', ax[0,1], vmin=2.5, vmax=7.5)
PlotModelRealization(MGridPrior[:,2], 'Prior 2', ax[0,2], vmin=2.5, vmax=7.5)
PlotModelRealization(MGridPost[:,0], 'Post 0', ax[1,0], vmin=2.5, vmax=7.5)
PlotModelRealization(MGridPost[:,1], 'Post 1', ax[1,1], vmin=2.5, vmax=7.5)
PlotModelRealization(MGridPost[:,2], 'Post 2', ax[1,2], vmin=2.5, vmax=7.5)
fig.colorbar(im, ax=ax)
#save figure
fig.savefig('grid.png')

#%%
# Comparison between prior and posterior scalar values
# prior values
#mean = np.mean(MScalarPrior, axis=1)
#std = np.std(MScalarPrior, axis=1, ddof=1)

#print(f'Prior mean: krw={mean[0]:.3f}, ew={mean[1]:.3f}, eo={mean[2]:.3f}')
#print(f'Prior std:  krw={std[0]:.3f}, ew={std[1]:.3f}, eo={std[2]:.3f}')

# posterior values
#mean = np.mean(MScalarPost, axis=1)
#std = np.std(MScalarPost, axis=1, ddof=1)

#print(f'Posterior mean: krw={mean[0]:.3f}, ew={mean[1]:.3f}, eo={mean[2]:.3f}')
#print(f'Posterior std:  krw={std[0]:.3f}, ew={std[1]:.3f}, eo={std[2]:.3f}')

# histogram
#fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
#for i in range(3):
#    ax[0, i].hist(MScalarPrior[i,:], bins=20, density=True, facecolor='g', alpha=0.8)
#    ax[1, i].hist(MScalarPost[i,:], bins=20, density=True, facecolor='g', alpha=0.8)
#fig.suptitle('Histograms')
#ax[0, 0].set_title('Prior krw')
#ax[0, 1].set_title('Prior ew')
#ax[0, 2].set_title('Prior eo')
#ax[1, 0].set_title('Posterior krw')
#ax[1, 1].set_title('Posterior ew')
#ax[1, 2].set_title('Posterior eo')



# # Comparison of data mismatch objective function
# # prior (l = 0)
CeInv = np.power(CeDiag, -1)
OPrior = calcDataMismatchObjectiveFunction(dObs[:,np.newaxis], DPrior, CeInv)
OPost = calcDataMismatchObjectiveFunction(dObs[:,np.newaxis], DPost, CeInv)

print(f'Mean: prior={np.mean(OPrior):.3e}, post={np.mean(OPost):.3e}')
print(f'Std: prior={np.std(OPrior, ddof=1):.3e}, post={np.std(OPost, ddof=1):.3e}')

fig, ax = plt.subplots(figsize=(Ne,Ne))
colors=['red','green']
x = np.stack((OPrior, OPost), axis=1)
#ax.hist(x, bins=40, alpha=0.8, color=colors, range=(x.min(),1e7))
ax.hist(x, bins=40, alpha=0.8, color=colors)
ax.set_title('Histograms Objetive Functions')


# %%
#%%
# Plots of oil and water rate at wells
# Split the data into data array
wellDataPriorArray = np.split(DPrior,  NWells)
wellDataPostArray = np.split(DPost, NWells)
wellDataObsArray = np.split(dObs, NWells)
wellDataObsErrorArray = np.split(CeDiag, NWells)

wellSubtitle = ['PRD1', 'PRD2', 'PRD3', 'PRD4']
timeMonths = dTime[:-1] #/ 30.0

# Plot oil data
# get data and store in an array
prior = np.array(wellDataPriorArray) # wellDataPriorArray[0:11:2]
post = np.array(wellDataPostArray) # wellDataPostArray[0:11:2]
plotObsData = wellDataObsArray #wellDataObsArray[0:11:2]
errObsData = wellDataObsErrorArray #wellDataObsErrorArray[0:11:2]

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,8))
axFlat = ax.flatten()
for i in range(NWells):
    axFlat[i].set_facecolor('white')
    axFlat[i].plot(timeMonths, prior[i,:], color='#00FFFF', alpha=0.5)
    axFlat[i].plot(timeMonths, post[i,:], color='black', alpha=0.99)
    axFlat[i].errorbar(timeMonths, plotObsData[i], yerr=errObsData[i], fmt='_ r', capthick=1, capsize=3)
    axFlat[i].set_title(wellSubtitle[i])
fig.suptitle('Oil rate')
#save figure
fig.savefig('oil.png')



# # Plot water data
# #prior = wellDataPriorArray[1:12:2]
# #post = wellDataPostArray[1:12:2]
# #plotObsData = wellDataObsArray[1:12:2]
# #errObsData = wellDataObsErrorArray[1:12:2]

# #fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
# #axFlat = ax.flatten()
# #for i in range(6):
# #    axFlat[i].set_facecolor('black')
# #    axFlat[i].plot(timeMonths, prior[i], color='white', alpha=0.2)
# #    axFlat[i].plot(timeMonths, post[i], color='green', alpha=0.2)
# #    axFlat[i].errorbar(timeMonths, plotObsData[i], yerr=errObsData[i], fmt='_ r', capthick=2, capsize=3)
# #    axFlat[i].set_title(wellSubtitle[i])
# #fig.suptitle('Water rate')

# # %%


# # %%
# #plt.plot(prior[0,:])
# plt.plot(post[0,:])

# # %%
# prior[0].shape

# # %%




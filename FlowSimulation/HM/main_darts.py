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

class ModelLDA(DartsModel):
    
    def __init__(self, perm):
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
        
        #self.permx = load_single_keyword('data/Egg/data.in','PERMX')
        self.permx=perm
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

import sys
#{j} {destDir} {time_range[-1]} \n')
j = int(sys.argv[1])
destDir = str(sys.argv[2])
t= int(sys.argv[3])
print(j, destDir, t)



job_file = os.path.join(destDir,f'data_model{j}') 
grid = f'{job_file}_grid.pkl'

mGridColumn = np.array(pd.read_pickle(f'{grid}')).reshape([60,60,1])
print(f'rodando job {j}')
print(f'{os.getcwd()}')
modelo=ModelLDA(perm = np.exp(mGridColumn))
modelo.init()
modelo.run(t)
data = pd.DataFrame.from_dict(modelo.physics.engine.time_data)
#write timedata to output file
#data.to_pickle(f'{destDir}/data_model'+str(j)+'.pkl')
os.chdir(destDir)
data.to_pickle(f'data_model'+str(j)+'.pkl')
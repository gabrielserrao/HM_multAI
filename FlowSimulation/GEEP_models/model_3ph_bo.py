from physics.physics_comp_sup import SuperPhysics
from model_base import BaseModel
from darts.engines import sim_params
import numpy as np
from physics.properties_black_oil import *
from physics.property_container import *

# Model class creation here!
class Model(BaseModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        self.zero = 1e-12
        """Physical properties"""
        # Create property containers:
        self.pvt = 'physics_bo.in'
        self.property_container = model_properties(phases_name=['gas', 'oil', 'wat'], components_name=['g', 'o', 'w'],
                                                   pvt=self.pvt, min_z=self.zero / 10)

        self.phases = self.property_container.phases_name

        """ properties correlations """
        self.property_container.flash_ev = flash_black_oil(self.pvt)
        self.property_container.density_ev = dict([('gas', DensityGas(self.pvt)),
                                                   ('oil', DensityOil(self.pvt)),
                                                   ('wat', DensityWat(self.pvt))])
        self.property_container.viscosity_ev = dict([('gas', ViscGas(self.pvt)),
                                                     ('oil', ViscOil(self.pvt)),
                                                     ('wat', ViscWat(self.pvt))])
        self.property_container.rel_perm_ev = dict([('gas', GasRelPerm(self.pvt)),
                                                    ('oil', OilRelPerm(self.pvt)),
                                                    ('wat', WatRelPerm(self.pvt))])
        self.property_container.capillary_pressure_ev = dict([('pcow', CapillaryPressurePcow(self.pvt)),
                                                              ('pcgo', CapillaryPressurePcgo(self.pvt))])

        self.property_container.rock_compress_ev = RockCompactionEvaluator(self.pvt)

        """ Activate physics """
        self.physics = SuperPhysics(self.property_container, self.timer, n_points=500, min_p=0, max_p=450,
                                    min_z=self.zero / 10, max_z=1 - self.zero / 10)

        self.inj_stream = [1e-8, 1e-8]
        self.ini_stream = [0.07296, 0.4832]

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 0.001
        self.params.mult_ts = 2
        self.params.max_ts = 15
        self.params.tolerance_newton = 1e-2
        self.params.tolerance_linear = 1e-3

        self.params.max_i_newton = 20
        self.params.max_i_linear = 30
        self.params.newton_type = sim_params.newton_local_chop
        self.params.nonlinear_norm_type = sim_params.L1

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 320, self.ini_stream)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if w.name[0:3] == 'INJ':
                w.control = self.physics.new_bhp_inj(400, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(250)

    def export_pro_vtk(self, file_name='Saturation'):
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[0:self.reservoir.nb * 3:3]
        z1 = Xn[1:self.reservoir.nb * 3:3]
        z2 = Xn[2:self.reservoir.nb * 3:3]

        sg = np.zeros(len(P))
        so = np.zeros(len(P))
        sw = np.zeros(len(P))

        for i in range(len(P)):
            values = value_vector([0] * self.physics.n_ops)
            state = value_vector((P[i], z1[i], z2[i]))
            self.physics.property_itor.evaluate(state, values)
            sg[i] = values[0]
            so[i] = values[1]
            sw[i] = 1 - sg[i] - so[i]

        self.export_vtk(file_name, local_cell_data={'GasSat': sg, 'OilSat': so, 'WatSat': sw})

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_w_itor]


class model_properties(property_container):
    def __init__(self, phases_name, components_name, pvt, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        self.pvt = pvt
        super().__init__(phases_name, components_name, Mw, min_z)
        self.pvt = pvt
        self.surf_dens = get_table_keyword(self.pvt, 'DENSITY')[0]
        self.surf_oil_dens = self.surf_dens[0]
        self.surf_wat_dens = self.surf_dens[1]
        self.surf_gas_dens = self.surf_dens[2]

        self.x = np.zeros((self.nph, self.nc))

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

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        (xgo, V, pbub) = self.flash_ev.evaluate(pressure, zc)

        for i in range(self.nph):
            self.x[i, i] = 1

        if V < 0:
            ph = [1, 2]
        else:  # assume oil and water are always exists
            ph = [0, 1, 2]

        self.x[1][0] = xgo
        self.x[1][1] = 1 - xgo

        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, pbub, xgo)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, pbub)  # output in [cp]

        self.nu[2] = zc[2]
        # two phase undersaturated condition
        if pressure > pbub:
            self.nu[0] = 0
            self.nu[1] = zc[1]
        else:
            self.nu[1] = zc[1] / (1 - xgo)
            self.nu[0] = 1 - self.nu[1] - self.nu[2]

        self.compute_saturation(ph)

        for j in ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[0], self.sat[2])

        pcow = self.capillary_pressure_ev['pcow'].evaluate(self.sat[2])
        pcgo = self.capillary_pressure_ev['pcgo'].evaluate(self.sat[0])

        self.pc = np.array([-pcgo, 0, pcow])

        return self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, ph

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        ph = []
        for j in range(self.nph):
            if zc[j] > self.min_z:
                ph.append(j)
            self.dens_m[j] = self.density_ev[self.phases_name[j]].dens_sc

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m



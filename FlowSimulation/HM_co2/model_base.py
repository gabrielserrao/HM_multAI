from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
import numpy as np
from darts.tools.keyword_file_tools import load_single_keyword, save_few_keywords
import os


class BaseModel(DartsModel):
    def __init__(self, n_points=1000):
        # call base class constructor
        super().__init__()
        self.n_points = n_points
        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        # create reservoir from UNISIM - 20 layers (81*58*20, Corner-point grid)
        self.permx = 100 #load_single_keyword('reservoir.in', 'PERMX')
        self.permy = self.permx #load_single_keyword('reservoir.in', 'PERMY')
        self.permz = 0.1* self.permx#load_single_keyword('reservoir.in', 'PERMZ')
        self.poro = 0.1 #load_single_keyword('reservoir.in', 'PORO')
        self.depth = 1000 #load_single_keyword('reservoir.in', 'DEPTH')

        if os.path.exists(('width.in')):
            #print('Reading dx, dy and dz specifications...')
            self.dx = 100 #load_single_keyword('width.in', 'DX')
            self.dy = 100#load_single_keyword('width.in', 'DY')
            self.dz = 1 #load_single_keyword('width.in', 'DZ')

        # Import other properties from files
        #filename = 'grid.grdecl'
        self.actnum = 1 #load_single_keyword(filename, 'ACTNUM')
        #self.coord = load_single_keyword(filename, 'COORD')
        #self.zcorn = load_single_keyword(filename, 'ZCORN')

        is_CPG = False  # True for re-calculation of dx, dy and dz from CPG grid

        self.reservoir = StructReservoir(self.timer, nx=81, ny=58, nz=20, dx=self.dx, dy=self.dy, dz=self.dz,
                                         permx=self.permx, permy=self.permy, permz=self.permz, poro=self.poro,
                                         depth=self.depth, actnum=self.actnum) #, coord=self.coord, zcorn=self.zcorn,
                                         #is_cpg=is_CPG)

        poro = np.array(self.reservoir.mesh.poro, copy=False)
        poro[poro == 0.0] = 1.E-4

        if is_CPG:
            dx, dy, dz = self.reservoir.get_cell_cpg_widths()
            save_few_keywords('width.in', ['DX', 'DY', 'DZ'], [dx, dy, dz])

        well_dia = 0.152
        well_rad = well_dia / 2


        keep_reading = True
        prev_well_name = ''
        with open('WELLS.INC') as f:
            while keep_reading:
                buff = f.readline()
                if 'COMPDAT' in buff:
                    while True:  # be careful here
                        buff = f.readline()
                        if len(buff) != 0:
                            CompDat = buff.split()

                            if len(CompDat) != 0 and '/' != CompDat[0]:  # skip the empty line and '/' line

                                # define well
                                if CompDat[0] == prev_well_name:
                                    pass
                                else:
                                    self.reservoir.add_well(CompDat[0], wellbore_diameter=well_dia)
                                    prev_well_name = CompDat[0]

                                # define perforation
                                for i in range(int(CompDat[3]), int(CompDat[4]) + 1):
                                    self.reservoir.add_perforation(self.reservoir.wells[-1],
                                                                   int(CompDat[1]), int(CompDat[2]), i,
                                                                   well_radius=well_rad,
                                                                   multi_segment=False)

                            if len(CompDat) != 0 and '/' == CompDat[0]:
                                keep_reading = False
                                break


        self.timer.node["initialization"].stop()

    def wells4ParaView(self):
        name = []
        type = []
        ix = []
        iy = []
        keep_reading = True
        with open('WELLS.INC') as f:
            while keep_reading:
                buff = f.readline()
                if 'WELSPECS' in buff:
                    while True:  # be careful here
                        buff = f.readline()
                        if len(buff) != 0:
                            welspecs = buff.split()

                            if len(welspecs) != 0 and welspecs[0] != '/' and welspecs[0][:2] != '--':  # skip the empty line and '/' line
                                name += [welspecs[0]]
                                if 'GROUP1' in welspecs[1]:
                                    type += ['PRD']
                                else:
                                    type += ['INJ']
                                ix += [welspecs[2]]
                                iy += [welspecs[3]]
                                # define perforation

                            if len(welspecs) != 0 and welspecs[0] == '/':
                                keep_reading = False
                                break
        f.close()

        def str2file(fp, name_in, list_in):
            fp.write("%s = [" % name_in)
            for item in list_in:
                fp.write("\'%s\', " % item)
            fp.write("]\n")

        def num2file(fp, name_in, list_in):
            fp.write("%s = [" % name_in)
            for item in list_in:
                fp.write("%d, " % int(item))
            fp.write("]\n")

        f = open('well_gen.txt', 'w')
        str2file(f, 'well_list', name)
        str2file(f, 'well_type', type)
        num2file(f, 'well_x', ix)
        num2file(f, 'well_y', iy)
        f.close()
        print('done')
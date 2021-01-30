import numpy as np
import openmdao.api as om

from scipy.interpolate import interp1d
from scipy.constants import degree

from toa.data import Airplane
from toa.models.aero.graphs import ar_areff_x
from toa.models.aero.graphs import ar_areff_y


class GroundEffectComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='h', val=np.zeros(nn), desc='Airplane altitude from runway level', units='m')

        self.add_output(name="CLag", val=np.zeros(nn), desc='CLa variation due to ground effect', units='1/rad')
        self.add_output(name='dalpha_zero', val=np.zeros(nn), desc='Alpha zero CL variation due to ground effect', units='deg')
        self.add_output(name='phi', val=np.zeros(nn), desc='Induced drag variation due to ground effect', units=None)

        self.declare_partials(of='CLag', wrt='h', method='fd')
        self.declare_partials(of='dalpha_zero', wrt='h', method='fd')
        self.declare_partials(of='phi', wrt='h', method='fd')

    def compute(self, inputs, outputs,**kwargs):
        airplane = self.options['airplane']
        h = inputs['h']

        h_b = h/airplane.wing.span
        h_c = h/airplane.wing.mac
        # CLalpha
        if 2*h_b <= 2:
            ar_areff = interp1d(ar_areff_x, ar_areff_y, kind='cubic')(2 * h_b)
        else:
            ar_areff = 1
        ar = airplane.wing.span ** 2 / airplane.wing.area
        areff = ar / ar_areff
        beta = 1
        kaff = airplane.coeffs.cla / (2 * np.pi)
        a1 = (areff ** 2 * beta ** 2 / kaff ** 2)
        a2 = (1 + np.tan(airplane.wing.sweep_12 * degree) ** 2 / beta ** 2)
        CLag = 2 * np.pi * areff / (2 + (a1 * a2 + 4) ** 0.5)

        # Alpha0
        dalpha_zero = airplane.wing.t_c * (-0.1177 * (1 / h_c ** 2) + 3.5655 * (1 / h_c))

        # deltaK
        phi = (33 * h_b ** (3/2)) / (1 + 33 * h_b ** (3/2))

        outputs['CLag'] = CLag
        outputs['dalpha_zero'] = dalpha_zero
        outputs['phi'] = phi


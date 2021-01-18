import numpy as np
import openmdao.api as om
from scipy.constants import degree

from toa.data import Airplane


class DragCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all  data')
        self.options.declare('landing_gear', default=True,
                             desc='Accounts landing gear drag')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='flap_angle', shape=(1,), desc='Flap deflection',
                       units='rad')
        self.add_input(name='CL', shape=(nn,), desc='Lift coefficient', units=None)
        self.add_input(name='mass', shape=(nn,), desc=' mass', units='kg')
        self.add_input(name='grav', shape=(1,), desc='Gravity acceleration',
                       units='m/s**2')

        self.add_output(name='CD', shape=(nn,), desc='Drag coefficient', units=None)

        self.declare_partials(of='CD', wrt=['*'], method='fd')

    def compute(self, inputs, outputs, **kwargs):
        fa = inputs['flap_angle']
        CL = inputs['CL']
        mass = inputs['mass']
        grav = inputs['grav']

        ap = self.options['airplane']

        delta_cd_flap = (
                    ap.flap.lambda_f * ap.flap.cf_c ** 1.38 * ap.flap.sf_s * np.sin(
                fa) ** 2)

        if self.options['landing_gear']:
            delta_cd_gear = (mass * grav) / ap.wing.area * 3.16e-5 * ap.limits.MTOW ** (
                -0.215)
        else:
            delta_cd_gear = 0

        CD0_total = ap.polar.CD0 + delta_cd_gear + delta_cd_flap

        if ap.engine.mount == 'rear':
            delta_e_flap = 0.0046 * fa / degree
        else:
            delta_e_flap = 0.0026 * fa / degree

        ar = ap.wing.span ** 2 / ap.wing.area

        k_total = 1 / (1 / ap.polar.k + np.pi * ar * delta_e_flap)

        outputs['CD'] = CD0_total + k_total * CL ** 2

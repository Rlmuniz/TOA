import numpy as np
import openmdao.api as om
from scipy.constants import degree

from toa.data import Airplane


class DragCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all  data')
        self.options.declare('landing_gear', types=bool, default=True)
        self.options.declare('partial_coloring', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='flap_angle', val=0.0, desc='Flap deflection',
                       units='rad')
        self.add_input(name='CL', val=np.zeros(nn), desc='Lift coefficient', units=None)
        self.add_input(name='mass', val=np.zeros(nn), desc=' mass', units='kg')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration', units='m/s**2')
        self.add_input(name='phi', val=np.zeros(nn), desc='Induced drag variation due to ground effect', units=None)

        self.add_output(name='CD', shape=(nn,), desc='Drag coefficient', units=None)

        self.declare_partials(of='CD', wrt=['*'], method='fd')

        if self.options['partial_coloring']:
            self.declare_coloring(wrt=['*'], method='fd', tol=1.0E-6, num_full_jacs=2,
                                  show_summary=True, show_sparsity=True, min_improve_pct=10.)

    def compute(self, inputs, outputs, **kwargs):
        fa = inputs['flap_angle']
        CL = inputs['CL']
        mass = inputs['mass']
        grav = inputs['grav']
        phi = inputs['phi']

        ap = self.options['airplane']

        delta_cd_flap = (
                ap.flap.lambda_f * ap.flap.cf_c ** 1.38 * ap.flap.sf_s * np.sin(fa) ** 2)

        if self.options['landing_gear']:
            delta_cd_gear = (mass * grav) / ap.wing.area * 3.16e-5 * ap.limits.MTOW ** (-0.215)
        else:
            delta_cd_gear = 0

        CD0_total = ap.polar.CD0 + delta_cd_gear + delta_cd_flap

        if ap.engine.mount == 'rear':
            delta_e_flap = 0.0046 * fa / degree
        else:
            delta_e_flap = 0.0026 * fa / degree

        ar = ap.wing.span ** 2 / ap.wing.area

        k_total = 1 / (1 / ap.polar.k + np.pi * ar * delta_e_flap)

        outputs['CD'] = CD0_total + phi * k_total * CL ** 2

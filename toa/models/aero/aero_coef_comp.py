import numpy as np
import openmdao.api as om
from toa.airplanes import AirplaneData


class AeroCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='alpha', shape=(nn,), desc='Angle of attack',
                       units='rad')
        self.add_input(name='de', shape=(nn,), desc='Elevator angle', units='rad')
        self.add_input(name='tas', shape=(nn,), desc='True Airspeed', units='m/s')
        self.add_input(name='q', shape=(nn,), desc='Pitch Rate', units='rad/s')

        self.add_output(name='CL', val=np.zeros(nn), desc='Lift coefficient',
                        units=None)
        self.add_output(name='CD', val=np.zeros(nn), desc='Drag coefficient',
                        units=None)
        self.add_output(name='Cm', val=np.zeros(nn), desc='Moment coefficient',
                        units=None)

        # partials

    def setup_partials(self):
        self.declare_partials(of='CL', wrt='alpha', method='fd', form='central', step=1e-4)
        self.declare_partials(of='CL', wrt='de', method='fd', form='central', step=1e-4)
        self.declare_partials(of='CL', wrt='tas', method='fd', form='central', step=1e-4)
        self.declare_partials(of='CL', wrt='q', method='fd', form='central', step=1e-4)

        self.declare_partials(of='CD', wrt='alpha', method='fd', form='central', step=1e-4)
        self.declare_partials(of='CD', wrt='de', method='fd', form='central', step=1e-4)
        self.declare_partials(of='CD', wrt='tas', method='fd', form='central', step=1e-4)
        self.declare_partials(of='CD', wrt='q', method='fd', form='central', step=1e-4)

        self.declare_partials(of='Cm', wrt='alpha', method='fd', form='central', step=1e-4)
        self.declare_partials(of='Cm', wrt='de', method='fd', form='central', step=1e-4)
        self.declare_partials(of='Cm', wrt='tas', method='fd', form='central', step=1e-4)
        self.declare_partials(of='Cm', wrt='q', method='fd', form='central', step=1e-4)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        alpha = inputs['alpha']
        de = inputs['de']
        q = inputs['q']
        tas = inputs['tas']

        qhat = q * airplane.cbar / (2 * tas)

        CL = airplane.CL0 + airplane.CLa * alpha + airplane.CLde * de + airplane.CLq * qhat
        outputs['CL'] = CL
        outputs['CD'] = airplane.CDmin + airplane.kCDi * CL ** 2
        outputs['Cm'] = airplane.Cm0 + airplane.Cma * alpha + airplane.Cmde * de + airplane.Cmq * qhat


class AeroCoeffCompInitialRun(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        nz = np.zeros(nn)

        self.add_input(name='alpha', shape=(1,), desc='Angle of attack', units='rad')
        self.add_input(name='de', shape=(nn,), desc='Elevator angle', units='rad')

        self.add_output(name='CL', shape=(nn,), desc='Lift coefficient', units=None)
        self.add_output(name='CD', shape=(nn,), desc='Drag coefficient', units=None)
        self.add_output(name='Cm', shape=(nn,), desc='Moment coefficient', units=None)

    # partials
    def setup_partials(self):
        self.declare_partials(of='CL', wrt='alpha', method='fd', form='central', step=1e-4)
        self.declare_partials(of='CL', wrt='de', method='fd', form='central', step=1e-4)

        self.declare_partials(of='CD', wrt='alpha', method='fd', form='central', step=1e-4)
        self.declare_partials(of='CD', wrt='de', method='fd', form='central', step=1e-4)

        self.declare_partials(of='Cm', wrt='alpha', method='fd', form='central', step=1e-4)
        self.declare_partials(of='Cm', wrt='de', method='fd', form='central', step=1e-4)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        alpha = inputs['alpha']
        de = inputs['de']

        CL = airplane.CL0 + airplane.CLa * alpha + airplane.CLde * de
        outputs['CL'] = CL
        outputs['CD'] = airplane.CDmin + airplane.kCDi * CL ** 2
        outputs['Cm'] = airplane.Cm0 + airplane.Cma * alpha + airplane.Cmde * de
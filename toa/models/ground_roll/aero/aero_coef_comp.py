import numpy as np
import openmdao.api as om
from toa.airplanes import AirplaneData


class AeroCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='alpha', val=np.zeros(nn), desc='Angle of attack', units='rad')
        self.add_input(name='de', shape=(nn,), desc='Elevator angle', units='m/s')

        self.add_output(name='CL', shape=(nn,), desc='Lift coefficient', units=None)
        self.add_output(name='CD', shape=(nn,), desc='Drag coefficient', units=None)
        self.add_output(name='Cm', shape=(nn,), desc='Moment coefficient', units=None)

        # partials
        ar = np.arange(nn)
        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='de', rows=ar, cols=ar)

        self.declare_partials(of='CD', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='de', rows=ar, cols=ar)

        self.declare_partials(of='CM', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CM', wrt='de', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        alpha = inputs['alpha']
        de = inputs['de']

        CL = airplane.CL0 + airplane.CLa * alpha + airplane.CLde * de
        outputs['CL'] = CL
        outputs['CD'] = airplane.CDmin + airplane.kCDi * CL ** 2
        outputs['Cm'] = airplane.Cm0 + airplane.Cma * alpha + airplane.Cmde * de

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane_data']
        alpha = inputs['alpha']
        de = inputs['de']

        multi = 2 * airplane.kCDi * (airplane.CL0 + airplane.CLa * alpha + airplane.CLde * de)

        partials['CL', 'alpha'] = airplane.CLa
        partials['CL', 'de'] = airplane.CLde

        partials['CD', 'alpha'] = multi * partials['CL', 'alpha']
        partials['CD', 'de'] = multi * partials['CL', 'de']

        partials['Cm', 'alpha'] = airplane.Cma
        partials['Cm', 'de'] = airplane.Cmde

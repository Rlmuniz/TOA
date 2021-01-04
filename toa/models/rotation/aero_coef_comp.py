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
        self.add_input(name='de', val=np.zeros(nn), desc='Elevator angle', units='m/s')
        self.add_input(name='pitch_rate', val=np.zeros(nn), desc='Pitch rate', units='rad/s')
        self.add_input(name='tas', val=np.zeros(nn), desc='True airspeed', units='m/s')

        self.add_output(name='CL', val=np.zeros(nn), desc='Lift coefficient', units=None)
        self.add_output(name='CD', val=np.zeros(nn), desc='Drag coefficient', units=None)
        self.add_output(name='Cm', val=np.zeros(nn), desc='Moment coefficient', units=None)

        # partials
        ar = np.arange(nn)
        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='pitch_rate', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='tas', rows=ar, cols=ar)

        self.declare_partials(of='CD', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='pitch_rate', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='tas', rows=ar, cols=ar)

        self.declare_partials(of='CM', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CM', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='CM', wrt='pitch_rate', rows=ar, cols=ar)
        self.declare_partials(of='CM', wrt='tas', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        alpha = inputs['alpha']
        de = inputs['de']
        pitch_rate = inputs['pitch_rate']
        tas = inputs['tas']

        qhat = 0.5 * pitch_rate * airplane.cbar / tas

        CL = airplane.CL0 + airplane.CLa * alpha + airplane.CLde * de + airplane.CLq * qhat
        outputs['CL'] = CL
        outputs['CD'] = airplane.CDmin + airplane.kCDi * CL ** 2
        outputs['Cm'] = airplane.Cm0 + airplane.Cma * alpha + airplane.Cmde * de + airplane.Cmq * qhat

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane_data']
        alpha = inputs['alpha']
        de = inputs['de']
        pitch_rate = inputs['pitch_rate']
        tas = inputs['tas']

        qhat = 0.5 * pitch_rate * airplane.cbar / tas
        CL = airplane.CL0 + airplane.CLa * alpha + airplane.CLde * de + airplane.CLq * qhat
        multi = 2 * airplane.kCDi * CL

        partials['CL', 'alpha'] = airplane.CLa
        partials['CL', 'de'] = airplane.CLde
        partials['CL', 'pitch_rate'] = 0.5 * airplane.CLq * airplane.cbar / tas
        partials['CL', 'tas'] = -0.5 * airplane.CLq * airplane.cbar * pitch_rate / tas ** 2

        partials['CD', 'alpha'] = multi * partials['CL', 'alpha']
        partials['CD', 'de'] = multi * partials['CL', 'de']
        partials['CD', 'pitch_rate'] = multi * partials['CL', 'pitch_rate']
        partials['CD', 'tas'] = multi * partials['CL', 'tas']

        partials['Cm', 'alpha'] = airplane.Cma
        partials['Cm', 'de'] = airplane.Cmde
        partials['Cm', 'pitch_rate'] = 0.5 * airplane.Cmq * airplane.cbar / tas
        partials['Cm', 'tas'] = -0.5 * airplane.Cmq * airplane.cbar * pitch_rate / tas ** 2

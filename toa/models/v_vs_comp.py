import numpy as np
import openmdao.api as om

from toa.data import Airplane
from toa.data import get_airplane_data


class VVstallRatioComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)
        ones = np.ones(nn)

        self.add_input(name='CLmax', val=ones, desc='Max Lift Coefficient', units=None)
        self.add_input(name='mass', val=ones, desc='Airplane mass', units='kg')
        self.add_input(name='grav', val=9.80665, units='m/s**2', desc='Gravity acceleration')
        self.add_input(name='V', val=ones, desc='Body x axis velocity', units='m/s')
        self.add_input(name='rho', val=ones, desc='Atmosphere density', units='kg/m**3')

        self.add_output(name='Vstall', val=ones, desc='Stall speed', units='m/s')
        self.add_output(name='V_Vstall', val=ones, desc='Ratio between V and Vstall', units=None)

        self.declare_partials(of='Vstall', wrt='CLmax', rows=ar, cols=zz)
        self.declare_partials(of='Vstall', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='Vstall', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='Vstall', wrt='rho', rows=ar, cols=zz)

        self.declare_partials(of='V_Vstall', wrt='CLmax', rows=ar, cols=zz)
        self.declare_partials(of='V_Vstall', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='V_Vstall', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='V_Vstall', wrt='rho', rows=ar, cols=zz)
        self.declare_partials(of='V_Vstall', wrt='V', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane']
        CLmax = inputs['CLmax']
        mass = inputs['mass']
        grav = inputs['grav']
        rho = inputs['rho']

        outputs['Vstall'] = np.sqrt((2 * mass * grav) / (rho * airplane.wing.area * CLmax))
        outputs['V_Vstall'] = inputs['V'] / outputs['Vstall']

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane']
        S = airplane.wing.area
        CLmax = inputs['CLmax']
        mass = inputs['mass']
        grav = inputs['grav']
        rho = inputs['rho']
        V = inputs['V']

        partials['Vstall', 'CLmax'] = -np.sqrt(2) * np.sqrt(grav * mass / (CLmax * S * rho)) / (2 * CLmax)
        partials['Vstall', 'mass'] = np.sqrt(2) * np.sqrt(grav * mass / (CLmax * S * rho)) / (2 * mass)
        partials['Vstall', 'grav'] = np.sqrt(2) * np.sqrt(grav * mass / (CLmax * S * rho)) / (2 * grav)
        partials['Vstall', 'rho'] = -np.sqrt(2) * np.sqrt(grav * mass / (CLmax * S * rho)) / (2 * rho)

        partials['V_Vstall', 'CLmax'] = np.sqrt(2) * V / (4 * CLmax * np.sqrt(grav * mass / (CLmax * S * rho)))
        partials['V_Vstall', 'mass'] = -np.sqrt(2) * V / (4 * mass * np.sqrt(grav * mass / (CLmax * S * rho)))
        partials['V_Vstall', 'grav'] = -np.sqrt(2) * V / (4 * grav * np.sqrt(grav * mass / (CLmax * S * rho)))
        partials['V_Vstall', 'rho'] = np.sqrt(2) * V / (4 * rho * np.sqrt(grav * mass / (CLmax * S * rho)))
        partials['V_Vstall', 'V'] = np.sqrt(2) / (2 * np.sqrt(grav * mass / (CLmax * S * rho)))


if __name__ == '__main__':
    prob = om.Problem()
    airplane = get_airplane_data('b734')
    num_nodes = 1
    prob.model.add_subsystem('comp', VVstallRatioComp(num_nodes=1, airplane=airplane))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)
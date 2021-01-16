import numpy as np
import openmdao.api as om

from toa.data import Airplane


class RotationEOM(om.ExplicitComponent):
    """Models the rotation phase (2 DoF) in the takeoff run."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='thrust', val=np.zeros(nn), desc='Engine total thrust',
                       units='N')
        self.add_input(name='lift', val=np.zeros(nn), desc='Lift', units='N')
        self.add_input(name='drag', val=np.zeros(nn), desc='Drag force', units='N')
        self.add_input(name='moment', val=np.zeros(nn), desc='Aerodynamic moment',
                       units='N*m')
        self.add_input(name='V', val=np.zeros(nn), desc='Body x axis velocity',
                       units='m/s')
        self.add_input(name='mass', val=np.zeros(nn), desc='Airplane mass', units='kg')
        self.add_input(name='alpha', val=np.zeros(nn), desc='Angle of attack',
                       units='rad')
        self.add_input(name='q', val=np.zeros(nn), desc='Pitch rate', units='rad/s')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration',
                       units='m/s**2')

        self.add_output(name='x_dot', val=np.zeros(nn), desc='Derivative of position',
                        units='m/s')
        self.add_output(name='v_dot', val=np.zeros(nn), desc="Body x axis acceleration",
                        units='m/s**2')
        self.add_output(name='theta_dot', val=np.zeros(nn), desc="Alpha derivative",
                        units='rad/s')
        self.add_output(name='q_dot', val=np.zeros(nn), desc="Pitch rate derivative",
                        units='rad/s**2')
        self.add_output(name='f_mg', val=np.zeros(nn), desc='Main wheel reaction force',
                        units='N')

        self.declare_partials(of='v_dot', wrt=['*'], method='fd')
        self.declare_partials(of='x_dot', wrt='V', method='fd')
        self.declare_partials(of='theta_dot', wrt='q', method='fd')
        self.declare_partials(of='q_dot', wrt=['*'], method='fd')
        self.declare_partials(of='f_mg', wrt=['*'], method='fd')

    def compute(self, inputs, outputs, **kwargs):
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        moment = inputs['moment']
        V = inputs['V']
        mass = inputs['mass']
        alpha = inputs['alpha']
        q = inputs['q']
        grav = inputs['grav']
        rw_slope = inputs['rw_slope']
        airplane = self.options['airplane']

        mu = 0.002
        xmg = airplane.landing_gear.main.x
        weight = mass * grav
        cosslope = np.cos(rw_slope)
        sinslope = np.cos(rw_slope)
        cosalpha = np.cos(alpha)

        f_mg = weight * cosslope - lift
        f_rr = mu * f_mg
        m_mg = - xmg * f_mg

        outputs['v_dot'] = (thrust * cosalpha - drag - f_rr - weight * sinslope) / mass
        outputs['x_dot'] = V
        outputs['q_dot'] = (moment + m_mg) / airplane.inertia.iy
        outputs['theta_dot'] = q
        outputs['f_mg'] = f_mg

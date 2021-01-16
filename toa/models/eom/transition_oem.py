import numpy as np
import openmdao.api as om

from toa.data import Airplane


class TransitionOEM(om.ExplicitComponent):
    """Models the transition phase (3 DoF) from liftoff to screen height."""

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
        self.add_input(name='gam', val=np.zeros(nn), desc='Flight path angle',
                       units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration',
                       units='m/s**2')

        self.add_output(name='v_dot', val=np.zeros(nn), desc='Body x axis acceleration',
                        units='m/s**2')
        self.add_output(name='gam_dot', val=np.zeros(nn), desc='Flight path angle rate',
                        units='rad/s')
        self.add_output(name='x_dot', val=np.zeros(nn), desc='Derivative of position',
                        units='m/s')
        self.add_output(name='h_dot', val=np.zeros(nn), desc="Climb rate", units='m/s')
        self.add_output(name='q_dot', val=np.zeros(nn), desc="Pitch rate derivate",
                        units='rad/s**2')
        self.add_output(name='theta_dot', val=np.zeros(nn), desc="Pitch rate",
                        units='rad/s')

        self.declare_partials(of='v_dot', wrt=['*'], method='fd')
        self.declare_partials(of='gam_dot', wrt=['*'], method='fd')
        self.declare_partials(of='x_dot', wrt=['*'], method='fd')
        self.declare_partials(of='h_dot', wrt=['*'], method='fd')
        self.declare_partials(of='q_dot', wrt='moment', method='fd')
        self.declare_partials(of='theta_dot', wrt='q', method='fd')

    def compute(self, inputs, outputs, **kwargs):
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        moment = inputs['moment']
        V = inputs['V']
        mass = inputs['mass']
        alpha = inputs['alpha']
        q = inputs['q']
        gam = inputs['gam']
        grav = inputs['grav']
        airplane = self.options['airplane']

        weight = mass * grav

        cosgam = np.cos(gam)
        singam = np.sin(gam)
        cosalpha = np.cos(alpha)
        sinalpha = np.cos(alpha)

        outputs['v_dot'] = (thrust * cosalpha - drag - weight * singam) / mass
        outputs['gam_dot'] = (thrust * sinalpha + lift - weight * cosgam) / (mass * V)
        outputs['x_dot'] = outputs['v_dot'] * cosgam
        outputs['h_dot'] = outputs['v_dot'] * singam
        outputs['q_dot'] = moment / airplane.inertia.iy
        outputs['theta_dot'] = q

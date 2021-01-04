import numpy as np
import openmdao.api as om

from toa.airplanes import AirplaneData


class RotationEOM(om.ExplicitComponent):
    """Models the rotation phase (2 DoF) in the takeoff run."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='thrust', val=np.zeros(nn), desc='Engine total thrust', units='N')
        self.add_input(name='drag', val=np.zeros(nn), desc='Drag force', units='N')
        self.add_input(name='lift', val=np.zeros(nn), desc='Lift force', units='N')
        self.add_input(name='moment', val=np.zeros(nn), desc='Moment', units='N*m')
        self.add_input(name='mu', val=np.zeros(nn), desc='Friction coefficient', units=None)
        self.add_input(name='rw_slope', val=np.zeros(nn), desc='Runway slope', units='N')
        self.add_input(name='grav', val=np.zeros(nn), desc='Gravity acceleration', units='m/s**2')
        self.add_input(name='mass', val=np.zeros(nn), desc='Airplane mass', units='kg')
        self.add_input(name='alpha', val=np.zeros(nn), desc='Angle of attack', units='rad')

        self.add_output(name='v_dot', val=np.zeros(nn), desc="Body x axis acceleration", units='m/s**2')
        self.add_output(name='q_dot', val=np.zeros(nn), desc="Pitch rate derivative", units='rad/s**2')
        self.add_output(name='mlg_reaction', shape=(nn,), desc='Main landing gear reaction', units='N')

        ar = np.arange(nn)
        self.declare_partials(of='v_dot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='mu', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='rw_slope', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='grav', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='alpha', rows=ar, cols=ar)

        self.declare_partials(of='q_dot', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='q_dot', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='q_dot', wrt='rw_slope', rows=ar, cols=ar)
        self.declare_partials(of='q_dot', wrt='grav', rows=ar, cols=ar)
        self.declare_partials(of='q_dot', wrt='mass', rows=ar, cols=ar)

        self.declare_partials(of='mlg_reaction', wrt='lift', rows=ar, cols=ar, val=-1.0)
        self.declare_partials(of='mlg_reaction', wrt='rw_slope', rows=ar, cols=ar)
        self.declare_partials(of='mlg_reaction', wrt='grav', rows=ar, cols=ar)
        self.declare_partials(of='mlg_reaction', wrt='mass', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']

        thrust = inputs['thrust']
        drag = inputs['drag']
        lift = inputs['lift']
        moment = inputs['moment']
        mu = inputs['mu']
        mass = inputs['mass']
        weight = inputs['grav'] * mass

        calpha = np.cos(inputs['alpha'])
        sslope = np.sin(inputs['rw_slope'])
        cslope = np.cos(inputs['rw_slope'])

        mlg_reaction = (weight * cslope - lift)
        m_mlg = airplane.xm * mlg_reaction

        outputs['v_dot'] = (thrust * calpha - drag - mu * mlg_reaction - weight * sslope) / mass
        outputs['q_dot'] = (moment + m_mlg) / airplane.Iy
        outputs['mlg_reaction'] = mlg_reaction

    def compute_partials(self, inputs, partials, *kwargs):
        airplane = self.options['airplane_data']

        mass = inputs['mass']
        mu = inputs['mu']
        grav = inputs['grav']
        weight = grav * mass
        lift = inputs['lift']
        drag = inputs['drag']
        thrust = inputs['thrust']

        calpha = np.cos(inputs['alpha'])
        salpha = np.sin(inputs['alpha'])
        cslope = np.cos(inputs['rw_slope'])
        sslope = np.sin(inputs['rw_slope'])

        partials['v_dot', 'thrust'] = calpha / mass
        partials['v_dot', 'drag'] = - 1 / mass
        partials['v_dot', 'lift'] = mu * mass
        partials['v_dot', 'mu'] = (-weight * cslope + lift) / mass
        partials['v_dot', 'rw_slope'] = grav * (mu * sslope - cslope)
        partials['v_dot', 'grav'] = -mu * cslope - sslope
        partials['v_dot', 'mass'] = (drag - lift * mu - thrust * calpha) / mass ** 2
        partials['v_dot', 'alpha'] = -thrust * salpha / mass

        partials['q_dot', 'lift'] = -airplane.xm / airplane.Iy
        partials['q_dot', 'moment'] = 1 / airplane.Iy
        partials['q_dot', 'rw_slope'] = -weight * airplane.xm * sslope / airplane.Iy
        partials['q_dot', 'grav'] = mass * airplane.xm * cslope / airplane.Iy
        partials['q_dot', 'mass'] = grav * airplane.xm * cslope / airplane.Iy

        partials['mlg_reaction', 'rw_slope'] = -weight * sslope
        partials['mlg_reaction', 'grav'] = mass * cslope
        partials['mlg_reaction', 'mass'] = grav * cslope

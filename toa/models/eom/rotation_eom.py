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

        self.mu = 0.002

        self.add_input(name='thrust', val=np.zeros(nn), desc='Engine total thrust', units='N')
        self.add_input(name='drag', val=np.zeros(nn), desc='Drag force', units='N')
        self.add_input(name='lift', val=np.zeros(nn), desc='Lift force', units='N')
        self.add_input(name='moment', val=np.zeros(nn), desc='Moment', units='N*m')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration', units='m/s**2')
        self.add_input(name='mass', val=np.zeros(nn), desc='Airplane mass', units='kg')
        self.add_input(name='alpha', val=np.zeros(nn), desc='Angle of attack', units='rad')

        self.add_output(name='dXdt:v', val=np.zeros(nn), desc="Body x axis acceleration", units='m/s**2')
        self.add_output(name='dXdt:q', val=np.zeros(nn), desc="Pitch rate derivative", units='rad/s**2')
        self.add_output(name='mlg_reaction', shape=(nn,), desc='Main landing gear reaction', units='N')

        ar = np.arange(nn)
        self.declare_partials(of='dXdt:v', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='rw_slope', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='dXdt:v', wrt='grav', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='dXdt:v', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='alpha', rows=ar, cols=ar)

        self.declare_partials(of='dXdt:q', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='rw_slope', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='dXdt:q', wrt='grav', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='dXdt:q', wrt='mass', rows=ar, cols=ar)

        self.declare_partials(of='mlg_reaction', wrt='lift', rows=ar, cols=ar, val=-1.0)
        self.declare_partials(of='mlg_reaction', wrt='rw_slope', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='mlg_reaction', wrt='grav', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='mlg_reaction', wrt='mass', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']

        thrust = inputs['thrust']
        drag = inputs['drag']
        lift = inputs['lift']
        moment = inputs['moment']
        mu = self.mu
        mass = inputs['mass']
        weight = inputs['grav'] * mass

        calpha = np.cos(inputs['alpha'])
        sslope = np.sin(inputs['rw_slope'])
        cslope = np.cos(inputs['rw_slope'])

        mlg_reaction = (weight * cslope - lift)
        m_mlg = airplane.xm * mlg_reaction

        outputs['dXdt:v'] = (thrust * calpha - drag - mu * mlg_reaction - weight * sslope) / mass
        outputs['dXdt:q'] = (moment + m_mlg) / airplane.Iy
        outputs['mlg_reaction'] = mlg_reaction

    def compute_partials(self, inputs, partials, *kwargs):
        airplane = self.options['airplane_data']

        mass = inputs['mass']
        mu = self.mu
        grav = inputs['grav']
        weight = grav * mass
        lift = inputs['lift']
        drag = inputs['drag']
        thrust = inputs['thrust']

        calpha = np.cos(inputs['alpha'])
        salpha = np.sin(inputs['alpha'])
        cslope = np.cos(inputs['rw_slope'])
        sslope = np.sin(inputs['rw_slope'])

        partials['dXdt:v', 'thrust'] = calpha / mass
        partials['dXdt:v', 'drag'] = - 1 / mass
        partials['dXdt:v', 'lift'] = mu * mass
        partials['dXdt:v', 'rw_slope'] = grav * (mu * sslope - cslope)
        partials['dXdt:v', 'grav'] = -mu * cslope - sslope
        partials['dXdt:v', 'mass'] = (drag - lift * mu - thrust * calpha) / mass ** 2
        partials['dXdt:v', 'alpha'] = -thrust * salpha / mass

        partials['dXdt:q', 'lift'] = -airplane.xm / airplane.Iy
        partials['dXdt:q', 'moment'] = 1 / airplane.Iy
        partials['dXdt:q', 'rw_slope'] = -weight * airplane.xm * sslope / airplane.Iy
        partials['dXdt:q', 'grav'] = mass * airplane.xm * cslope / airplane.Iy
        partials['dXdt:q', 'mass'] = grav * airplane.xm * cslope / airplane.Iy

        partials['mlg_reaction', 'rw_slope'] = -weight * sslope
        partials['mlg_reaction', 'grav'] = mass * cslope
        partials['mlg_reaction', 'mass'] = grav * cslope

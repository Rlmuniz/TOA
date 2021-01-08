import numpy as np
import openmdao.api as om

from toa.airplanes import AirplaneData


class AllWheelsOnGroundReactionForces(om.ExplicitComponent):
    """Compute the reaction forces on main and nose landing gear."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.mu = 0.002

        # Inputs
        self.add_input(name='lift', shape=(nn,), desc='Lift', units='N')
        self.add_input(name='moment', shape=(nn,), desc='Moment', units='N*m')
        self.add_input(name='thrust', shape=(nn,), desc='Thrust', units='N')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='mass', shape=(nn,), desc='Airplane mass', units='kg')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration', units='m/s**2')

        # Outputs
        self.add_output(name='mlg_reaction', shape=(nn,), desc='Main landing gear reaction', units='N')
        self.add_output(name='nlg_reaction', shape=(nn,), desc='Nose landing gear reaction', units='N')

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='mlg_reaction', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='mlg_reaction', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='mlg_reaction', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='mlg_reaction', wrt='rw_slope', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='mlg_reaction', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='mlg_reaction', wrt='grav', rows=ar, cols=np.zeros(nn))

        self.declare_partials(of='nlg_reaction', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='nlg_reaction', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='nlg_reaction', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='nlg_reaction', wrt='rw_slope', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='nlg_reaction', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='nlg_reaction', wrt='grav', rows=ar, cols=np.zeros(nn))

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        lift = inputs['lift']
        moment = inputs['moment']
        thrust = inputs['thrust']
        rw_slope = inputs['rw_slope']
        mass = inputs['mass']
        grav = inputs['grav']
        mu = self.mu

        weight = grav * mass
        cs = np.cos(rw_slope)

        outputs['mlg_reaction'] = (-moment - thrust * airplane.zt + (airplane.xm - airplane.xn) * (
                    weight * cs - lift) + (mu * airplane.xm - airplane.xm) * (weight * cs - lift)) / (
                                              airplane.xm - airplane.xn)
        outputs['nlg_reaction'] = (moment + thrust * airplane.zt + (-mu * airplane.zmn + airplane.xm) * (
                    weight * cs - lift)) / (airplane.xm - airplane.xn)

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane_data']
        rw_slope = inputs['rw_slope']
        mass = inputs['mass']
        grav = inputs['grav']
        mu = self.mu

        weight = grav * mass
        ss = np.sin(rw_slope)
        cs = np.sin(rw_slope)
        dx = (airplane.xm - airplane.xn)

        partials['mlg_reaction', 'lift'] = (-mu * airplane.zmn + airplane.xn) / dx
        partials['mlg_reaction', 'moment'] = -1 / dx
        partials['mlg_reaction', 'thrust'] = -airplane.zt / dx
        partials['mlg_reaction', 'rw_slope'] = -weight * (mu * airplane.zmn - airplane.xn) * ss / dx
        partials['mlg_reaction', 'mass'] = grav * (mu * airplane.zmn - airplane.xn) * cs / dx

        partials['nlg_reaction', 'lift'] = (mu * airplane.zmn - airplane.xm) / dx
        partials['nlg_reaction', 'moment'] = 1 / dx
        partials['nlg_reaction', 'thrust'] = airplane.zt / dx
        partials['nlg_reaction', 'rw_slope'] = grav * mass * (mu * airplane.zmn - airplane.xm) * ss / dx
        partials['nlg_reaction', 'mass'] = grav * (-mu * airplane.zmn + airplane.xm) * cs / dx

import openmdao.api as om
import dymos as dm
from scipy.constants import degree
import matplotlib.pyplot as plt
from toa.airplanes import b747

from toa.ode.ground_roll import GroundRollODE
from toa.ode.rotation import RotationODE

p = om.Problem(model=om.Group())
traj = p.model.add_subsystem('traj', dm.Trajectory())

transcription0 = dm.GaussLobatto(num_segments=20, order=3)
ground_roll = dm.Phase(ode_class=GroundRollODE, transcription=transcription0,
                       ode_init_kwargs={'airplane_data': b747})
traj.add_phase(name='ground_roll', phase=ground_roll)

# Set time options
ground_roll.set_time_options(fix_initial=True, units='s')

# Configure states
ground_roll.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                      rate_source='dXdt:v', lower=1, targets=['v'])
ground_roll.add_state('x', fix_initial=True, fix_final=False, units='m',
                      rate_source='v', lower=0, upper=5000)
ground_roll.add_state('mass', fix_initial=False, fix_final=False, units='kg',
                      rate_source='dXdt:mass_fuel', upper=330000)

# Configure controls
ground_roll.add_control(name='de', units='rad', lower=-0.5, upper=0.5, targets=['de'])

# Configure path constraints
ground_roll.add_path_constraint('nlg_reaction', lower=0, units='N')
ground_roll.add_path_constraint('mlg_reaction', lower=0, units='N')
ground_roll.add_boundary_constraint('nlg_reaction', loc='final',
                                    constraint_name='final_F_ALG', lower=0.01,
                                    units='N', shape=(1,))

# Rotation
transcription1 = dm.GaussLobatto(num_segments=10, order=3)
rotation = dm.Phase(ode_class=RotationODE, transcription=transcription1,
                    ode_init_kwargs={'airplane_data': b747})
traj.add_phase(name='rotation', phase=rotation)

# Set time options
rotation.set_time_options(units='s')

# Configure states
rotation.add_state('v', fix_initial=False, fix_final=False, units='m/s',
                   rate_source='dXdt:v', lower=0, targets=['v'])
rotation.add_state('x', fix_initial=False, fix_final=False, units='m', rate_source='v',
                   upper=5000)
rotation.add_state('q', fix_initial=True, fix_final=False, units='rad/s',
                   rate_source='dXdt:q', targets=['q'])
rotation.add_state('alpha', fix_initial=True, fix_final=False, units='rad',
                   rate_source='q', targets=['alpha'], lower=0, upper=30 * degree)
rotation.add_state('mass', fix_initial=False, fix_final=False, units='kg',
                   rate_source='dXdt:mass_fuel', upper=330000)

# Configure controls
rotation.add_control(name='de', units='rad', lower=-0.5, upper=0.5, targets=['de'])

rotation.add_path_constraint('mlg_reaction', lower=0, units='N')
rotation.add_boundary_constraint('mlg_reaction', loc='final',
                                 constraint_name='final_F_MLG', lower=0.01, units='N',
                                 shape=(1,))

traj.link_phases(phases=['ground_roll', 'rotation'], vars=['time', 'x', 'v', 'mass'])

# Set objective: maximize mass
ground_roll.add_objective('mass', scaler=-1)

# Set the driver.
p.driver = om.ScipyOptimizeDriver()
p.driver.options['disp'] = True

# Setup the problem
p.setup(check=True)

p.set_val('traj.ground_roll.states:x',
          ground_roll.interpolate(ys=[0, 4000], nodes='state_input'),
          units='m')
p.set_val('traj.ground_roll.states:v',
          ground_roll.interpolate(ys=[0, 100], nodes='state_input'),
          units='m/s')
p.set_val('traj.ground_roll.states:mass',
          ground_roll.interpolate(ys=[333000, 330000], nodes='state_input'),
          units='kg')
p.set_val('traj.rotation.states:x',
          rotation.interpolate(ys=[4000, 5000], nodes='state_input'),
          units='m')
p.set_val('traj.rotation.states:v',
          rotation.interpolate(ys=[100, 130], nodes='state_input'),
          units='m/s')
p.set_val('traj.rotation.states:alpha',
          rotation.interpolate(ys=[0, 20], nodes='state_input'),
          units='deg')
p.set_val('traj.rotation.states:q',
          rotation.interpolate(ys=[0, 30], nodes='state_input'),
          units='deg/s')
p.set_val('traj.rotation.states:mass',
          rotation.interpolate(ys=[333000, 330000], nodes='state_input'), units='kg')

p.set_val('traj.ground_roll.t_duration', 60.0)
p.set_val('traj.rotation.t_initial', 60.0)
p.set_val('traj.rotation.t_duration', 10.0)

p.run_driver()

plt.figure(0)
plt.clf();
plt.plot(p.get_val('traj.ground_roll.timeseries.time'),
         p.get_val('traj.ground_roll.timeseries.states:x'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.states:x'), '.-')
plt.title('position')
plt.show()

plt.figure(1)
plt.clf();
plt.plot(p.get_val('traj.ground_roll.timeseries.time'),
         p.get_val('traj.ground_roll.timeseries.F_MLG'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.F_MLG'), '.-',
         p.get_val('traj.ground_roll.timeseries.time'),
         p.get_val('traj.ground_roll.timeseries.F_ALG'), '.-')
plt.title('Landing gear forces')
plt.grid()
plt.show()
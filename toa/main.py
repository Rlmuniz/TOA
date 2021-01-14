import openmdao.api as om
import dymos as dm
from dymos.models.atmosphere import USatm1976Comp
import matplotlib.pyplot as plt

from toa.data.airplanes.examples import b747
from toa.ode.ground_roll import GroundRollODE
from toa.ode.rotation import RotationODE
from scipy.constants import degree

p = om.Problem(model=om.Group())

# set driver
p.driver = om.ScipyOptimizeDriver()
p.driver.options['disp'] = True

external_params = p.model.add_subsystem('external_params', subsys=om.IndepVarComp())
external_params.add_output('elevation', val=0.0, units='m', desc='Runway elevation')
external_params.add_output('rw_slope', val=0.0, units='rad', desc='Runway slope')

traj = p.model.add_subsystem('traj', dm.Trajectory())
transcription0 = dm.GaussLobatto(num_segments=20, order=3)
initial_run = dm.Phase(ode_class=GroundRollODE, transcription=transcription0,
                       ode_init_kwargs={'airplane_data': b747})
traj.add_phase(name='initial_run', phase=initial_run)

# Set time options
initial_run.set_time_options(fix_initial=True, units='s')

# Configure states
initial_run.add_state('V', fix_initial=True, fix_final=False, units='m/s',
                      rate_source='ground_run_eom.dXdt:v', lower=0, targets=['tas_comp.V', 'ground_run_eom.V'])
initial_run.add_state('x', fix_initial=True, fix_final=False, units='m',
                      rate_source='ground_run_eom.dXdt:x', lower=0, upper=3000)
initial_run.add_state('mass', fix_initial=False, fix_final=False, units='kg',
                      rate_source='prop.dXdt:mass_fuel', lower=0, upper=396800, targets=['ground_run_eom.mass', 'aero.mass'])

# Configure controls
initial_run.add_control(name='de', units='rad', lower=-30.0*degree, upper=30.0*degree, targets=['de'])

# Configure path constraints
initial_run.add_path_constraint('ground_run_eom.rf_mainwheel', lower=0, units='N')
initial_run.add_path_constraint('ground_run_eom.rf_nosewheel', lower=0, units='N')
initial_run.add_boundary_constraint('ground_run_eom.rf_nosewheel', loc='final', constraint_name='final_F_ALG',
                                    equals=0.01, units='N', shape=(1,))

initial_run.add_objective('mass', loc='initial', scaler=-1)

initial_run.add_timeseries_output('thrust')
initial_run.add_timeseries_output('L')
initial_run.add_timeseries_output('D')

# Rotation
transcription1 = dm.GaussLobatto(num_segments=10, order=3)
rotation = dm.Phase(ode_class=RotationODE, transcription=transcription1,
                    ode_init_kwargs={'airplane_data': b747})
traj.add_phase(name='rotation', phase=rotation)

# Set time options
rotation.set_time_options(units='s')

# Configure states
rotation.add_state('V', fix_initial=False, fix_final=False, units='m/s',
                   rate_source='rotation_eom.dXdt:v', lower=0, targets=['tas_comp.V', 'rotation_eom.V'])
rotation.add_state('x', fix_initial=False, fix_final=False, units='m', rate_source='rotation_eom.dXdt:x', upper=3000)
rotation.add_state('q', fix_initial=True, fix_final=False, units='rad/s', rate_source='rotation_eom.dXdt:q',
                   targets=['rotation_eom.q', 'aero.q'])
rotation.add_state('alpha', fix_initial=True, fix_final=False, units='rad', rate_source='rotation_eom.dXdt:alpha',
                   targets=['alpha'], lower=0, upper=30*degree)
rotation.add_state('mass', fix_initial=False, fix_final=False, units='kg', rate_source='prop.dXdt:mass_fuel', targets=['aero.mass', 'rotation_eom.mass'],
                   upper=396800)

# Configure controls
rotation.add_control(name='de', units='rad', lower=-30.0*degree, upper=30.0*degree, targets=['de'])

rotation.add_path_constraint('rotation_eom.rf_mainwheel', lower=0, units='N')
rotation.add_boundary_constraint('rotation_eom.rf_mainwheel', loc='final', constraint_name='final_F_MLG', equals=0.01,
                                 units='N', shape=(1,))

rotation.add_timeseries_output('thrust')
rotation.add_timeseries_output('L')
rotation.add_timeseries_output('D')
# Connect external parameters
traj.add_parameter('flap_angle', targets={'ground_roll': ['aero.flap_angle'], 'rotation': ['aero.flap_angle']},
                   shape=(1,), desc='Flap Angle', units='deg', opt=False, dynamic=False)
traj.add_parameter('elevation',
                   targets={'ground_roll': ['h', 'prop.fuel_flow.elevation'], 'rotation': ['h', 'prop.fuel_flow.elevation']},
                   shape=(1,), units='m', opt=False, dynamic=False)
traj.add_parameter('rw_slope',
                   targets={'ground_roll': ['ground_run_eom.rw_slope'], 'rotation': ['rotation_eom.rw_slope']},
                   shape=(1,), units='rad', opt=False, dynamic=False)
traj.add_parameter('Vw', targets={'ground_roll': ['tas_comp.Vw'], 'rotation': ['tas_comp.Vw']}, shape=(1,),
                   desc='Wind speed along the runway, defined as positive for a headwind', units='m/s', opt=False,
                   dynamic=False)

traj.link_phases(phases=['initial_run', 'rotation'], vars=['time', 'x', 'V', 'mass'])

p.model.connect('external_params.elevation', 'traj.parameters:elevation')
p.model.connect('external_params.rw_slope', 'traj.parameters:rw_slope')

# Setup the problem
p.setup(check=True)

p.set_val('traj.parameters:flap_angle', 0.0)
p.set_val('traj.parameters:Vw', 0.0)

# Initial guesses
p.set_val('traj.initial_run.states:x',
          initial_run.interpolate(ys=[0, 2400], nodes='state_input'),
          units='m')
p.set_val('traj.initial_run.states:V',
          initial_run.interpolate(ys=[0, 100], nodes='state_input'),
          units='m/s')
p.set_val('traj.initial_run.states:mass',
          initial_run.interpolate(ys=[396800, 396400], nodes='state_input'),
          units='kg')

p.set_val('traj.rotation.states:x',
          rotation.interpolate(ys=[2400, 3000], nodes='state_input'),
          units='m')
p.set_val('traj.rotation.states:V',
          rotation.interpolate(ys=[100, 120], nodes='state_input'),
          units='m/s')
p.set_val('traj.rotation.states:alpha',
          rotation.interpolate(ys=[0, 20], nodes='state_input'),
          units='rad')
p.set_val('traj.rotation.states:q',
          rotation.interpolate(ys=[0, 30], nodes='state_input'),
          units='deg/s')
p.set_val('traj.rotation.states:mass',
          rotation.interpolate(ys=[396400, 396000], nodes='state_input'),
          units='kg')

p.set_val('traj.initial_run.t_duration', 60.0)
p.set_val('traj.rotation.t_initial', 60.0)
p.set_val('traj.rotation.t_duration', 10.0)

dm.run_problem(p)

plt.figure(0)
plt.plot(p.get_val('traj.initial_run.timeseries.time'),
         p.get_val('traj.initial_run.timeseries.states:x'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.states:x'), '.-')
plt.title('position')
plt.grid()

plt.figure(1)
plt.plot(p.get_val('traj.initial_run.timeseries.time'),
         p.get_val('traj.initial_run.timeseries.rf_mainwheel', units='kN'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.rf_mainwheel', units='kN'), '.-',
         p.get_val('traj.initial_run.timeseries.time'),
         p.get_val('traj.initial_run.timeseries.rf_nosewheel', units='kN'), '.-')
plt.title('Landing gear forces')
plt.grid()


plt.figure(2)
plt.plot(p.get_val('traj.initial_run.timeseries.time'),
         p.get_val('traj.initial_run.timeseries.controls:de', units='deg'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.controls:de', units='deg'), '.-')
plt.title('Elevator deflection')
plt.grid()

plt.figure(3)
plt.plot(p.get_val('traj.initial_run.timeseries.time'),
         p.get_val('traj.initial_run.timeseries.states:V', units='kn'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.states:V', units='kn'), '.-')
plt.title('V')
plt.grid()

plt.figure(4)
plt.plot(p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.states:alpha', units='deg'), '.-')
plt.title('Angle of Attack')
plt.grid()

plt.figure(5)
plt.plot(p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.states:q', units='deg/s'), '.-')
plt.title('Pitch rate')
plt.grid()

plt.figure(6)
plt.plot(p.get_val('traj.initial_run.timeseries.time'),
         p.get_val('traj.initial_run.timeseries.thrust', units='kN'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.thrust', units='kN'), '.-')
plt.title('Thrust')
plt.grid()

plt.figure(7)
plt.plot(p.get_val('traj.initial_run.timeseries.time'),
         p.get_val('traj.initial_run.timeseries.D', units='kN'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.D', units='kN'), '.-',
         p.get_val('traj.initial_run.timeseries.time'),
         p.get_val('traj.initial_run.timeseries.L', units='kN'),
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.L', units='kN'))
plt.title('Forces')
plt.grid()

plt.figure(8)
plt.plot(p.get_val('traj.initial_run.timeseries.time'),
         p.get_val('traj.initial_run.timeseries.states:mass', units='kg'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.states:mass', units='kg'), '.-')
plt.title('mass')
plt.grid()
plt.show()

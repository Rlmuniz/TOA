import openmdao.api as om
import dymos as dm
from dymos.models.atmosphere import USatm1976Comp
import matplotlib.pyplot as plt
from toa.data import b747

from toa.ode.ground_roll import GroundRollODE
from toa.ode.rotation import RotationODE

p = om.Problem(model=om.Group())

# set driver
p.driver = om.ScipyOptimizeDriver()
p.driver.options['disp'] = True

external_params = p.model.add_subsystem('external_params', subsys=om.IndepVarComp())
external_params.add_output('elevation', val=0.0, units='m', desc='Runway elevation')
external_params.add_output('rw_slope', val=0.0, units='rad', desc='Runway slope')

p.model.add_subsystem('atmo', subsys=USatm1976Comp(num_nodes=1))

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
                      rate_source='prop.dXdt:mass_fuel', upper=396800, targets=['ground_run_eom.mass'])

# Configure controls
initial_run.add_control(name='de', units='deg', lower=-30.0, upper=30.0, targets=['de'])

# Configure path constraints
initial_run.add_path_constraint('ground_run_eom.rf_mainwheel', lower=0, units='N')
initial_run.add_path_constraint('ground_run_eom.rf_nosewheel', lower=0, units='N')
initial_run.add_boundary_constraint('ground_run_eom.rf_nosewheel', loc='final', constraint_name='final_F_ALG',
                                    upper=0.01, units='N', shape=(1,))

# Configure phase parameters
initial_run.add_parameter('alpha', targets=['alpha'], units='deg', opt=True, lower=-5.0, upper=0.0, dynamic=False)

initial_run.add_objective('mass', loc='initial', scaler=-1)

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
rotation.add_state('q', fix_initial=True, fix_final=False, lower=0, units='deg/s', rate_source='rotation_eom.dXdt:q',
                   targets=['rotation_eom.q', 'aero.q'])
rotation.add_state('alpha', fix_initial=False, fix_final=False, units='deg', rate_source='rotation_eom.dXdt:alpha',
                   targets=['alpha'], lower=-10, upper=20)
rotation.add_state('mass', fix_initial=False, fix_final=False, units='kg', rate_source='prop.dXdt:mass_fuel',
                   upper=396800)

# Configure controls
rotation.add_control(name='de', units='deg', lower=-30.0, upper=30.0, targets=['de'])

rotation.add_path_constraint('rotation_eom.rf_mainwheel', lower=0, units='N')
rotation.add_boundary_constraint('rotation_eom.rf_mainwheel', loc='final', constraint_name='final_F_MLG', upper=0.01,
                                 units='N', shape=(1,))

# Connect external parameters
traj.add_parameter('grav', targets={'ground_roll': ['ground_run_eom.grav'], 'rotation': ['rotation_eom.grav']},
                   shape=(1,), desc='Gravity acceleration', units='m/s**2', opt=False, dynamic=False)
traj.add_parameter('elevation',
                   targets={'ground_roll': ['prop.fuel_flow.elevation'], 'rotation': ['prop.fuel_flow.elevation']},
                   shape=(1,), units='m', opt=False, dynamic=False)
traj.add_parameter('rw_slope',
                   targets={'ground_roll': ['ground_run_eom.rw_slope'], 'rotation': ['rotation_eom.rw_slope']},
                   shape=(1,), units='rad', opt=False, dynamic=False)
traj.add_parameter('Vw', targets={'ground_roll': ['tas_comp.Vw'], 'rotation': ['tas_comp.Vw']}, shape=(1,),
                   desc='Wind speed along the runway, defined as positive for a headwind', units='m/s', opt=False,
                   dynamic=False)
traj.add_parameter('rho', targets={'ground_roll': ['aero.rho'], 'rotation': ['aero.rho']}, shape=(1,),
                   desc='Atmosphere density', units='kg/m**3', opt=False, dynamic=False)
traj.add_parameter('sos', targets={'ground_roll': ['prop.sos'], 'rotation': ['prop.sos']}, shape=(1,),
                   desc='Speed of sound', units='m/s', opt=False, dynamic=False)
traj.add_parameter('p_amb', targets={'ground_roll': ['prop.thrust_comp.p_amb'], 'rotation': ['prop.thrust_comp.p_amb']},
                   shape=(1,), desc='Ambient pressure', units='Pa', opt=False, dynamic=False)

traj.link_phases(phases=['initial_run', 'rotation'], vars=['time', 'x', 'V', 'mass', 'alpha'])

p.model.connect('external_params.elevation', 'traj.parameters:elevation')
p.model.connect('external_params.rw_slope', 'traj.parameters:rw_slope')

#p.model.connect('atmo.rho', 'traj.parameters:rho')
#p.model.connect('atmo.sos', 'traj.parameters:sos')
#p.model.connect('atmo.pres', 'traj.parameters:p_amb')

# Setup the problem
p.setup(check=True)

p.set_val('traj.parameters:grav', 9.80665)
p.set_val('traj.parameters:rho', 1.225)
p.set_val('traj.parameters:sos', 340)
p.set_val('traj.parameters:p_amb', 101325.0)
p.set_val('traj.parameters:Vw', 0.0)

# Initial guesses
p.set_val('traj.initial_run.states:x',
          initial_run.interpolate(ys=[0, 2400], nodes='state_input'),
          units='m')
p.set_val('traj.initial_run.states:V',
          initial_run.interpolate(ys=[0, 100], nodes='state_input'),
          units='m/s')
p['traj.initial_run.states:mass'][:] = 396800
p['traj.initial_run.parameters:alpha'] = 0

p.set_val('traj.rotation.states:x',
          rotation.interpolate(ys=[2400, 3000], nodes='state_input'),
          units='m')
p.set_val('traj.rotation.states:V',
          rotation.interpolate(ys=[100, 130], nodes='state_input'),
          units='m/s')
p.set_val('traj.rotation.states:alpha',
          rotation.interpolate(ys=[0, 15], nodes='state_input'),
          units='deg')
p.set_val('traj.rotation.states:q',
          rotation.interpolate(ys=[0, 30], nodes='state_input'),
          units='deg/s')
p['traj.rotation.states:mass'][:] = 396800

p.set_val('traj.initial_run.t_duration', 60.0)
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
         p.get_val('traj.ground_roll.timeseries.mlg_reaction'), '.-',
         p.get_val('traj.rotation.timeseries.time'),
         p.get_val('traj.rotation.timeseries.mlg_reaction'), '.-',
         p.get_val('traj.ground_roll.timeseries.time'),
         p.get_val('traj.ground_roll.timeseries.nlg_reaction'), '.-')
plt.title('Landing gear forces')
plt.grid()
plt.show()

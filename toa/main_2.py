import openmdao.api as om
import dymos as dm

from dymos.examples.plotting import plot_results

from toa.data import get_airplane_data
from toa.ode.initialrun_ode import InitialRunODE
from toa.ode.rotation_ode import RotationODE

from toa.runway import Runway

p = om.Problem(model=om.Group())

# set driver
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'

runway = Runway(3500, 0.0, 0.0, 0.0, 0.0)
airplane = get_airplane_data('b734')

traj = p.model.add_subsystem('traj', dm.Trajectory())
initialrun = dm.Phase(ode_class=InitialRunODE,
                      transcription=dm.Radau(num_segments=20),
                      ode_init_kwargs={'airplane': airplane})
traj.add_phase(name='initialrun', phase=initialrun)

initialrun.set_time_options(fix_initial=True, units='s')

# Initial run states
initialrun.add_state(name='V', units='kn', rate_source='initial_run_eom.v_dot',
                     targets=['V'], fix_initial=True, fix_final=False, lower=0)
initialrun.add_state(name='x', units='m', rate_source='initial_run_eom.x_dot',
                     targets=['mlg_pos.x'], fix_initial=True, fix_final=False,
                     lower=airplane.landing_gear.main.x)
initialrun.add_state(name='mass', units='kg', rate_source='prop.m_dot',
                     targets=['mass'], fix_initial=False, fix_final=False, lower=0.0,
                     upper=airplane.limits.MTOW)

# Initial run parameters
initialrun.add_parameter(name='de', val=0.0, units='deg', desc='Elevator deflection',
                         lower=-30.0, upper=30.0, targets=['aero.de'], opt=True)
initialrun.add_parameter(name='theta', val=0.0, units='deg', desc='Pitch Angle',
                         targets=['aero.alpha', 'initial_run_eom.alpha',
                                  'mlg_pos.theta'], opt=False)
initialrun.add_parameter(name='h', val=airplane.landing_gear.main.z, units='m',
                         desc='Pitch Angle',
                         targets=['aero.alpha', 'initial_run_eom.alpha',
                                  'mlg_pos.theta'], opt=False)

# Initial run boundary constraint
initialrun.add_boundary_constraint(name='initial_run_eom.f_ng', loc='final', units='N',
                                   lower=0.0, upper=0.5)

initialrun.add_objective('mass', loc='initial', scaler=-1)

initialrun.add_timeseries_output('aero.CD')
initialrun.add_timeseries_output('aero.CL')
initialrun.add_timeseries_output('aero.Cm')
initialrun.add_timeseries_output('initial_run_eom.f_mg', units='kN')
initialrun.add_timeseries_output('initial_run_eom.f_ng', units='kN')
initialrun.add_timeseries_output('prop.thrust', units='kN')
initialrun.add_timeseries_output('prop.m_dot', units='kg/s')

## Rotation
rotation = dm.Phase(ode_class=RotationODE,
                    transcription=dm.Radau(num_segments=5),
                    ode_init_kwargs={'airplane': airplane})
traj.add_phase(name='rotation', phase=rotation)

rotation.set_time_options(fix_initial=False, units='s')

# Rotation states
rotation.add_state(name='V', units='kn', rate_source='rotation_eom.v_dot',
                   targets=['V'], fix_initial=False, fix_final=False, lower=0)
rotation.add_state(name='x', units='m', rate_source='rotation_eom.x_dot',
                   targets=['mlg_pos.x'], fix_initial=False, fix_final=False, lower=0)
rotation.add_state(name='h', units='m', rate_source='rotation_eom.h_dot',
                   targets=['mlg_pos.h'], fix_initial=True, fix_final=False)
rotation.add_state(name='mass', units='kg', rate_source='prop.m_dot', targets=['mass'],
                   fix_initial=False, fix_final=False, lower=0)
rotation.add_state(name='theta', units='deg', rate_source='rotation_eom.theta_dot',
                   targets=['alpha'],
                   fix_initial=True, fix_final=False, lower=0)
rotation.add_state(name='q', units='deg/s', rate_source='rotation_eom.q_dot',
                   targets=['q'], fix_initial=True, fix_final=False)

# Rotation controls
rotation.add_control(name='de', units='deg', desc='Elevator defletion', opt=True,
                     fix_initial=False, fix_final=False, targets=['aero.de'],
                     lower=-30.0, upper=30.0)

# Rotation path constraints
rotation.add_path_constraint(name='rotation_eom.f_mg', lower=0, units='N')
rotation.add_path_constraint(name='clmax_cl.diff', units=None, lower=0.0)
rotation.add_path_constraint(name='alphamax_alpha.diff', units='deg', lower=0.0)

# Rotation boundary constraint
rotation.add_boundary_constraint(name='mlg_pos.x_mlg', loc='final', units='m', upper=runway.tora)
rotation.add_boundary_constraint(name='rotation_eom.f_mg', loc='final', units='N',
                                 lower=0.0, upper=0.5)

rotation.add_timeseries_output('aero.CD')
rotation.add_timeseries_output('aero.CL')
rotation.add_timeseries_output('aero.Cm')
rotation.add_timeseries_output('rotation_eom.f_mg', units='kN')
rotation.add_timeseries_output('prop.thrust', units='kN')
rotation.add_timeseries_output('prop.m_dot', units='kg/s')

# Trajectory parameters
traj.add_parameter(name='dih', val=0.0, units='deg', lower=-10.0, upper=10.0,
                   desc='Horizontal stabilizer angle',
                   targets={
                       'initialrun': ['aero.dih'], 'rotation': ['aero.dih'],
                       'transition': ['aero.dih']
                   },
                   opt=True)
traj.add_parameter(name='Vw', val=0.0, units='m/s',
                   desc='Wind speed along the runway, defined as positive for a headwind',
                   targets={
                       'initialrun': ['Vw'], 'rotation': ['Vw'],
                       'transition': ['Vw']
                   },
                   opt=False, dynamic=False)
traj.add_parameter(name='flap_angle', val=0.0, units='deg', desc='Flap defletion',
                   targets={
                       'initialrun': ['aero.flap_angle'], 'rotation': ['aero.flap_angle'],
                       'transition': ['aero.flap_angle']
                   },
                   opt=False, dynamic=False)
traj.add_parameter(name='elevation', val=0.0, units='m', desc='Runway elevation',
                   targets={
                       'initialrun': ['elevation'], 'rotation': ['elevation'],
                       'transition': ['elevation']
                   },
                   opt=False, dynamic=False)
traj.add_parameter(name='rw_slope', val=0.0, units='rad', desc='Runway slope',
                   targets={
                       'initialrun': ['initial_run_eom.rw_slope'],
                       'rotation': ['rotation_eom.rw_slope'],
                   },
                   opt=False, dynamic=False)

traj.link_phases(phases=['initialrun', 'rotation'],
                 vars=['time', 'V', 'x', 'mass', 'de', 'h'])
traj.link_phases(phases=['rotation', 'transition'],
                 vars=['time', 'V', 'x', 'mass', 'de', 'theta', 'q'])

p.setup()

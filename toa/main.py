import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt

from toa.data import get_airplane_data
from toa.ode.initialrun_ode import InitialRunODE
from toa.ode.rotation_ode import RotationODE
from dymos.examples.plotting import plot_results

from toa.runway import Runway

p = om.Problem(model=om.Group())

# set driver
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'

runway = Runway(2200, 0.0, 0.0, 0.0, 0.0)
airplane = get_airplane_data('b744')

traj = p.model.add_subsystem('traj', dm.Trajectory())
initialrun = dm.Phase(ode_class=InitialRunODE,
                      transcription=dm.GaussLobatto(num_segments=20),
                      ode_init_kwargs={'airplane': airplane})
traj.add_phase(name='initialrun', phase=initialrun)

initialrun.set_time_options(fix_initial=True, units='s')

# Initial run states
initialrun.add_state(name='V', units='kn', rate_source='initial_run_eom.v_dot',
                     targets=['V'], fix_initial=True, fix_final=False, lower=0)
initialrun.add_state(name='x', units='m', rate_source='initial_run_eom.x_dot',
                     fix_initial=True, fix_final=False, lower=0)
initialrun.add_state(name='mass', units='kg', rate_source='prop.m_dot',
                     targets=['mass'], fix_initial=False, fix_final=False)

# Initial run parameters
initialrun.add_parameter(name='de', val=0.0, units='deg', desc='Elevator deflection',
                         lower=-30.0, upper=30.0, targets=['aero.de'], opt=True)
initialrun.add_parameter(name='theta', val=0.0, units='deg', desc='Pitch Angle',
                         lower=-5.0, upper=5.0,
                         targets=['aero.alpha', 'initial_run_eom.alpha'], opt=True)

# Initial run path constraints
initialrun.add_path_constraint(name='initial_run_eom.f_mg', lower=0, units='N')
initialrun.add_path_constraint(name='initial_run_eom.f_ng', lower=0, units='N')

# Initial run boundary constraint
initialrun.add_boundary_constraint(name='mass', loc='initial', units='kg',
                                   upper=airplane.limits.MTOW)
initialrun.add_boundary_constraint(name='initial_run_eom.f_ng', loc='final', units='N',
                                   lower=0.0, upper=1.0)

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
                    transcription=dm.GaussLobatto(num_segments=5),
                    ode_init_kwargs={'airplane': airplane})
traj.add_phase(name='rotation', phase=rotation)

rotation.set_time_options(fix_initial=False, units='s')

# Rotation states
rotation.add_state(name='V', units='kn', rate_source='rotation_eom.v_dot',
                   targets=['V'], fix_initial=False, fix_final=False, lower=0)
rotation.add_state(name='x', units='m', rate_source='rotation_eom.x_dot',
                   fix_initial=False, fix_final=False, lower=0)
rotation.add_state(name='mass', units='kg', rate_source='prop.m_dot', targets=['mass'],
                   fix_initial=False, fix_final=False, lower=0)
rotation.add_state(name='theta', units='deg', rate_source='rotation_eom.theta_dot',
                   targets=['alpha'],
                   fix_initial=False, fix_final=False, lower=0, upper=20)
rotation.add_state(name='q', units='deg/s', rate_source='rotation_eom.q_dot',
                   targets=['q'],
                   fix_initial=True, fix_final=False, lower=0, upper=30)

# Rotation controls
rotation.add_control(name='de', units='deg', desc='Elevator defletion', opt=True,
                     fix_initial=False, fix_final=False, targets=['aero.de'],
                     lower=-30.0, upper=30.0)

# Rotation path constraints
rotation.add_path_constraint(name='rotation_eom.f_mg', lower=0, units='N')

# Rotation boundary constraint
rotation.add_boundary_constraint(name='x', loc='final', units='m', upper=runway.tora)
rotation.add_boundary_constraint(name='rotation_eom.f_mg', loc='final', units='N',
                                 lower=0.0, upper=1.0)

rotation.add_timeseries_output('aero.CD')
rotation.add_timeseries_output('aero.CL')
rotation.add_timeseries_output('aero.Cm')
rotation.add_timeseries_output('rotation_eom.f_mg', units='kN')
rotation.add_timeseries_output('prop.thrust', units='kN')
rotation.add_timeseries_output('prop.m_dot', units='kg/s')
# Trajectory parameters
traj.add_parameter(name='elevation', val=0.0, units='m', desc='Runway elevation',
                    targets={'initialrun': ['elevation'], 'rotation': ['elevation']},
                    opt=False, dynamic=False)
traj.add_parameter(name='rw_slope', val=0.0, units='rad', desc='Runway slope',
                    targets={
                        'initialrun': ['initial_run_eom.rw_slope'],
                        'rotation': ['rotation_eom.rw_slope']
                    },
                    opt=False, dynamic=False)

traj.link_phases(phases=['initialrun', 'rotation'],
                 vars=['V', 'x', 'mass', 'de', 'theta'])

p.setup()

p.set_val('traj.initialrun.t_initial', 0)
p.set_val('traj.initialrun.t_duration', 60)

p.set_val('traj.parameters:elevation', runway.elevation)
p.set_val('traj.parameters:rw_slope', runway.slope)

p['traj.initialrun.states:x'] = initialrun.interpolate(ys=[0, 0.8 * runway.tora],
                                                       nodes='state_input')
p['traj.initialrun.states:V'] = initialrun.interpolate(ys=[0, 150], nodes='state_input')
p['traj.initialrun.states:mass'] = initialrun.interpolate(
        ys=[airplane.limits.MTOW, airplane.limits.MTOW - 600], nodes='state_input')
p['traj.initialrun.parameters:de'] = 0.0
p['traj.initialrun.parameters:theta'] = 0.0

p['traj.rotation.states:x'] = rotation.interpolate(ys=[0.8 * runway.tora, runway.tora],
                                                   nodes='state_input')
p['traj.rotation.states:V'] = rotation.interpolate(ys=[150, 200], nodes='state_input')
p['traj.rotation.states:mass'] = rotation.interpolate(
        ys=[airplane.limits.MTOW - 600, airplane.limits.MTOW - 1000],
        nodes='state_input')
p['traj.rotation.states:theta'] = rotation.interpolate(ys=[0.0, 20.0],
                                                       nodes='state_input')
p['traj.rotation.states:q'] = rotation.interpolate(ys=[0.0, 30.0],
                                                       nodes='state_input')

dm.run_problem(p)
sim_out = traj.simulate()

print(f"RTOW: {p.get_val('traj.initialrun.timeseries.states:mass', units='kg')[0]} kg")
print(
    f"Rotation speed (VR): {p.get_val('traj.initialrun.timeseries.states:V', units='kn')[-1]} kn")
print(
    f"Liftoff speed (Vlof): {p.get_val('traj.rotation.timeseries.states:V', units='kn')[-1]} kn")
print(
    f"Takeoff distance: {p.get_val('traj.rotation.timeseries.states:x', units='m')[-1]} m")
## Plots

time_driver = {
    'initialrun': p.get_val('traj.initialrun.timeseries.time'),
    'rotation': p.get_val('traj.rotation.timeseries.time')
}
time_sim = {
    'initialrun': sim_out.get_val('traj.initialrun.timeseries.time'),
    'rotation': sim_out.get_val('traj.rotation.timeseries.time')
}
de_driver = {
    'initialrun': p.get_val('traj.initialrun.timeseries.parameters:de'),
    'rotation': p.get_val('traj.rotation.timeseries.controls:de')
}
de_sim = {
    'initialrun': sim_out.get_val('traj.initialrun.timeseries.parameters:de'),
    'rotation': sim_out.get_val('traj.rotation.timeseries.controls:de')
}
theta_driver = {
    'initialrun': p.get_val('traj.initialrun.timeseries.parameters:theta'),
    'rotation': p.get_val('traj.rotation.timeseries.states:theta')
}
theta_sim = {
    'initialrun': sim_out.get_val('traj.initialrun.timeseries.parameters:theta'),
    'rotation': sim_out.get_val('traj.rotation.timeseries.states:theta')
}
q_driver = {
    'initialrun': len(p.get_val('traj.initialrun.timeseries.time'))*[0.0],
    'rotation': p.get_val('traj.rotation.timeseries.states:q')
}
q_sim = {
    'initialrun': len(sim_out.get_val('traj.initialrun.timeseries.time'))*[0.0],
    'rotation': sim_out.get_val('traj.rotation.timeseries.states:q')
}

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
commom_states = ['x', 'V', 'mass']
for i, state in enumerate(commom_states):
    x_driver = {
    'initialrun': p.get_val(f"traj.initialrun.timeseries.states:{state}"),
    'rotation': p.get_val(f"traj.rotation.timeseries.states:{state}")
}
    x_sim = {
        'initialrun': sim_out.get_val(f"traj.initialrun.timeseries.states:{state}"),
        'rotation': sim_out.get_val(f"traj.rotation.timeseries.states:{state}")
    }
    
    axes[i].set_ylabel(state)
    axes[i].plot(time_driver['initialrun'], x_driver['initialrun'], 'bo')
    axes[i].plot(time_driver['rotation'], x_driver['rotation'], 'ro')
    axes[i].plot(time_sim['initialrun'], x_sim['initialrun'], 'b-')
    axes[i].plot(time_sim['rotation'], x_sim['rotation'], 'r-')

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 6))
params = ['CL', 'CD', 'Cm', 'm_dot', 'thrust']
for i, param in enumerate(params):
    x_driver = {
        'initialrun': p.get_val(f"traj.initialrun.timeseries.{param}"),
        'rotation': p.get_val(f"traj.rotation.timeseries.{param}")
    }
    x_sim = {
        'initialrun': sim_out.get_val(f"traj.initialrun.timeseries.{param}"),
        'rotation': sim_out.get_val(f"traj.rotation.timeseries.{param}")
    }

    axes[i].set_ylabel(param)
    axes[i].plot(time_driver['initialrun'], x_driver['initialrun'], 'bo')
    axes[i].plot(time_driver['rotation'], x_driver['rotation'], 'ro')
    axes[i].plot(time_sim['initialrun'], x_sim['initialrun'], 'b-')
    axes[i].plot(time_sim['rotation'], x_sim['rotation'], 'r-')

plt.subplot(3,1,1)
plt.plot(time_driver['initialrun'], de_driver['initialrun'], 'bo')
plt.plot(time_driver['rotation'], de_driver['rotation'], 'ro')
plt.plot(time_sim['initialrun'], de_sim['initialrun'], 'b-')
plt.plot(time_sim['rotation'], de_sim['rotation'], 'r-')

plt.subplot(3,1,2)
plt.plot(time_driver['initialrun'], theta_driver['initialrun'], 'bo')
plt.plot(time_driver['rotation'], theta_driver['rotation'], 'ro')
plt.plot(time_sim['initialrun'], theta_sim['initialrun'], 'b-')
plt.plot(time_sim['rotation'], theta_sim['rotation'], 'r-')

plt.subplot(3,1,3)
plt.plot(time_driver['initialrun'], q_driver['initialrun'], 'bo')
plt.plot(time_driver['rotation'], q_driver['rotation'], 'ro')
plt.plot(time_sim['initialrun'], q_sim['initialrun'], 'b-')
plt.plot(time_sim['rotation'], q_sim['rotation'], 'r-')

plt.show()

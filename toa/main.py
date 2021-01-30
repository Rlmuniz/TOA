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
                     fix_initial=True, fix_final=False, lower=0)
initialrun.add_state(name='mass', units='kg', rate_source='prop.m_dot',
                     targets=['mass'], fix_initial=False, fix_final=False)

# Initial run parameters
initialrun.add_parameter(name='de', val=0.0, units='deg', desc='Elevator deflection',
                         lower=-30.0, upper=30.0, targets=['aero.de'], opt=True)
initialrun.add_parameter(name='alpha', val=0.0, units='deg', desc='Angle of Attack',
                         targets=['aero.alpha', 'initial_run_eom.alpha'], opt=False)

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
                   fix_initial=True, fix_final=False, lower=0, upper=20)
rotation.add_state(name='q', units='deg/s', rate_source='rotation_eom.q_dot',
                   targets=['q'], fix_initial=True, fix_final=False)

# Rotation controls
rotation.add_control(name='de', units='deg', desc='Elevator defletion', opt=True,
                     fix_initial=False, fix_final=False, targets=['aero.de'],
                     lower=-30.0, upper=30.0)

# Rotation path constraints
rotation.add_path_constraint(name='rotation_eom.f_mg', lower=0, units='N')
rotation.add_path_constraint(name='q', units='deg/s', lower=0.0, upper=10.0)
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

## Transition
transition = dm.Phase(ode_class=TransitionODE,
                      transcription=dm.GaussLobatto(num_segments=5),
                      ode_init_kwargs={'airplane': airplane})
traj.add_phase(name='transition', phase=transition)

transition.set_time_options(fix_initial=False, units='s')

# Transition states
transition.add_state(name='V', units='kn', rate_source='transition_eom.v_dot',
                     targets=['V'], fix_initial=False, fix_final=False, lower=0)
transition.add_state(name='gam', units='deg', rate_source='transition_eom.gam_dot',
                     targets=['gam'], fix_initial=True, fix_final=False)
transition.add_state(name='x', units='m', rate_source='transition_eom.x_dot',
                     fix_initial=False, fix_final=False, lower=0)
transition.add_state(name='h', units='ft', rate_source='transition_eom.h_dot',
                     fix_initial=True, lower=0)
transition.add_state(name='theta', units='deg', rate_source='transition_eom.theta_dot',
                     targets=['theta'],
                     fix_initial=False, fix_final=False, lower=0, upper=20)
transition.add_state(name='q', units='deg/s', rate_source='transition_eom.q_dot',
                     targets=['q'],
                     fix_initial=False, fix_final=False, lower=0, upper=30)
transition.add_state(name='mass', units='kg', rate_source='prop.m_dot',
                     targets=['mass'],
                     fix_initial=False, fix_final=False, lower=0)

# Transition controls
transition.add_control(name='de', units='deg', desc='Elevator deflection', opt=True,
                       fix_initial=False, fix_final=False, targets=['aero.de'],
                       lower=-30.0, upper=30.0)

# Transition path constraints
transition.add_path_constraint(name='h', lower=0.0, upper=35.0)
transition.add_path_constraint(name='gam', lower=0.0)
# Rotation boundary constraint
transition.add_boundary_constraint(name='x', loc='final', units='m', upper=runway.toda)
transition.add_boundary_constraint(name='h', loc='final', units='ft', equals=35.0)

transition.add_timeseries_output('aero.CD')
transition.add_timeseries_output('aero.CL')
transition.add_timeseries_output('aero.Cm')
transition.add_timeseries_output('alpha_comp.alpha', units='deg')
transition.add_timeseries_output('prop.thrust', units='kN')
transition.add_timeseries_output('prop.m_dot', units='kg/s')

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
                 vars=['time', 'V', 'x', 'mass', 'de'])
traj.link_phases(phases=['rotation', 'transition'],
                 vars=['time', 'V', 'x', 'mass', 'de', 'theta', 'q'])

p.setup()

p.set_val('traj.initialrun.t_initial', 0)
p.set_val('traj.initialrun.t_duration', 60)
p.set_val('traj.rotation.t_initial', 60)
p.set_val('traj.rotation.t_duration', 5)
p.set_val('traj.transition.t_initial', 65)
p.set_val('traj.transition.t_duration', 10)

p.set_val('traj.parameters:elevation', runway.elevation)
p.set_val('traj.parameters:rw_slope', runway.slope)
p.set_val('traj.parameters:flap_angle', 0.0)
p.set_val('traj.parameters:dih', 0.0)
p.set_val('traj.parameters:Vw', 0.0)

p['traj.initialrun.states:x'] = initialrun.interpolate(ys=[0, 0.7 * runway.tora],
                                                       nodes='state_input')
p['traj.initialrun.states:V'] = initialrun.interpolate(ys=[0, 150], nodes='state_input')
p['traj.initialrun.states:mass'] = initialrun.interpolate(
        ys=[airplane.limits.MTOW, airplane.limits.MTOW - 600], nodes='state_input')
p['traj.initialrun.parameters:de'] = 0.0
p['traj.initialrun.parameters:alpha'] = 0.0

p['traj.rotation.states:x'] = rotation.interpolate(
        ys=[0.7 * runway.tora, 0.8 * runway.tora],
        nodes='state_input')
p['traj.rotation.states:V'] = rotation.interpolate(ys=[150, 160], nodes='state_input')
p['traj.rotation.states:mass'] = rotation.interpolate(
        ys=[airplane.limits.MTOW - 600, airplane.limits.MTOW - 1000],
        nodes='state_input')
p['traj.rotation.states:theta'] = rotation.interpolate(ys=[0.0, 15.0],
                                                       nodes='state_input')
p['traj.rotation.states:q'] = rotation.interpolate(ys=[0.0, 10.0],
                                                   nodes='state_input')

p['traj.transition.states:x'] = transition.interpolate(
        ys=[0.8 * runway.tora, runway.toda],
        nodes='state_input')
p['traj.transition.states:h'] = transition.interpolate(ys=[0.0, 35.0],
                                                       nodes='state_input')
p['traj.transition.states:V'] = transition.interpolate(ys=[160, 200],
                                                       nodes='state_input')
p['traj.transition.states:gam'] = transition.interpolate(ys=[0.0, 10.0],
                                                         nodes='state_input')
p['traj.transition.states:mass'] = transition.interpolate(
        ys=[airplane.limits.MTOW - 1000, airplane.limits.MTOW - 1400],
        nodes='state_input')
p['traj.transition.states:theta'] = transition.interpolate(ys=[15.0, 10.0],
                                                           nodes='state_input')
p['traj.transition.states:q'] = transition.interpolate(ys=[10.0, 0.0],
                                                       nodes='state_input')

dm.run_problem(p)
sim_out = traj.simulate()

print(f"RTOW: {p.get_val('traj.initialrun.timeseries.states:mass', units='kg')[0]} kg")
print(
    f"Rotation speed (VR): {p.get_val('traj.initialrun.timeseries.states:V', units='kn')[-1]} kn")
print(
    f"Liftoff speed (Vlof): {p.get_val('traj.rotation.timeseries.states:V', units='kn')[-1]} kn")
print(
    f"End of transition speed (V3): {p.get_val('traj.transition.timeseries.states:V', units='kn')[-1]} kn")
print(
    f"Takeoff distance: {p.get_val('traj.transition.timeseries.states:x', units='m')[-1]} m")
print(f"Horizontal stabilizer: {p.get_val('traj.parameters:dih')} deg")

## Plots
plot_results([('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.states:V',
               'Time (s)', 'Speed (kt)'),
              ('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.states:mass',
               'time (s)', 'Mass (kg)'),
              ],
             title='Initial Run',
             p_sol=p, p_sim=sim_out)

plot_results([('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:V',
               'Time (s)', 'Speed (kt)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:mass',
               'time (s)', 'Mass (kg)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:theta',
               'time (s)', 'Pitch angle (deg)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:q',
               'time (s)', 'Pitch rate (deg/s)'),
              ],
             title='Rotation',
             p_sol=p, p_sim=sim_out)

plot_results([('traj.transition.timeseries.time', 'traj.transition.timeseries.states:V',
               'Time (s)', 'Speed (kt)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:h',
               'time (s)', 'Height (ft)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:mass',
               'time (s)', 'Mass (kg)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:theta',
               'time (s)', 'theta (deg)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:q',
               'time (s)', 'q (deg/s)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:gam',
               'time (s)', 'gamma (deg)'),
              ],
             title='Transition',
             p_sol=p, p_sim=sim_out)

plot_results([('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.CL',
               'Time (s)', 'Initial Run'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.CL',
               'Time (s)', 'Rotation'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.CL',
               'Time (s)', 'Transition'),
              ],
             title='Lift Coefficent',
             p_sol=p, p_sim=sim_out)
plot_results([('traj.initialrun.timeseries.time', 'traj.initialrun.timeseries.parameters:de',
               'Time (s)', 'Initial Run'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.controls:de',
               'Time (s)', 'Rotation'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.controls:de',
               'Time (s)', 'Transition'),
              ],
             title='Elevator Deflection',
             p_sol=p, p_sim=sim_out)
plot_results([('traj.transition.timeseries.time', 'traj.transition.timeseries.alpha',
               'Time (s)', 'Alpha'),],
             title='Alpha Transition',
             p_sol=p, p_sim=sim_out)
"""
time_driver = {
    'initialrun': p.get_val('traj.initialrun.timeseries.time'),
    'rotation': p.get_val('traj.rotation.timeseries.time'),
    'transition': p.get_val('traj.transition.timeseries.time')
}
time_sim = {
    'initialrun': sim_out.get_val('traj.initialrun.timeseries.time'),
    'rotation': sim_out.get_val('traj.rotation.timeseries.time'),
    'transition': sim_out.get_val('traj.transition.timeseries.time')
}
de_driver = {
    'initialrun': p.get_val('traj.initialrun.timeseries.parameters:de'),
    'rotation': p.get_val('traj.rotation.timeseries.controls:de'),
    'transition': p.get_val('traj.transition.timeseries.controls:de')
}
de_sim = {
    'initialrun': sim_out.get_val('traj.initialrun.timeseries.parameters:de'),
    'rotation': sim_out.get_val('traj.rotation.timeseries.controls:de'),
    'transition': sim_out.get_val('traj.transition.timeseries.controls:de')
}
theta_driver = {
    'initialrun': p.get_val('traj.initialrun.timeseries.parameters:alpha'),
    'rotation': p.get_val('traj.rotation.timeseries.states:theta'),
    'transition': p.get_val('traj.transition.timeseries.states:theta')
}
theta_sim = {
    'initialrun': sim_out.get_val('traj.initialrun.timeseries.parameters:alpha'),
    'rotation': sim_out.get_val('traj.rotation.timeseries.states:theta'),
    'transition': sim_out.get_val('traj.transition.timeseries.states:theta'),
}
q_driver = {
    'initialrun': len(p.get_val('traj.initialrun.timeseries.time')) * [0.0],
    'rotation': p.get_val('traj.rotation.timeseries.states:q'),
    'transition': p.get_val('traj.transition.timeseries.states:q'),
}
q_sim = {
    'initialrun': len(sim_out.get_val('traj.initialrun.timeseries.time')) * [0.0],
    'rotation': sim_out.get_val('traj.rotation.timeseries.states:q'),
    'transition': sim_out.get_val('traj.transition.timeseries.states:q'),
}

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
commom_states = ['x', 'V', 'mass']
for i, state in enumerate(commom_states):
    x_driver = {
        'initialrun': p.get_val(f"traj.initialrun.timeseries.states:{state}"),
        'rotation': p.get_val(f"traj.rotation.timeseries.states:{state}"),
        'transition': p.get_val(f"traj.transition.timeseries.states:{state}"),
    }
    x_sim = {
        'initialrun': sim_out.get_val(f"traj.initialrun.timeseries.states:{state}"),
        'rotation': sim_out.get_val(f"traj.rotation.timeseries.states:{state}"),
        'transition': sim_out.get_val(f"traj.transition.timeseries.states:{state}"),
    }

    axes[i].plot(time_driver['initialrun'], x_driver['initialrun'], marker='o',
                 color='tab:red', linestyle='None',
                 label='solution' if i == 0 else None)
    axes[i].plot(time_driver['rotation'], x_driver['rotation'], marker='o',
                 color='tab:red', linestyle='None')
    axes[i].plot(time_driver['transition'], x_driver['transition'], marker='o',
                 color='tab:red', linestyle='None')
    axes[i].plot(time_sim['initialrun'], x_sim['initialrun'], marker=None,
                 color='tab:blue', linestyle='-')
    axes[i].plot(time_sim['rotation'], x_sim['rotation'], marker=None, color='tab:blue',
                 linestyle='-', label='simulation' if i == 0 else None)
    axes[i].plot(time_sim['transition'], x_sim['transition'], marker=None,
                 color='tab:blue', linestyle='-')
    axes[i].set_ylabel(state)
    axes[i].set_xlabel('Time (s)')
    axes[i].grid(True)
    fig.legend(loc='lower center', ncol=2)

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 6))
params = ['CL', 'CD', 'Cm', 'm_dot', 'thrust']
for i, param in enumerate(params):
    x_driver = {
        'initialrun': p.get_val(f"traj.initialrun.timeseries.{param}"),
        'rotation': p.get_val(f"traj.rotation.timeseries.{param}"),
        'transition': p.get_val(f"traj.transition.timeseries.{param}")
    }
    x_sim = {
        'initialrun': sim_out.get_val(f"traj.initialrun.timeseries.{param}"),
        'rotation': sim_out.get_val(f"traj.rotation.timeseries.{param}"),
        'transition': sim_out.get_val(f"traj.transition.timeseries.{param}")
    }

    axes[i].plot(time_driver['initialrun'], x_driver['initialrun'], marker='o',
                 color='tab:red', linestyle='None',
                 label='solution' if i == 0 else None)
    axes[i].plot(time_driver['rotation'], x_driver['rotation'], marker='o',
                 color='tab:red', linestyle='None')
    axes[i].plot(time_driver['transition'], x_driver['transition'], marker='o',
                 color='tab:red', linestyle='None')
    axes[i].plot(time_sim['initialrun'], x_sim['initialrun'], marker=None,
                 color='tab:blue', linestyle='-')
    axes[i].plot(time_sim['rotation'], x_sim['rotation'], marker=None, color='tab:blue',
                 linestyle='-', label='simulation' if i == 0 else None)
    axes[i].plot(time_sim['transition'], x_sim['transition'], marker=None,
                 color='tab:blue', linestyle='-')
    axes[i].set_ylabel(param)
    axes[i].set_xlabel('Time (s)')
    axes[i].grid(True)
    fig.legend(loc='lower center', ncol=2)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time_driver['initialrun'], de_driver['initialrun'], marker='o',
         color='tab:red', linestyle='None', label='solution')
plt.plot(time_driver['rotation'], de_driver['rotation'], marker='o', color='tab:red',
         linestyle='None')
plt.plot(time_sim['initialrun'], de_sim['initialrun'], marker=None, color='tab:blue',
         linestyle='-')
plt.plot(time_sim['rotation'], de_sim['rotation'], marker=None, color='tab:blue',
         linestyle='-', label='simulation')
plt.xlabel('Time (s)')
plt.ylabel('Elevator (deg)')

plt.subplot(3, 1, 2)
plt.plot(time_driver['initialrun'], theta_driver['initialrun'], marker='o',
         color='tab:red', linestyle='None', label='solution')
plt.plot(time_driver['rotation'], theta_driver['rotation'], marker='o', color='tab:red',
         linestyle='None')
plt.plot(time_sim['initialrun'], theta_sim['initialrun'], marker=None, color='tab:blue',
         linestyle='-')
plt.plot(time_sim['rotation'], theta_sim['rotation'], marker=None, color='tab:blue',
         linestyle='-', label='simulation')
plt.xlabel('Time (s)')
plt.ylabel('Theta (deg)')

plt.subplot(3, 1, 3)
plt.plot(time_driver['initialrun'], q_driver['initialrun'], marker='o', color='tab:red',
         linestyle='None', label='solution')
plt.plot(time_driver['rotation'], q_driver['rotation'], marker='o', color='tab:red',
         linestyle='None')
plt.plot(time_sim['initialrun'], q_sim['initialrun'], marker=None, color='tab:blue',
         linestyle='-')
plt.plot(time_sim['rotation'], q_sim['rotation'], marker=None, color='tab:blue',
         linestyle='-', label='simulation')
plt.xlabel('Time (s)')
plt.ylabel('Pitch rate (deg/s)')
plt.legend(loc='lower center', ncol=2)

plt.figure()
plt.plot(time_driver['initialrun'], p.get_val(f"traj.initialrun.timeseries.f_mg"),
         marker='o', color='tab:red', linestyle='None', label='solution')
plt.plot(time_driver['initialrun'], p.get_val(f"traj.initialrun.timeseries.f_ng"),
         marker='o', color='tab:orange', linestyle='None', label='solution')
plt.plot(time_driver['rotation'], p.get_val(f"traj.rotation.timeseries.f_mg"),
         marker='o', color='tab:red', linestyle='None')
plt.plot(time_sim['initialrun'], sim_out.get_val(f"traj.initialrun.timeseries.f_mg"),
         marker=None, color='tab:blue', linestyle='-')
plt.plot(time_sim['initialrun'], sim_out.get_val(f"traj.initialrun.timeseries.f_ng"),
         marker=None, color='tab:orange', linestyle='-')
plt.plot(time_sim['rotation'], sim_out.get_val(f"traj.rotation.timeseries.f_mg"),
         marker=None, color='tab:blue', linestyle='-', label='simulation')
plt.legend(loc='lower center', ncol=2)
plt.grid()
plt.show()
"""

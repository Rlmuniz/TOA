import openmdao.api as om
import dymos as dm

from toa.data import get_airplane_data
from toa.ode.initialrun_ode import InitialRunODE
from toa.ode.rotation_ode import RotationODE
from toa.runway import Runway


def run_takeoff_no_transition(airplane, runway, flap_angle=0.0, wind_speed=0.0):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.declare_coloring()

    p.model.linear_solver = om.DirectSolver()

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    # --------------------------------------------- Initial Run --------------------------------------------------------
    initial_run = dm.Phase(ode_class=InitialRunODE,
                           transcription=dm.GaussLobatto(num_segments=20, compressed=False),
                           ode_init_kwargs={'airplane': airplane})

    traj.add_phase('initial_run', initial_run)

    initial_run.set_time_options(fix_initial=True, units='s', duration_bounds=(10, 100))

    # Initial run states
    initial_run.add_state(name='V', units='m/s', rate_source='initial_run_eom.v_dot', targets=['V'], fix_initial=True,
                          fix_final=False, lower=0)
    initial_run.add_state(name='x', units='m', rate_source='initial_run_eom.x_dot',
                          targets=['mlg_pos.x'], fix_initial=True, fix_final=False,
                          lower=airplane.landing_gear.main.x)
    initial_run.add_state(name='mass', units='kg', rate_source='prop.m_dot',
                          targets=['mass'], fix_initial=False, fix_final=False, lower=0.0,
                          upper=airplane.limits.MTOW)

    # Initial run parameters
    initial_run.add_parameter(name='de', val=0.0, units='deg', desc='Elevator deflection',
                              targets=['aero.de'], opt=False)
    initial_run.add_parameter(name='theta', val=0.0, units='deg', desc='Pitch Angle',
                              targets=['aero.alpha', 'initial_run_eom.alpha',
                                       'mlg_pos.theta'], opt=False)
    initial_run.add_parameter(name='h', val=0.0, units='m',
                              desc='Vertical CG position',
                              targets=['mlg_pos.h'], opt=False)

    # path constraint
    initial_run.add_path_constraint(name='initial_run_eom.f_mg', lower=0, units='N')
    initial_run.add_path_constraint(name='initial_run_eom.f_ng', lower=0, units='N')

    initial_run.add_boundary_constraint(name='initial_run_eom.f_ng', loc='final', units='N', lower=0.0, upper=0.5,
                                        shape=(1,))

    initial_run.add_objective('mass', loc='initial', scaler=-1)

    initial_run.add_timeseries_output('initial_run_eom.f_mg', units='kN')
    initial_run.add_timeseries_output('initial_run_eom.f_ng', units='kN')
    initial_run.add_timeseries_output('aero.CL')
    initial_run.add_timeseries_output('aero.CD')
    initial_run.add_timeseries_output('aero.Cm')
    # --------------------------------------------- Rotation -----------------------------------------------------------
    rotation = dm.Phase(ode_class=RotationODE,
                        transcription=dm.GaussLobatto(num_segments=5, compressed=False),
                        ode_init_kwargs={'airplane': airplane})
    traj.add_phase(name='rotation', phase=rotation)

    rotation.set_time_options(fix_initial=False, units='s', duration_bounds=(1, 6))

    # Rotation states
    rotation.add_state(name='V', units='m/s', rate_source='rotation_eom.v_dot',
                       targets=['V'], fix_initial=False, fix_final=False, lower=0)
    rotation.add_state(name='x', units='m', rate_source='rotation_eom.x_dot',
                       targets=['mlg_pos.x'], fix_initial=False, fix_final=False, lower=0)
    rotation.add_state(name='h', units='m', rate_source='rotation_eom.h_dot',
                       targets=['mlg_pos.h'], fix_initial=True, fix_final=False)
    rotation.add_state(name='mass', units='kg', rate_source='prop.m_dot', targets=['mass'],
                       fix_initial=False, fix_final=False, lower=0.0)
    rotation.add_state(name='theta', units='deg', rate_source='rotation_eom.theta_dot',
                       targets=['aero.alpha', 'rotation_eom.alpha', 'mlg_pos.theta'],
                       fix_initial=True, fix_final=False, lower=0.0)
    rotation.add_state(name='q', units='deg/s', rate_source='rotation_eom.q_dot',
                       targets=['q'], fix_initial=True, fix_final=False, lower=0.0)

    # Rotation controls
    rotation.add_control(name='de', units='deg', lower=-30.0, upper=30.0, targets=['aero.de'], fix_initial=True,
                         rate_continuity=True)

    # Rotation path constraints
    rotation.add_path_constraint(name='rotation_eom.f_mg', lower=0, units='N')

    # Rotation boundary constraint
    rotation.add_boundary_constraint(name='rotation_eom.f_mg', loc='final', units='N',
                                     lower=0.0, upper=0.5, shape=(1,))
    rotation.add_boundary_constraint(name='x', loc='final', units='m', upper=runway.tora, shape=(1,))

    rotation.add_timeseries_output('rotation_eom.f_mg', units='kN')
    rotation.add_timeseries_output('aero.CL')
    rotation.add_timeseries_output('aero.CD')
    rotation.add_timeseries_output('aero.Cm')
    # ---------------------------------------- Trajectory Parameters ---------------------------------------------------
    traj.add_parameter(name='dih', val=0.0, units='deg', lower=-5.0, upper=5.0,
                       desc='Horizontal stabilizer angle',
                       targets={
                           'initial_run': ['aero.dih'],
                           'rotation': ['aero.dih'],
                           },
                       opt=True, dynamic=False)
    traj.add_parameter(name='Vw', val=0.0, units='m/s',
                       desc='Wind speed along the runway, defined as positive for a headwind',
                       targets={
                           'initial_run': ['Vw'],
                           'rotation': ['Vw'],
                           },
                       opt=False, dynamic=False)
    traj.add_parameter(name='flap_angle', val=0.0, units='deg', desc='Flap defletion',
                       targets={
                           'initial_run': ['aero.flap_angle'],
                           'rotation': ['aero.flap_angle'],
                           },
                       opt=False, dynamic=False)
    traj.add_parameter(name='elevation', val=0.0, units='m', desc='Runway elevation',
                       targets={
                           'initial_run': ['elevation'],
                           'rotation': ['elevation'],
                           },
                       opt=False, dynamic=False)
    traj.add_parameter(name='rw_slope', val=0.0, units='rad', desc='Runway slope',
                       targets={
                           'initial_run': ['initial_run_eom.rw_slope'],
                           'rotation': ['rotation_eom.rw_slope'],
                           },
                       opt=False, dynamic=False)

    # ------------------------------------------------ Link Phases -----------------------------------------------------
    traj.link_phases(phases=['initial_run', 'rotation'], vars=['time', 'V', 'x', 'mass'])

    p.setup(check=True)

    p.set_val('traj.initial_run.t_initial', 0)
    p.set_val('traj.initial_run.t_duration', 60)
    p.set_val('traj.rotation.t_initial', 60)
    p.set_val('traj.rotation.t_duration', 3)

    p.set_val('traj.parameters:elevation', runway.elevation)
    p.set_val('traj.parameters:rw_slope', runway.slope)
    p.set_val('traj.parameters:flap_angle', flap_angle)
    p.set_val('traj.parameters:dih', 0.0)
    p.set_val('traj.parameters:Vw', wind_speed)

    p['traj.initial_run.states:x'] = initial_run.interpolate(
            ys=[airplane.landing_gear.main.x, 0.7 * runway.tora],
            nodes='state_input')
    p['traj.initial_run.states:V'] = initial_run.interpolate(ys=[0, 70], nodes='state_input')
    p['traj.initial_run.states:mass'] = initial_run.interpolate(
            ys=[airplane.limits.MTOW, airplane.limits.MTOW - 100], nodes='state_input')
    p['traj.initial_run.parameters:h'] = airplane.landing_gear.main.z

    p['traj.rotation.states:x'] = rotation.interpolate(
            ys=[0.7 * runway.tora, 0.8 * runway.tora],
            nodes='state_input')
    p['traj.rotation.states:V'] = rotation.interpolate(ys=[70, 80], nodes='state_input')
    p['traj.rotation.states:mass'] = rotation.interpolate(
            ys=[airplane.limits.MTOW - 100, airplane.limits.MTOW - 200],
            nodes='state_input')
    p['traj.rotation.states:h'] = airplane.landing_gear.main.z
    p['traj.rotation.states:q'] = rotation.interpolate(ys=[0.0, 10.0], nodes='state_input')
    p['traj.rotation.states:theta'] = rotation.interpolate(ys=[0.0, 15.0], nodes='state_input')
    p['traj.rotation.controls:de'] = rotation.interpolate(ys=[0.0, -20.0], nodes='control_input')

    dm.run_problem(p)
    sim_out = traj.simulate()

    print(f"RTOW: {p.get_val('traj.initial_run.timeseries.states:mass', units='kg')[0]} kg")
    print(f"Rotation speed (VR): {p.get_val('traj.initial_run.timeseries.states:V', units='kn')[-1]} kn")
    print(f"Vlof speed (Vlof): {p.get_val('traj.rotation.timeseries.states:V', units='kn')[-1]} kn")
    print(f"Horizontal stabilizer: {p.get_val('traj.parameters:dih')} deg")

    return p, sim_out

if __name__ == '__main__':
    runway = Runway(3000, 0.0, 0.0, 0.0, 0.0)
    airplane = get_airplane_data('b734')

    sol, sim = run_takeoff_no_transition(airplane, runway)
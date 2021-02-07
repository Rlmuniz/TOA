import matplotlib.pyplot as plt
import numpy as np
from dymos.examples.plotting import plot_results

from toa.data import get_airplane_data
from toa.runway import Runway
from toa.traj.aeo_no_transition import run_takeoff_no_transition

runway = Runway(3000, 0.0, 0.0, 0.0, 0.0)
airplane = get_airplane_data('b734')

p, sim = run_takeoff_no_transition(airplane, runway)

"""
time_imp = np.concatenate((p.get_val('traj.initial_run.timeseries.time'), p.get_val('traj.rotation.timeseries.time')), axis=None)
time_sim = np.concatenate((sim.get_val('traj.initial_run.timeseries.time'), sim.get_val('traj.rotation.timeseries.time')), axis=None)
v_imp = np.concatenate((p.get_val('traj.initial_run.timeseries.states:V'), p.get_val('traj.rotation.timeseries.states:V')), axis=None)
v_sim = np.concatenate((sim.get_val('traj.initial_run.timeseries.states:V'), sim.get_val('traj.rotation.timeseries.states:V')), axis=None)
x_imp = np.concatenate((p.get_val('traj.initial_run.timeseries.states:x'), p.get_val('traj.rotation.timeseries.states:x')), axis=None)
x_sim = np.concatenate((sim.get_val('traj.initial_run.timeseries.states:x'), sim.get_val('traj.rotation.timeseries.states:x')), axis=None)
mass_imp = np.concatenate((p.get_val('traj.initial_run.timeseries.states:mass'), p.get_val('traj.rotation.timeseries.states:mass')), axis=None)
mass_sim = np.concatenate((sim.get_val('traj.initial_run.timeseries.states:mass'), sim.get_val('traj.rotation.timeseries.states:mass')), axis=None)
fmg_imp = np.concatenate((p.get_val('traj.initial_run.timeseries.f_mg'), p.get_val('traj.rotation.timeseries.f_mg')), axis=None)
fmg_sim = np.concatenate((sim.get_val('traj.initial_run.timeseries.f_mg'), sim.get_val('traj.rotation.timeseries.f_mg')), axis=None)
"""
"""
plt.subplot(3, 1, 1)
plt.plot(time_imp, v_imp)
plt.plot(time_sim, v_sim)
plt.xlabel('Time (s)')
plt.ylabel('V (m/s)')

plt.subplot(3, 1, 2)
plt.plot(time_imp, x_imp)
plt.plot(time_sim, x_sim)
plt.xlabel('Time (s)')
plt.ylabel('X (m)')

plt.subplot(3, 1, 3)
plt.plot(time_imp, mass_imp)
plt.plot(time_sim, mass_sim)
plt.xlabel('Time (s)')
plt.ylabel('Mass (kg)')

plt.figure()
plt.plot(time_imp, fmg_imp)
plt.plot(time_sim, fmg_sim)
plt.plot(p.get_val('traj.initial_run.timeseries.time'), p.get_val('traj.initial_run.timeseries.f_ng'))
plt.plot(sim.get_val('traj.initial_run.timeseries.time'), sim.get_val('traj.initial_run.timeseries.f_ng'))

plt.figure()
plt.plot(p.get_val('traj.rotation.timeseries.controls:de'), p.get_val('traj.rotation.timeseries.controls:de'))
plt.plot(sim.get_val('traj.rotation.timeseries.controls:de'), sim.get_val('traj.rotation.timeseries.controls:de'))
plt.show()
"""
plot_results([('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.states:V',
               'Time (s)', 'V (m/s)'),
              ('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.states:mass',
               'time (s)', 'Mass (kg)')],
             title='Initial Run States',
             p_sol=p, p_sim=sim)

plot_results([('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.f_mg',
               'Time (s)', 'MLG Force (kN)'),
              ('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.f_ng',
               'time (s)', 'NLG Force (kN)')],
             title='Initial Run Timeseries',
             p_sol=p, p_sim=sim)

plot_results([('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.CL',
               'Time (s)', 'CL'),
              ('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.CD',
               'time (s)', 'CD'),
              ('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.Cm',
               'time (s)', 'Cm')],
             title='Initial Run Forces',
             p_sol=p, p_sim=sim)

plot_results([('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:V',
               'Time (s)', 'V (m/s)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:mass',
               'time (s)', 'Mass (kg)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:h',
               'time (s)', 'H (m)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:theta',
               'time (s)', 'Theta (deg)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:q',
               'time (s)', 'q (deg/s)')],
             title='Rotation States',
             p_sol=p, p_sim=sim)

plot_results([('traj.rotation.timeseries.time', 'traj.rotation.timeseries.CL',
               'Time (s)', 'CL'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.CD',
               'time (s)', 'CD'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.Cm',
               'time (s)', 'Cm')],
             title='Rotation Forces',
             p_sol=p, p_sim=sim)

plot_results([('traj.rotation.timeseries.time', 'traj.rotation.timeseries.controls:de',
               'Time (s)', 'De (deg)')],
             title='Elevaton Deflection',
             p_sol=p, p_sim=sim)

plt.show()
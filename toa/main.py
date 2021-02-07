from dymos.examples.plotting import plot_results
import matplotlib.pyplot as plt
from toa.data import get_airplane_data
from toa.runway import Runway
from toa.traj.aeo import run_takeoff

runway = Runway(3000, 0.0, 0.0, 0.0, 0.0)
airplane = get_airplane_data('b734')

p, sim = run_takeoff(airplane, runway, flap_angle=0)

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
               'Time (s)', 'De Rot(deg)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.controls:de',
               'Time (s)', 'De Tran(deg)')],
             title='Elevaton Deflection',
             p_sol=p, p_sim=sim)

plot_results([('traj.transition.timeseries.time', 'traj.transition.timeseries.states:V',
               'Time (s)', 'V (m/s)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:mass',
               'time (s)', 'Mass (kg)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:h',
               'time (s)', 'H (m)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:theta',
               'time (s)', 'Theta (deg)'),
              ('traj.transition.timeseries.time', 'traj.transition.timeseries.states:q',
               'time (s)', 'q (deg/s)')],
             title='Transition States',
             p_sol=p, p_sim=sim)
plt.show()
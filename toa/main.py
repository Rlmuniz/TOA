from dymos.examples.plotting import plot_results
import matplotlib.pyplot as plt
from toa.data import get_airplane_data
from toa.runway import Runway
from toa.traj.aeo import run_takeoff

runway = Runway(3000, 0.0, 0.0, 0.0, 0.0)
airplane = get_airplane_data('b734')

sol, sim = run_takeoff(airplane, runway)

plot_results([('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.states:V',
               'Time (s)', 'Speed (kt)'),
              ('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.initial_run.timeseries.time', 'traj.initial_run.timeseries.states:mass',
               'time (s)', 'Mass (kg)'),
              ],
             title='Initial Run',
             p_sol=sol, p_sim=sim)

plot_results([('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:V',
               'Time (s)', 'Speed (kt)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:x',
               'time (s)', 'Position (m)'),
              ('traj.rotation.timeseries.time', 'traj.rotation.timeseries.states:mass',
               'time (s)', 'Mass (kg)'),
              ],
             title='Rotation',
             p_sol=sol, p_sim=sim)

plt.show()


import matplotlib.pyplot as plt
from toa.data import get_airplane_data
from toa.runway import Runway
from toa.traj.aeo import run_takeoff
from toa.utils.plotting import plot_results

runway = Runway(2000, 0.0, 0.0, 0.0, 0.02)
airplane = get_airplane_data('b734')

p, sim = run_takeoff(airplane, runway, flap_angle=20.0, wind_speed=0.0)

plot_results(p, sim)
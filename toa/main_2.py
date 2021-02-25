
import matplotlib.pyplot as plt
from toa.data import get_airplane_data
from toa.runway import Runway
from toa.traj.aeo_poli import run_takeoff
from toa.utils.poli import plot_results
from toa.utils.validation import validate_result

runway = Runway(1800, 0.0, 0.0, 0.0, 0.0)
airplane = get_airplane_data('b734')

flap_angle = 5
p, sim = run_takeoff(airplane, runway, flap_angle=flap_angle, wind_speed=0.0, order=2)

validate_result(p, airplane, runway, flap_angle)

plot_results(p, sim)
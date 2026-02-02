import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy.f2py.crackfortran import verbose
from stable_baselines3 import PPO
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import inspect

from ._hidden_functions import drag_interp, thrust_main_interp, thrust_rot_interp

# Environment class
class OrbitMaintenanceEnv(gym.Env):
    """
    Enhanced environment with realistic thrusts, hierarchical control, v_theta check.
    High-level action: [des_delta_v_tang (norm), target_alpha (rad)]
    """
    def __init__(self):
        super().__init__()

        # Constants (scaled thrusts)
        self.mu = 3.986e14  # m^3/s^2 (Earth's gravitational parameter)
        self.R_earth = 6.371e6  # m (Earth radius)
        self.initial_altitude = 400000  # m (initial altitude above Earth surface)
        self.initial_r = self.R_earth + self.initial_altitude  # m (initial radial distance)
        self.initial_v_theta = np.sqrt(self.mu / self.initial_r)  # m/s (initial tangential velocity for circular orbit)
        self.initial_v_r = 0.0  # m/s (initial radial velocity)
        self.initial_phi = 0.0  # rad (initial orbital angle)
        self.initial_psi = np.pi / 2  # rad (initial attitude angle, aligned for prograde)
        self.initial_omega = self.initial_v_theta / self.initial_r  # rad/s (initial angular velocity for min drag alignment)
        self.dry_mass = 1000.0  # kg (dry mass of the station)
        self.initial_fuel_mass = 400.0  # kg (initial fuel mass, hydrazine)
        self.isp = 220.0  # s (specific impulse)
        self.g0 = 9.81  # m/s^2 (standard gravity)
        self.arm = 2.0  # m (lever arm for rotational thrusters)
        self.I = 1000.0  # kg m^2 (moment of inertia)

        self.drag_coeff = 2.2  # dimensionless (drag coefficient)
        self.area_mass_ratio = 0.1  # m^2/kg (area-to-mass ratio)
        self.atm_density_scale = 1e-12  # kg/m^3 (reference atmospheric density at initial altitude)
        self.scale_height = 100000  # m (atmospheric scale height)

        # Bounds for normalization
        self.altitude_bounds = 10000  # m (altitude deviation bound)
        self.v_r_bounds = 10  # m/s (radial velocity bound)
        self.v_theta_bounds = 10  # m/s (tangential velocity deviation bound)
        self.omega_dev_bounds = 0.01  # rad/s (angular velocity deviation bound)
        self.fuel_bounds = self.initial_fuel_mass  # kg (fuel mass bound for normalization)
        self.delta_v_bounds = 0.1  # m/s (desired delta-v bound)
        self.alpha_bounds = np.pi  # rad (alpha angle bound)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        #self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Norm des_delta_v, target_alpha
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)  # Norm des_delta_v, target_alpha

        self.dt = 600  # s (main time step)
        self.dt_sub = 10  # s (sub-step for attitude control)
        self.N_sub = int(self.dt / self.dt_sub)  # dimensionless (number of sub-steps)


        # PID params (students tune)
        self.Kp = 0.0  # dimensionless (proportional gain)
        self.Ki = 0.0  # 1/s (integral gain)
        self.Kd = 0.0  # s (derivative gain)

        # Hidden interpolators (scaled)
        self.drag_interp = drag_interp
        self.thrust_main_interp = thrust_main_interp
        self.thrust_rot_interp = thrust_rot_interp

        self.reset()

    def reset(self, seed=1, options=None):
        if seed is not None:
            np.random.seed(seed)
        perturbation_r = 0# np.random.uniform(-1000, 1000)  # m (radial perturbation)

        self.current_r = self.initial_r #+ perturbation_r  # m (current radial distance)
        self.current_phi = self.initial_phi  # rad (current orbital angle)
        self.current_v_r = 0  # m/s (current radial velocity)
        self.current_v_theta = np.sqrt(self.mu / self.current_r) + 0  # m/s (current tangential velocity)
        self.current_psi = self.initial_psi + 0  # rad (current attitude angle)
        self.current_omega = self.initial_omega + 0  # rad/s (current angular velocity)

        # Ensure bounded
        self.current_phi = self.current_phi % (2 * np.pi)
        self.current_psi = self.current_psi % (2 * np.pi)

        self.fuel_mass = self.initial_fuel_mass  # kg (current fuel mass)
        self.time = 0  # s (current time)
        self.integral_error = 0.0  # rad s (integral error for PID)
        self.prev_error = 0.0  # rad (previous error for PID)
        return self._get_obs(), {}

    # Rotation inner control loop functions:
    def control_signal(self, error, integral_error, derivative, current_psi, current_omega, target_alpha,
                              dt_sub):
        """
        Default PID-based rotational thruster valve computation.
        Override this method in a subclass for custom controllers (e.g. MPC, nonlinear, RL-based, etc.).
        You have access to self (gains, interpolators, arm, I, isp, g0, etc.) and all passed state variables.
        """
        pid_output = self.Kp * error + self.Ki * integral_error + self.Kd * derivative
        return pid_output

    def pid_attitude_control(self, current_psi, current_omega, current_v_r, current_v_theta, current_phi, target_alpha, dt_sub, N_sub):
        integral_error = 0.0
        prev_error = 0.0
        total_fuel_use_rot = 0.0

        for _ in range(N_sub):
            # Compute alpha
            v_x = current_v_r * np.cos(current_phi) - current_v_theta * np.sin(current_phi)
            v_y = current_v_r * np.sin(current_phi) + current_v_theta * np.cos(current_phi)
            angle_v = np.arctan2(v_y, v_x)
            alpha = (current_psi - angle_v + np.pi) % (2 * np.pi) - np.pi  # clamped to [-π, π]
            error = target_alpha - alpha
            integral_error += error * dt_sub
            derivative = (error - prev_error) / dt_sub

            # Get rotational valve commands, clipped to [0, 1]
            signal = self.control_signal(
                error, integral_error, derivative, current_psi, current_omega, target_alpha, dt_sub
            )
            valve_rot_pos = np.clip(signal, 0, 1) if signal > 0 else 0
            valve_rot_neg = np.clip(-signal, 0, 1) if signal < 0 else 0

            prev_error = error

            thrust_rot_pos = self.thrust_rot_interp(valve_rot_pos)
            thrust_rot_neg = self.thrust_rot_interp(valve_rot_neg)
            net_torque = (thrust_rot_pos - thrust_rot_neg) * self.arm
            delta_omega = net_torque * dt_sub / self.I
            current_omega += delta_omega
            current_psi += current_omega * dt_sub
            current_psi %= (2 * np.pi)

            fuel_use_rot = ((thrust_rot_pos + thrust_rot_neg) * dt_sub) / (self.isp * self.g0)
            total_fuel_use_rot += fuel_use_rot

        return current_psi, current_omega, total_fuel_use_rot

    # Dynamic outer level step:
    def step(self, action):
        terminated = False
        truncated = False

        des_delta_v_tang_norm, target_alpha_norm = action
        des_delta_v_tang = des_delta_v_tang_norm * self.delta_v_bounds  # m/s (desired tangential delta-v)
        target_alpha = target_alpha_norm * self.alpha_bounds  # rad (target alpha angle)

        total_fuel_use = 0.0  # kg (total fuel used in step)
        current_mass = self.dry_mass + self.fuel_mass  # kg (current total mass)

        # Low-level PID for attitude over sub-steps
        for _ in range(self.N_sub):
            # call pid_attitude_control function
            self.current_psi, self.current_omega, fuel_use_rot = self.pid_attitude_control (self.current_psi, self.current_omega, self.current_v_r, self.current_v_theta, self.current_phi, target_alpha, self.dt_sub, 1)


        # Drain fuel from main tank
        total_fuel_use += fuel_use_rot

        # Compute final alpha after PID
        v_x = self.current_v_r * np.cos(self.current_phi) - self.current_v_theta * np.sin(self.current_phi)
        v_y = self.current_v_r * np.sin(self.current_phi) + self.current_v_theta * np.cos(self.current_phi)
        angle_v = np.arctan2(v_y, v_x)
        alpha = (self.current_psi - angle_v + np.pi) % (2 * np.pi) - np.pi  # Clamped to [-π, π]


        # Apply in tangential direction (assume aligned after PID)
        #self.current_v_theta += actual_delta_v  # m/s (update v_theta)

        # Apply main thrust for des_delta_v_tang (aligned prograde)
        delta_v = des_delta_v_tang if des_delta_v_tang > 0 else 0  # m/s (positive delta-v only)
        valve_main = np.clip(delta_v * current_mass / (400.0 * self.dt), 0, 1)
        thrust_main = self.thrust_main_interp(valve_main)  # N (main thrust)
        actual_delta_v = thrust_main * self.dt / current_mass  # m/s (actual delta-v magnitude)

        # Project delta_v based on alpha (cos for tangential efficiency, sin for radial waste)
        delta_v_theta = actual_delta_v * np.cos(alpha)  # Tangential component
        delta_v_r = actual_delta_v * np.sin(alpha)  # Radial component (if misaligned)

        # Update velocities
        self.current_v_theta += delta_v_theta
        self.current_v_r += delta_v_r
        fuel_use_main = (thrust_main * self.dt) / (self.isp * self.g0)  # kg (main fuel use)
        total_fuel_use += fuel_use_main  # kg (accumulate)
        self.fuel_mass -= total_fuel_use  # kg (update fuel mass)

        #terminated = self.fuel_mass < 0 or self.current_v_theta < 0 # bool (termination check)
        if self.current_v_theta < 0:
            print("Warning: Orbit reversed or decayed severely!")
            reward = -1000  # dimensionless (penalty)
            terminated = True
            return self._get_obs(), reward, terminated, False, {}

        if self.fuel_mass < 0:
            print("Warning: Fuel depleted!")
            self.fuel_mass = 0  # kg (clamp to zero)
            reward = -1000  # dimensionless (penalty)
            terminated = True
            return self._get_obs(), reward, terminated, False, {}

        # Dynamics integration (full dt)
        def dynamics(t, y):
            r, phi, v_r, v_theta, psi, omega = y  # unpack state

            phi = phi % (2 * np.pi)
            psi = psi % (2 * np.pi)

            accel_r = -self.mu / r**2 + v_theta**2 / r  # m/s^2 (radial acceleration)
            accel_theta = -(v_r * v_theta) / r  # m/s^2 (tangential acceleration)
            v_mag = np.sqrt(v_r**2 + v_theta**2)  # m/s (velocity magnitude)
            if v_mag > 0:
                hat_v_r = v_r / v_mag  # dimensionless (unit radial velocity)
                hat_v_theta = v_theta / v_mag  # dimensionless (unit tangential velocity)
                v_x_ = v_r * np.cos(phi) - v_theta * np.sin(phi)  # m/s (inertial x-velocity)
                v_y_ = v_r * np.sin(phi) + v_theta * np.cos(phi)  # m/s (inertial y-velocity)
                angle_v = np.arctan2(v_y_, v_x_)  # rad (velocity angle)
                alpha = (psi - angle_v + np.pi) % (2 * np.pi) - np.pi  # rad (alpha)  # clampled to [-π, π]
                drag_factor = self.drag_interp(alpha)  # dimensionless (interpolated drag factor)
                density = self.atm_density_scale * np.exp(-(r - self.R_earth - self.initial_altitude) / self.scale_height)  # kg/m^3 (density)
                drag_mag = 0.5 * self.drag_coeff * drag_factor * self.area_mass_ratio * density * v_mag**2  # m/s^2 (drag magnitude)
                accel_r_drag = -drag_mag * hat_v_r  # m/s^2 (radial drag acceleration)
                accel_theta_drag = -drag_mag * hat_v_theta  # m/s^2 (tangential drag acceleration)
            else:
                accel_r_drag = accel_theta_drag = 0  # m/s^2 (no drag)

            return [v_r, v_theta / r, accel_r + accel_r_drag, accel_theta + accel_theta_drag, 0, 0]  # state derivatives

        sol = solve_ivp(dynamics, [0, self.dt], [self.current_r, self.current_phi, self.current_v_r, self.current_v_theta, self.current_psi, self.current_omega], method='RK45', rtol=1e-8, atol=1e-10)
        self.current_r, self.current_phi, self.current_v_r, self.current_v_theta, self.current_psi, self.current_omega = sol.y[:, -1]  # update state
        self.time += self.dt  # s (update time)

        delta_alt = self.current_r - self.R_earth - self.initial_altitude  # m (altitude deviation)

        # Ensure angles are bounded
        self.current_phi = self.current_phi % (2 * np.pi)
        self.current_psi = self.current_psi % (2 * np.pi)

        alpha = self.current_psi - np.arctan2(v_y, v_x) % (2 * np.pi) - np.pi  # rad (final alpha) clampled to [-π, π]
    #    reward = -np.abs(delta_alt) / 100 - total_fuel_use * 1.5 - np.abs(alpha) * 0.1  # dimensionless (reward)
        reward = self.reward_function()

        # Track current energy in the system)
        self.mechanical_energy = (self.current_v_r**2 + self.current_v_theta**2) / 2 - self.mu / self.current_r
        self.chemical_energy = self.fuel_mass * (self.isp * self.g0)**2 / 2  # Chemical potential (exhaust KE equivalent)
        self.total_energy = self.mechanical_energy + self.chemical_energy
        self.energy = self.total_energy  # The actual tracked energy
        return self._get_obs(), reward, terminated, truncated, {}

    # Define the reward function
    def reward_function(self):
        # Example reward function (to be customized as needed)

        # Compute the current altitude deviation
        delta_alt = self.current_r - self.R_earth - self.initial_altitude  # m (altitude deviation)

        # Compute total fuel used
        total_fuel_use = self.initial_fuel_mass - self.fuel_mass

        # Compute alpha:
        v_x = self.current_v_r * np.cos(self.current_phi) - self.current_v_theta * np.sin(self.current_phi)
        v_y = self.current_v_r * np.sin(self.current_phi) + self.current_v_theta * np.cos(self.current_phi)
        alpha = self.current_psi - np.arctan2(v_y, v_x) % (2 * np.pi) - np.pi  # rad (final alpha) clampled to [-π, π]

        #return -np.abs(delta_alt) / 100 - total_fuel_use * 1.5 - np.abs(alpha) * 0.1
        return -np.abs(delta_alt) / 1000 - total_fuel_use/100 - np.abs(alpha)

    # Observation function (for training RL agent):
    def _get_obs(self):
        delta_alt = self.current_r - self.R_earth - self.initial_altitude  # m (altitude deviation)
        delta_v_theta = self.current_v_theta - np.sqrt(self.mu / self.current_r)  # m/s (tangential velocity deviation)
        v_x = self.current_v_r * np.cos(self.current_phi) - self.current_v_theta * np.sin(self.current_phi)  # m/s (inertial x-velocity)
        v_y = self.current_v_r * np.sin(self.current_phi) + self.current_v_theta * np.cos(self.current_phi)  # m/s (inertial y-velocity)
        angle_v = np.arctan2(v_y, v_x)  # rad (velocity angle)
       # alpha = self.current_psi - angle_v  # rad (alpha)
        alpha = (self.current_psi - angle_v + np.pi) % (2 * np.pi) - np.pi  # rad (alpha)
        omega_dev = self.current_omega - (self.current_v_theta / self.current_r)  # rad/s (omega deviation)
        fuel_norm = 2 * (self.fuel_mass / self.initial_fuel_mass) - 1  # dimensionless (normalized fuel [-1,1])
        return np.array([delta_alt / self.altitude_bounds, self.current_v_r / self.v_r_bounds, delta_v_theta / self.v_theta_bounds,
                         np.sin(alpha), np.cos(alpha), omega_dev / self.omega_dev_bounds, fuel_norm], dtype=np.float32)  # normalized observation


# Simulation function
def run_simulation(env, policy, num_steps=25920, plot_results=True):
    """
    Run a simulation episode using the provided policy.

    Parameters:
    - env: The environment instance.
    - policy: A callable that takes obs (and optionally step) and returns action.
    - num_steps: Maximum number of steps to simulate.
    - plot_results: If True, plot the results after simulation.

    Returns:
    - data: Dictionary containing collected simulation data.
    """
    obs, _ = env.reset()
    data = {
        'times': [],  # hours
        'delta_alts': [],  # km
        'alphas': [],  # rad
        'fuels': [],  # kg
        'delta_v_thetas': [],  # m/s
        'des_delta_v': [],  # m/s
        'target_alphas': [],  # rad
        'v_thetas': [],  # m/s
        'xs': [],  # km
        'ys': [],  # km
        'energy': [],  # J
        'mechanical energy': [],  # J
        'chemical energy': []  # J
    }

    terminated = False
    truncated = False
    step = 0

    num_params = len(inspect.signature(policy).parameters)

    while step < num_steps and not (terminated or truncated):
        #print(f'terminated: {terminated}, truncated: {truncated}, step: {step}')
        if num_params == 1:
            action = policy(obs)
        elif num_params == 2:
            action = policy(obs, step)
        else:
            raise ValueError("Policy must take 1 (obs) or 2 (obs, step) parameters.")

        obs, reward, terminated, truncated, info = env.step(action)


        # Collect data
        data['times'].append(env.time / 3600)  # hours
        delta_alt = (env.current_r - env.R_earth) - env.initial_altitude
        data['delta_alts'].append(delta_alt / 1000)  # km
        data['energy'].append(env.energy)  # J
        data['mechanical energy'].append(env.mechanical_energy)  # J
        data['chemical energy'].append(env.chemical_energy)  # J

        # Alpha
        v_x = env.current_v_r * np.cos(env.current_phi) - env.current_v_theta * np.sin(env.current_phi)  # m/s
        v_y = env.current_v_r * np.sin(env.current_phi) + env.current_v_theta * np.cos(env.current_phi)  # m/s
        angle_v = np.arctan2(v_y, v_x)  # rad
       # alpha = env.current_psi - angle_v  # rad
        #alpha = env.current_psi - angle_v % (2 * np.pi) - np.pi   # rad
        alpha = (env.current_psi - angle_v + np.pi) % (2 * np.pi) - np.pi  # rad
        data['alphas'].append(alpha)  # rad

        data['fuels'].append(env.fuel_mass)  # kg
        data['delta_v_thetas'].append(env.current_v_theta - np.sqrt(env.mu / env.current_r))  # m/s
        data['v_thetas'].append(env.current_v_theta)  # m/s
        data['xs'].append(env.current_r * np.cos(env.current_phi) / 1000)  # km
        data['ys'].append(env.current_r * np.sin(env.current_phi) / 1000)  # km

        # Actions
        des_delta_v_tang_norm, target_alpha_norm = action
        des_delta_v_tang = des_delta_v_tang_norm * env.delta_v_bounds  # m/s
        target_alpha = target_alpha_norm * env.alpha_bounds  # rad
        data['des_delta_v'].append(des_delta_v_tang)  # m/s
        data['target_alphas'].append(target_alpha)  # rad
        #print(f'terminated: {terminated}, truncated: {truncated}, step: {step}')
        if terminated or truncated:
            print(f'terminated: {terminated}, truncated: {truncated}, step: {step}')
            print("Episode done after", step + 1, "steps.")
            break

        step += 1

    # Calculate reward
    print(f'Days in orbit: {data['times'][-1]/24} days')
    print(f'Fuel mass remaining: {data['fuels'][-1]} kg')
    score = (2 * data['times'][-1]/24)**2 + data['fuels'][-1]**2
    print(f'Score = {score }')
    if plot_results:
        plot_simulation(data)

    return data

# Plotting function
def plot_simulation(data):
    """
    Plot the simulation results for closed-loop.

    Parameters:
    - data: Dictionary containing simulation data.
    """
    plt.figure(figsize=(12, 15))

    # Altitude deviation vs. time
    plt.subplot(6, 1, 1)
    plt.plot(data['times'], data['delta_alts'], label='Altitude Deviation (km)')
    plt.axhline(0, color='r', linestyle='--', label='Target')
    plt.xlabel('Time (hours)')
    plt.ylabel('Altitude Deviation (km)')
    plt.title('Orbit Maintenance')
    plt.legend()
    plt.grid(True)

    # Orientation deviation alpha vs. time
    plt.subplot(6, 1, 2)
    plt.plot(data['times'], data['alphas'], label='Alpha (rad)')
    plt.axhline(0, color='r', linestyle='--', label='Optimal Alignment')
    plt.xlabel('Time (hours)')
    plt.ylabel('Alpha (rad)')
    plt.title('Orientation Deviation')
    plt.legend()
    plt.grid(True)

    # Fuel reserves vs. time
    plt.subplot(6, 1, 3)
    plt.plot(data['times'], data['fuels'], label='Fuel Mass (kg)')
    plt.axhline(0, color='r', linestyle='--', label='Depleted')
    plt.xlabel('Time (hours)')
    plt.ylabel('Fuel Mass (kg)')
    plt.title('Fuel Reserves')
    plt.legend()
    plt.grid(True)

    # Applied actions vs. time
    ax = plt.subplot(6, 1, 4)
    # Left y-axis: Desired Delta-v (small values)
    ax.plot(data['times'], data['des_delta_v'],
            label='Desired Δv Tang (m/s)', color='tab:blue')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Desired Δv Tang (m/s)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    # Right y-axis: Target Alpha
    ax2 = ax.twinx()
    ax2.plot(data['times'], data['target_alphas'],
             label='Target Alpha (rad)', color='tab:orange')
    ax2.set_ylabel('Target Alpha (rad)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax.set_title('High-Level Actions')
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax.grid(True)

    # Phase space: delta_alt vs. delta_v_theta
    plt.subplot(6, 1, 5)
    plt.plot(data['delta_alts'], data['delta_v_thetas'], label='Trajectory')
    plt.xlabel('Altitude Deviation (km)')
    plt.ylabel('Tangential Velocity Deviation (m/s)')
    plt.title('Phase Space (Altitude vs. Tangential Vel Dev)')
    plt.legend()
    plt.grid(True)

    # Energy
    plt.subplot(6, 1, 6)
    plt.plot(data['times'], data['energy'], label='Total energy')
    plt.plot(data['times'], data['mechanical energy'], label='Mechanical energy')
    plt.plot(data['times'], data['chemical energy'], label='Chemical energy')
    plt.xlabel('Time (hours)')
    plt.ylabel('Energy (J)')
    plt.title('Mechanical and chemical energy of the station (kinetic plus potential)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

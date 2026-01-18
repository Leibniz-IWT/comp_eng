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
        self.initial_fuel_mass = 200.0  # kg (initial fuel mass, hydrazine)
        self.isp = 220.0  # s (specific impulse)
        self.g0 = 9.81  # m/s^2 (standard gravity)
        self.arm = 2.0  # m (lever arm for rotational thrusters)
        self.I = 1000.0  # kg m^2 (moment of inertia)

        self.drag_coeff = 2.2  # dimensionless (drag coefficient)
        self.area_mass_ratio = 0.01  # m^2/kg (area-to-mass ratio)
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
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Norm des_delta_v, target_alpha

        self.dt = 600  # s (main time step)
        self.dt_sub = 10  # s (sub-step for attitude control)
        self.N_sub = int(self.dt / self.dt_sub)  # dimensionless (number of sub-steps)

        # PID params (students tune)
        self.Kp = 0.5  # dimensionless (proportional gain)
        self.Ki = 0.1  # 1/s (integral gain)
        self.Kd = 0.2  # s (derivative gain)

        # Hidden interpolators (scaled)
        alphas = np.linspace(-np.pi, np.pi, 200)  # rad (alpha values for interpolation)
        real_drag = 1 + np.abs(np.sin(alphas))  # dimensionless (real drag factor)
        noisy_drag = real_drag + np.random.normal(0, 0.05, alphas.shape)  # dimensionless (noisy drag factor)
        self.drag_interp = interp1d(alphas, noisy_drag, 'linear', fill_value='extrapolate')

        valves = np.linspace(0, 1, 100)  # dimensionless (valve fractions 0-1)
        real_thrust_main = 0.1 * valves**1.2  # N (real main thrust)
        noisy_thrust_main = real_thrust_main + np.random.normal(0, 0.001, valves.shape)  # N (noisy main thrust)
        self.thrust_main_interp = interp1d(valves, noisy_thrust_main, 'linear', bounds_error=False, fill_value=(0, noisy_thrust_main[-1]))

        real_thrust_rot = 0.005 * valves  # N (real rotational thrust)
        noisy_thrust_rot = real_thrust_rot + np.random.normal(0, 0.00005, valves.shape)  # N (noisy rotational thrust)
        self.thrust_rot_interp = interp1d(valves, noisy_thrust_rot, 'linear', bounds_error=False, fill_value=(0, noisy_thrust_rot[-1]))

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        perturbation_r = np.random.uniform(-1000, 1000)  # m (radial perturbation)
        self.current_r = self.initial_r + perturbation_r  # m (current radial distance)
        self.current_phi = self.initial_phi  # rad (current orbital angle)
        self.current_v_r = np.random.uniform(-1, 1)  # m/s (current radial velocity)
        self.current_v_theta = np.sqrt(self.mu / self.current_r) + np.random.uniform(-1, 1)  # m/s (current tangential velocity)
        self.current_psi = self.initial_psi + np.random.uniform(-0.1, 0.1)  # rad (current attitude angle)
        self.current_omega = self.initial_omega + np.random.uniform(-0.001, 0.001)  # rad/s (current angular velocity)
        self.fuel_mass = self.initial_fuel_mass  # kg (current fuel mass)
        self.time = 0  # s (current time)
        self.integral_error = 0.0  # rad s (integral error for PID)
        self.prev_error = 0.0  # rad (previous error for PID)
        return self._get_obs(), {}

    def step(self, action):

        des_delta_v_tang_norm, target_alpha_norm = action
        des_delta_v_tang = des_delta_v_tang_norm * self.delta_v_bounds  # m/s (desired tangential delta-v)
        target_alpha = target_alpha_norm * self.alpha_bounds  # rad (target alpha angle)

        total_fuel_use = 0.0  # kg (total fuel used in step)
        current_mass = self.dry_mass + self.fuel_mass  # kg (current total mass)

        # Low-level PID for attitude over sub-steps
        for _ in range(self.N_sub):
            # Compute alpha
            v_x = self.current_v_r * np.cos(self.current_phi) - self.current_v_theta * np.sin(self.current_phi)  # m/s (inertial x-velocity)
            v_y = self.current_v_r * np.sin(self.current_phi) + self.current_v_theta * np.cos(self.current_phi)  # m/s (inertial y-velocity)
            angle_v = np.arctan2(v_y, v_x)  # rad (velocity direction angle)
            alpha = self.current_psi - angle_v  # rad (current alpha)
            error = target_alpha - alpha  # rad (error)
            self.integral_error += error * self.dt_sub  # rad s (update integral)
            derivative = (error - self.prev_error) / self.dt_sub  # rad/s (derivative)
            pid_output = self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative  # dimensionless (PID output)
            self.prev_error = error  # rad (update previous error)

            # Valve for rot (clip to [0,1])
            valve_rot_pos = np.clip(pid_output, 0, 1) if pid_output > 0 else 0  # dimensionless (positive rotational valve)
            valve_rot_neg = np.clip(-pid_output, 0, 1) if pid_output < 0 else 0  # dimensionless (negative rotational valve)

            thrust_rot_pos = self.thrust_rot_interp(valve_rot_pos)  # N (positive rotational thrust)
            thrust_rot_neg = self.thrust_rot_interp(valve_rot_neg)  # N (negative rotational thrust)
            net_torque = (thrust_rot_pos - thrust_rot_neg) * self.arm  # N m (net torque)
            delta_omega = net_torque * self.dt_sub / self.I  # rad/s (delta angular velocity)
            self.current_omega += delta_omega  # rad/s (update omega)
            self.current_psi += self.current_omega * self.dt_sub  # rad (update psi, approximate)

            fuel_use_rot = ((thrust_rot_pos + thrust_rot_neg) * self.dt_sub) / (self.isp * self.g0)  # kg (rotational fuel use)
            total_fuel_use += fuel_use_rot  # kg (accumulate)

        # Apply main thrust for des_delta_v_tang (aligned prograde)
        delta_v = des_delta_v_tang if des_delta_v_tang > 0 else 0  # m/s (positive delta-v only)
        valve_main = np.clip(delta_v * current_mass / (0.1 * self.dt), 0, 1)  # dimensionless (main valve, approx from max thrust)
        thrust_main = self.thrust_main_interp(valve_main)  # N (main thrust)
        actual_delta_v = thrust_main * self.dt / current_mass  # m/s (actual delta-v)

        # Apply in tangential direction (assume aligned after PID)
        self.current_v_theta += actual_delta_v  # m/s (update v_theta)

        fuel_use_main = (thrust_main * self.dt) / (self.isp * self.g0)  # kg (main fuel use)
        total_fuel_use += fuel_use_main  # kg (accumulate)
        self.fuel_mass -= total_fuel_use  # kg (update fuel mass)

        #terminated = self.fuel_mass < 0 or self.current_v_theta < 0  # bool (termination check)
        if self.current_v_theta < 0:
            print("Warning: Orbit reversed or decayed severely!")
            reward = -1000  # dimensionless (penalty)
            return self._get_obs(), reward, terminated, False, {}
        if self.fuel_mass < 0:
            print("Warning: Fuel depleted!")
            self.fuel_mass = 0  # kg (clamp to zero)
            reward = -1000  # dimensionless (penalty)
            return self._get_obs(), reward, terminated, False, {}

        # Dynamics integration (full dt)
        def dynamics(t, y):
            r, phi, v_r, v_theta, psi, omega = y  # unpack state
            accel_r = -self.mu / r**2 + v_theta**2 / r  # m/s^2 (radial acceleration)
            accel_theta = -(v_r * v_theta) / r  # m/s^2 (tangential acceleration)
            v_mag = np.sqrt(v_r**2 + v_theta**2)  # m/s (velocity magnitude)
            if v_mag > 0:
                hat_v_r = v_r / v_mag  # dimensionless (unit radial velocity)
                hat_v_theta = v_theta / v_mag  # dimensionless (unit tangential velocity)
                v_x_ = v_r * np.cos(phi) - v_theta * np.sin(phi)  # m/s (inertial x-velocity)
                v_y_ = v_r * np.sin(phi) + v_theta * np.cos(phi)  # m/s (inertial y-velocity)
                angle_v = np.arctan2(v_y_, v_x_)  # rad (velocity angle)
                alpha = psi - angle_v  # rad (alpha)
                drag_factor = self.drag_interp(alpha)  # dimensionless (interpolated drag factor)
                density = self.atm_density_scale * np.exp(-(r - self.R_earth - self.initial_altitude) / self.scale_height)  # kg/m^3 (density)
                drag_mag = 0.5 * self.drag_coeff * drag_factor * self.area_mass_ratio * density * v_mag**2  # m/s^2 (drag magnitude)
                accel_r_drag = -drag_mag * hat_v_r  # m/s^2 (radial drag acceleration)
                accel_theta_drag = -drag_mag * hat_v_theta  # m/s^2 (tangential drag acceleration)
            else:
                accel_r_drag = accel_theta_drag = 0  # m/s^2 (no drag)
            return [v_r, v_theta / r, accel_r + accel_r_drag, accel_theta + accel_theta_drag, omega, 0]  # state derivatives

        sol = solve_ivp(dynamics, [0, self.dt], [self.current_r, self.current_phi, self.current_v_r, self.current_v_theta, self.current_psi, self.current_omega], method='RK45', rtol=1e-8)
        self.current_r, self.current_phi, self.current_v_r, self.current_v_theta, self.current_psi, self.current_omega = sol.y[:, -1]  # update state
        self.time += self.dt  # s (update time)

        delta_alt = self.current_r - self.R_earth - self.initial_altitude  # m (altitude deviation)
        alpha = self.current_psi - np.arctan2(v_y, v_x)  # rad (final alpha)
        reward = -np.abs(delta_alt) / 100 - total_fuel_use * 1.5 - np.abs(alpha) * 0.1  # dimensionless (reward)

        #terminated = terminated or (np.abs(delta_alt) > 10000)  # bool (add altitude termination)
      #  terminated = terminated or (np.abs(delta_alt) > 10000)  # bool (add altitude termination)
      #  truncated = self.time > 1000 * self.dt  # bool (time truncation)
        print(f'terminated = {terminated}')
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        delta_alt = self.current_r - self.R_earth - self.initial_altitude  # m (altitude deviation)
        delta_v_theta = self.current_v_theta - np.sqrt(self.mu / self.current_r)  # m/s (tangential velocity deviation)
        v_x = self.current_v_r * np.cos(self.current_phi) - self.current_v_theta * np.sin(self.current_phi)  # m/s (inertial x-velocity)
        v_y = self.current_v_r * np.sin(self.current_phi) + self.current_v_theta * np.cos(self.current_phi)  # m/s (inertial y-velocity)
        angle_v = np.arctan2(v_y, v_x)  # rad (velocity angle)
        alpha = self.current_psi - angle_v  # rad (alpha)
        omega_dev = self.current_omega - (self.current_v_theta / self.current_r)  # rad/s (omega deviation)
        fuel_norm = 2 * (self.fuel_mass / self.initial_fuel_mass) - 1  # dimensionless (normalized fuel [-1,1])
        return np.array([delta_alt / self.altitude_bounds, self.current_v_r / self.v_r_bounds, delta_v_theta / self.v_theta_bounds,
                         np.sin(alpha), np.cos(alpha), omega_dev / self.omega_dev_bounds, fuel_norm], dtype=np.float32)  # normalized observation

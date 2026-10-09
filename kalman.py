import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. SIMULATION CONFIGURATION (Units in Centimeters)
# =============================================================================
dt = 0.01          # Sampling period (100 Hz, matching microcontroller loop)
t_end = 5.0        # Total simulation time in seconds
t = np.arange(0, t_end, dt)

# Intentional Motion Parameters (X-Direction)
v_x = 20.0         # Intentional hand speed along X-axis (20 cm/s -> 100 cm total stroke)
x_ideal = v_x * t  # Smooth forward movement in cm
y_ideal = np.zeros_like(t) # Ideal Y-position (0 cm baseline)

# Tremor & Noise Parameters (Y-Direction)
f_tremor = 6.0     # Tremor frequency in Hz (Parkinsonian/Essential tremor: 4–8 Hz)
amp_tremor = 2.0   # Tremor amplitude in cm (2 cm peak / 4 cm peak-to-peak)
noise_std = 0.4    # Sensor noise standard deviation in cm

# Generate Simulated Raw Signal
tremor_y = amp_tremor * np.sin(2 * np.pi * f_tremor * t)
sensor_noise_y = np.random.normal(0, noise_std, len(t))
y_raw = y_ideal + tremor_y + sensor_noise_y

# =============================================================================
# 2. KALMAN FILTER SETUP (Units in cm and cm/s)
# =============================================================================
# State vector x = [position_y (cm), velocity_y (cm/s)]^T
A = np.array([[1.0, dt],
              [0.0, 1.0]])  # State transition matrix
H = np.array([[1.0, 0.0]])  # Measurement matrix (position only)

# Process Noise Covariance Q (Trust in kinematic model vs sudden acceleration changes)
q_var = 15.0  # Process noise magnitude in cm^2/s^4
Q = np.array([[0.25 * dt**4, 0.5 * dt**3],
              [0.5 * dt**3, dt**2]]) * q_var

# Measurement Noise Covariance R (Measurement noise variance in cm^2)
# Increasing R penalizes high-frequency fluctuations like tremors
R = np.array([[1.0]])

# Initial state and error covariance
x_hat = np.array([[0.0], [0.0]])
P = np.eye(2) * 1.0

# Array to store filtered Y position
y_corrected = np.zeros_like(t)

# =============================================================================
# 3. FILTER EXECUTION LOOP
# =============================================================================
for i in range(len(t)):
    z = np.array([[y_raw[i]]]) # Measurement in cm

    # Predict Step
    x_hat = A @ x_hat
    P = A @ P @ A.T + Q

    # Update Step
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R) # Kalman Gain
    x_hat = x_hat + K @ (z - H @ x_hat)
    P = (np.eye(2) - K @ H) @ P

    # Save estimated position
    y_corrected[i] = x_hat[0, 0]

# =============================================================================
# 4. PLOTTING AND VISUALIZATION
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Y-Position over Time
ax1.plot(t, y_raw, label='Raw Y-Position (Intentional + Tremor)', color='red', alpha=0.5, linewidth=1)
ax1.plot(t, y_corrected, label='Corrected Y-Position (Kalman Filter)', color='blue', linewidth=2)
ax1.plot(t, y_ideal, label='Ideal Target Y-Path', color='black', linestyle='--', linewidth=1.5)
ax1.set_title(f'Y-Axis Tremor Suppression Over Time ({f_tremor} Hz Tremor)')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Y Position (cm)')
ax1.legend(loc='upper right')
ax1.grid(True)

# Plot 2: 2D Spatial Trajectory (X vs Y with True Physical 1:1 Scale)
ax2.plot(x_ideal, y_raw, label='Uncorrected Hand Path', color='red', alpha=0.4)
ax2.plot(x_ideal, y_corrected, label='Corrected Hand Path', color='blue', linewidth=2)
ax2.plot(x_ideal, y_ideal, label='Ideal Straight Path', color='black', linestyle='--', linewidth=1.5)
ax2.set_title('2D Spatial Hand Path Trajectory (True 1:1 Scale in cm)')
ax2.set_xlabel('X Position (cm)')
ax2.set_ylabel('Y Position (cm)')
ax2.set_aspect('equal', adjustable='datalim')  # Enforces true 1:1 physical ratio
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()
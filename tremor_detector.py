import sys
import time
import serial
import numpy as np
from scipy.signal import butter, filtfilt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# --- CONFIGURATION ---
PORT = "COM6"           # Update to match your Arduino port
BAUD = 115200
FS = 100                # Sampling rate (Hz)
WINDOW_SEC = 1.5        # Window size in seconds
BUFFER_LEN = int(FS * WINDOW_SEC)
PROCESS_EVERY = 5       # Process signal every N incoming samples

# Detection Thresholds
TREMOR_BAND = (3.0, 12.0)    # Tremor frequency band in Hz
AMP_THRESHOLD = 0.30         # Minimum spectral power density to classify as tremor


# --- 1D KALMAN FILTER ---
class KalmanFilter1D:
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1, estimation_error=1.0, initial_val=0.0):
        self.q = process_noise
        self.r = measurement_noise
        self.p = estimation_error
        self.x = initial_val

    def filter_array(self, data):
        filtered = np.zeros_like(data)
        for i, z in enumerate(data):
            self.p += self.q
            k = self.p / (self.p + self.r)
            self.x += k * (z - self.x)
            self.p *= (1.0 - k)
            filtered[i] = self.x
        return filtered


# --- SERIAL CONNECTION ---
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2.0)
    ser.reset_input_buffer()
except Exception as e:
    print(f"Error opening serial port {PORT}: {e}")
    sys.exit(1)


# --- DATA BUFFERS & FILTERS ---
data_buffer = np.zeros((BUFFER_LEN, 6))  # [ax, ay, az, gx, gy, gz]
sample_count = 0

# 4th-order Butterworth Bandpass (3 Hz to 12 Hz)
nyq = 0.5 * FS
b_band, a_band = butter(4, [TREMOR_BAND[0] / nyq, TREMOR_BAND[1] / nyq], btype='band')
kalman = KalmanFilter1D(process_noise=1e-3, measurement_noise=0.05)


# --- PYQTGRAPH GUI SETUP ---
app = pg.mkQApp("Tremor Analysis System")
win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Tremor Detection System")
win.resize(1100, 750)

# Status Header Text
status_label = win.addLabel("STATUS: INITIALIZING...", row=0, col=0, colspan=2, size='16pt', bold=True)
win.nextRow()

# Time-Domain Plot (Linear Acceleration)
plot_accel = win.addPlot(title="Linear Acceleration Magnitude (Bandpass + Kalman Filtered)")
curve_accel = plot_accel.plot(pen=pg.mkPen('c', width=1.5))
plot_accel.setLabel('bottom', 'Samples')
plot_accel.setLabel('left', 'Linear Accel (m/s²)')
plot_accel.showGrid(x=True, y=True)

# Frequency-Domain Plot (FFT)
plot_fft = win.addPlot(title="Frequency Spectrum (FFT)")
curve_fft = plot_fft.plot(pen=pg.mkPen('y', width=1.5))
peak_marker = plot_fft.plot(pen=None, symbol='o', symbolBrush='r', symbolSize=10)
plot_fft.setLabel('bottom', 'Frequency (Hz)')
plot_fft.setLabel('left', 'Amplitude')
plot_fft.setXRange(0, 20)
plot_fft.showGrid(x=True, y=True)


# --- SIGNAL PROCESSING & ANALYTICS ---
def analyze_data(buffer):
    # 1. Compute total acceleration magnitude
    accel_mag = np.sqrt(np.sum(buffer[:, 0:3]**2, axis=1))

    # 2. Subtract gravity (DC offset) to isolate dynamic linear acceleration
    linear_accel = accel_mag - np.mean(accel_mag)

    # 3. Apply 3-12 Hz Bandpass Filter
    filtered_accel = filtfilt(b_band, a_band, linear_accel)

    # 4. Apply 1D Kalman Filter
    kalman_accel = kalman.filter_array(filtered_accel)

    # 5. Fast Fourier Transform (FFT)
    n = len(kalman_accel)
    fft_vals = np.fft.rfft(kalman_accel)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    amps = np.abs(fft_vals) / n

    # 6. Extract Peak Frequency within Tremor Band (3 - 12 Hz)
    band_mask = (freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])
    
    if np.any(band_mask):
        band_freqs = freqs[band_mask]
        band_amps = amps[band_mask]
        peak_idx = np.argmax(band_amps)
        peak_freq = band_freqs[peak_idx]
        peak_amp = band_amps[peak_idx]
    else:
        peak_freq, peak_amp = 0.0, 0.0

    # 7. Classification Logic
    is_tremor = peak_amp >= AMP_THRESHOLD

    # Update Visuals
    curve_accel.setData(kalman_accel)
    curve_fft.setData(freqs, amps)
    peak_marker.setData([peak_freq], [peak_amp])

    if is_tremor:
        status_label.setText(
            f"<span style='color: #FF3333;'>STATUS: TREMOR DETECTED | Frequency: {peak_freq:.1f} Hz | Amplitude: {peak_amp:.2f}</span>"
        )
    else:
        status_label.setText(
            f"<span style='color: #33FF33;'>STATUS: NO TREMOR | Peak Frequency: {peak_freq:.1f} Hz | Amplitude: {peak_amp:.2f}</span>"
        )


# --- MAIN LOOP UPDATE ---
def update():
    global data_buffer, sample_count

    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(",")

            if len(parts) == 6:
                try:
                    vals = [float(p) for p in parts]
                except ValueError:
                    continue

                # Shift buffer and insert new sample
                data_buffer = np.roll(data_buffer, -1, axis=0)
                data_buffer[-1, :] = vals
                sample_count += 1

                # Periodically purge serial backlog to avoid UI latency
                if ser.in_waiting > 200:
                    ser.reset_input_buffer()

                if sample_count % PROCESS_EVERY == 0:
                    analyze_data(data_buffer)

    except Exception as e:
        print(f"Update error: {e}")


# Timer Setup
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)  # 5 ms poll interval

# Clean Exit Handler
def cleanup():
    ser.close()
    print("Serial port closed cleanly.")

app.aboutToQuit.connect(cleanup)

if __name__ == '__main__':
    pg.exec()
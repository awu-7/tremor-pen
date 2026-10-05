import serial
import numpy as np
from scipy.signal import butter, filtfilt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import csv
from datetime import datetime
import os
import time

# Setup
PORT = "COM6"
BAUD = 115200
Fs = 200             # sampling rate from Arduino sketch delay(5) -> 200 Hz
BUFFER_LEN = Fs * 2  # 2-second sliding window
PROCESS_EVERY = 10   # process/plot every N samples

ser = serial.Serial(PORT, BAUD, timeout=3)
time.sleep(2)  # let Arduino reset after port opens
ser.reset_input_buffer()

data_buffer = np.zeros((BUFFER_LEN, 6))

# Bandpass filter 
nyq = Fs / 2    # Nyquist rate
low = 6 / nyq   # tremors are around 6-12 hz
high = 12 / nyq
b, a = butter(4, [low, high], btype='band')


# Logging
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"tremor_log_{timestamp_tag}.csv")
results_filename = os.path.join(log_dir, f"tremor_results_{timestamp_tag}.csv")

log_file = open(log_filename, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])

results_file = open(results_filename, "w", newline="")
results_writer = csv.writer(results_file)
results_writer.writerow(["timestamp", "accelPeakFreqHz", "accelPeakAmp", "gyroPeakFreqHz", "gyroPeakAmp"])

print(f"Logging raw data to: {log_filename}")
print(f"Logging tremor results to: {results_filename}")


# Plot setup
app = pg.mkQApp("Tremor Detection")
win = pg.GraphicsLayoutWidget(show=True, title="Live Tremor Detection")
win.resize(1000, 700)

# Accelerometer time domain plot
plot_accel = win.addPlot(title="Bandpass-filtered accel signal (3-15 Hz)")
curve_accel = plot_accel.plot(pen='y')
plot_accel.setLabel('bottom', 'Sample')
plot_accel.setLabel('left', 'Filtered Accel (LSB, 3-15 Hz)')

# Gyroscope time domain plot
plot_gyro = win.addPlot(title="Bandpass-filtered gyro signal (3-15 Hz)")
curve_gyro = plot_gyro.plot(pen='m')
plot_gyro.setLabel('bottom','Sample')
plot_gyro.setLabel('left', 'Filtered Gyro (LSB, 3-15 Hz)')

win.nextRow()

# Accelerometer FFT plot
plot_accel_FFT = win.addPlot(title="Accel FFT") # frequency plot
curve_accel_FFT = plot_accel_FFT.plot(pen='c')
accel_peak_marker = plot_accel_FFT.plot(pen=None, symbol='o', symbolBrush='r', symbolSize=10)
plot_accel_FFT.setLabel('bottom', 'Frequency (Hz)')
plot_accel_FFT.setLabel('left', 'Magnitude')

# Gyroscope FFT plot
plot_gyro_FFT = win.addPlot(title="Gyro FFT") # frequency plot
curve_gyro_FFT = plot_gyro_FFT.plot(pen='c')
gyro_peak_marker = plot_gyro_FFT.plot(pen=None, symbol='o', symbolBrush='r', symbolSize=10)
plot_gyro_FFT.setLabel('bottom', 'Frequency (Hz)')
plot_gyro_FFT.setLabel('left', 'Magnitude')

# Adjust axes
plot_accel.setXRange(0, 400)
plot_accel.enableAutoRange(axis='x', enable=False)
plot_accel.getViewBox().setLimits(minYRange=300)
plot_accel.hideButtons()

plot_accel_FFT.setXRange(0, 30)        # 0-30 Hz based on f_pos[accel_mask] range
plot_accel_FFT.enableAutoRange(axis='x', enable=False)
plot_accel_FFT.getViewBox().setLimits(minYRange=800)
plot_accel_FFT.getViewBox().setLimits(yMin=0)
plot_accel_FFT.hideButtons()

plot_gyro.setXRange(0, 400)
plot_gyro.enableAutoRange(axis='x', enable=False)
plot_gyro.getViewBox().setLimits(minYRange=300)
plot_gyro.hideButtons()

plot_gyro_FFT.setXRange(0, 30)       
plot_gyro_FFT.enableAutoRange(axis='x', enable=False)
plot_gyro_FFT.getViewBox().setLimits(minYRange=800)
plot_gyro_FFT.getViewBox().setLimits(yMin=0)
plot_gyro_FFT.hideButtons()

sample_count = 0 # initialize sample count

# Find peak frequency
def find_peak(filtered_signal, Fs, band_low=3, band_high=15):
    N = len(filtered_signal)
    Y = np.fft.fft(filtered_signal)
    f = np.fft.fftfreq(N, d=1/Fs)
    P = np.abs(Y) / N

    pos_mask = f >= 0       # only positive frequencies
    f_pos = f[pos_mask]     # limit peak detection to 3-15 hz range
    P_pos = P[pos_mask]
    
    band_mask = (f_pos >= 3) & (f_pos <= 15)
    if np.any(band_mask):
        band_amps = P_pos[band_mask]
        band_freqs = f_pos[band_mask]
        peak_idx = np.argmax(band_amps)
        peak_freq = band_freqs[peak_idx]
        peak_amp = band_amps[peak_idx]
    else:
        peak_freq, peak_amp = 0, 0

    return f_pos, P_pos, peak_freq, peak_amp

def analyze_tremor(buffer):
    accel_mag = np.sqrt(np.sum(buffer[:, 0:3] ** 2, axis=1))
    accel_mag = accel_mag - np.mean(accel_mag)
    filtered_accel = filtfilt(b, a, accel_mag)

    f_pos_a, P_pos_a, accel_peak_freq, accel_peak_amp = find_peak(filtered_accel, Fs)

    gyro_mag = np.sqrt(np.sum(buffer[:, 3:6] ** 2, axis=1))
    gyro_mag = gyro_mag - np.mean(gyro_mag)
    filtered_gyro = filtfilt(b, a, gyro_mag)

    f_pos_g, P_pos_g, gyro_peak_freq, gyro_peak_amp = find_peak(filtered_gyro, Fs)

    # log results
    results_writer.writerow([
        datetime.now().strftime("%H:%M:%S.%f"),
        f"{accel_peak_freq:.2f}", f"{accel_peak_amp:.2f}",
        f"{gyro_peak_freq:.2f}", f"{gyro_peak_amp:.2f}",
    ])
    print(f"Accel peak: {accel_peak_freq:.2f} Hz, {accel_peak_amp:.2f} amp  |  "
          f"Gyro peak: {gyro_peak_freq:.2f} Hz, {gyro_peak_amp:.2f} amp")

    # update plots
    curve_accel.setData(filtered_accel)
    curve_gyro.setData(filtered_gyro)

    accel_mask = f_pos_a < 30
    curve_accel_FFT.setData(f_pos_a[accel_mask], P_pos_a[accel_mask])
    accel_peak_marker.setData([accel_peak_freq], [accel_peak_amp])
    plot_accel_FFT.setTitle(f"Accel FFT - Peak: {accel_peak_freq:.1f} Hz, Amp: {accel_peak_amp:.2f}")

    gyro_mask = f_pos_g < 30
    curve_gyro_FFT.setData(f_pos_g[gyro_mask], P_pos_g[gyro_mask])
    gyro_peak_marker.setData([gyro_peak_freq], [gyro_peak_amp])
    plot_gyro_FFT.setTitle(f"Gyro FFT - Peak: {gyro_peak_freq:.1f} Hz, Amp: {gyro_peak_amp:.2f}")


def update():
    global data_buffer, sample_count
    try:
        if ser.in_waiting == 0:
            return

        line = ser.readline().decode('utf-8', errors='ignore').strip()
        parts = line.split(",")

        if len(parts) == 6:
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                print(f"Skipped bad line: [{line}]")
                return

            data_buffer = np.roll(data_buffer, -1, axis=0)
            data_buffer[-1, :] = vals
            sample_count += 1

            # log raw sample
            log_writer.writerow([datetime.now().strftime("%H:%M:%S.%f")] + vals)

            # periodically drop backlog to avoid lag
            if sample_count % 50 == 0:
                ser.reset_input_buffer()

            if sample_count % PROCESS_EVERY == 0:
                analyze_tremor(data_buffer)
        else:
            print(f"Skipped bad line: [{line}]")
    except Exception as e:
        print(f'Error: {e}')


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)  # check for new data every 1ms

if __name__ == '__main__':
    try:
        pg.exec()
    except Exception as e:
        import traceback
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        log_file.close()
        results_file.close()
        ser.close()
        print("Files and serial port closed.")
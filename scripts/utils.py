from scripts.config import os, wavfile, np

def load_signals_from_directory(directory):
    signals = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".wav"):
            fs, signal = wavfile.read(os.path.join(directory, filename))
            signal = signal / np.max(np.abs(signal))
            signals.append(signal)
    return signals, fs


def prepare_dataset(noisy_signals, clean_signals, window_size):
    X, y = [], []
    for noisy, clean in zip(noisy_signals, clean_signals):
        for i in range(window_size, len(noisy)):
            X.append(noisy[i-window_size:i])
            y.append(clean[i])
    return np.array(X), np.array(y)


def calculate_psnr(signal, reference_signal):
    mse = np.mean((signal - reference_signal) ** 2)  
    if mse == 0:  
        return float('inf')  #
    max_val = np.max(reference_signal)  
    psnr = 20 * np.log10(max_val / np.sqrt(mse))  
    return psnr


def calculate_snr(signal, reference_signal):
    signal_power = np.mean(reference_signal ** 2) 
    noise_power = np.mean((signal - reference_signal) ** 2)  
    if noise_power == 0: 
        return float('inf') 
    snr = 10 * np.log10(signal_power / noise_power) 
    return snr

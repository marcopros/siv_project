from scripts.config import np

def lms_filter(noisy_signal, desired_signal, mu, filter_order):
    n_samples = len(noisy_signal)
    weights = np.zeros(filter_order)
    filtered_signal = np.zeros(n_samples)

    for i in range(filter_order, n_samples):
        x = noisy_signal[i - filter_order:i][::-1]
        y = np.dot(weights, x)
        error = desired_signal[i] - y
        weights += 2 * mu * error * x
        filtered_signal[i] = y

    return filtered_signal, weights

def nlms_filter(noisy_signal, desired_signal, mu, filter_order, epsilon = 1e-6):
    n_samples = len(noisy_signal)
    weights = np.zeros(filter_order)
    filtered_signal = np.zeros(n_samples)

    for i in range(filter_order, n_samples):
        x = noisy_signal[i - filter_order:i][::-1]
        y = np.dot(weights, x)
        error = desired_signal[i] - y
        normalization_factor = np.dot(x, x) + epsilon
        weights += (2 * mu * error * x) / normalization_factor
        filtered_signal[i] = y

    return filtered_signal, weights
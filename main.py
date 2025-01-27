from scripts.utils import load_signals_from_directory, prepare_dataset, calculate_psnr, calculate_snr
from scripts.filters import lms_filter, nlms_filter
from scripts.config import os, train_test_split, torch, nn, optim, plt, np, mean_absolute_error, mean_squared_error, r2_score, wavfile, write
from models.model import AdvancedNN

window_size = 32
learning_rate = 0.01
mu = 0.01
filter_order = 32
epochs = 15
batch_size = 64

clean_signals, fs_clean = load_signals_from_directory('data/audios/clean')
noisy_signals, fs_noisy = load_signals_from_directory('data/audios/noise_train')

assert fs_clean == fs_noisy, 'Sampling frequencies do not match!'

X, y = prepare_dataset(noisy_signals, clean_signals, window_size)


X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)


X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

print(f"Full dataset size (X, y): {X.shape}, {y.shape}")
print(f"Training set size (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Validation set size (X_val, y_val): {X_val.shape}, {y_val.shape}")
print(f"Testing set size (X_test, y_test): {X_test.shape}, {y_test.shape}")

model = AdvancedNN(window_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

num_batches = len(X_train) // batch_size

for epoch in range(epochs):
    epoch_loss = 0  
    for i in range(num_batches):  
        start = i * batch_size  
        end = start + batch_size 
        batch_X = X_train[start:end] 
        batch_y = y_train[start:end] 

        optimizer.zero_grad()  
        outputs = model(batch_X).squeeze() 
        loss = criterion(outputs, batch_y) 
        loss.backward()  
        optimizer.step() 

        epoch_loss += loss.item()  
    with torch.no_grad(): 
        val_outputs = model(X_val).squeeze()  
        
        val_loss = criterion(val_outputs, y_val)  

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / num_batches:.4f}, Validation Loss: {val_loss.item():.4f}")

# torch.save(model.state_dict(), 'advanced_nn_model.pth')  # Save the model's parameters to a .pth file


metrics_nn = {'MSE': [], 'MAE': [], 'PSNR': [], 'SNR': [], 'R2': []}
metrics_lms = {'MSE': [], 'MAE': [], 'PSNR': [], 'SNR': [], 'R2': []}
metrics_nlms = {'MSE': [], 'MAE': [], 'PSNR': [], 'SNR': [], 'R2': []}


plt.figure(figsize=(14, 5 * len(clean_signals)))


for idx, (clean_signal, noisy_signal) in enumerate(zip(clean_signals, noisy_signals)):


    with torch.no_grad():  
        X_signal, _ = prepare_dataset([noisy_signal], [clean_signal], window_size)  
        filtered_nn = model(torch.tensor(X_signal, dtype=torch.float32)).squeeze().numpy()  


    filtered_lms, _ = lms_filter(noisy_signal, clean_signal, mu, window_size)  


    filtered_nlms, _ = nlms_filter(noisy_signal, clean_signal, mu, window_size)  


    mse_nn = mean_squared_error(clean_signal[window_size:], filtered_nn)  
    mae_nn = mean_absolute_error(clean_signal[window_size:], filtered_nn)  
    psnr_nn = calculate_psnr(filtered_nn, clean_signal[window_size:])  
    snr_nn = calculate_snr(filtered_nn, clean_signal[window_size:])  
    r2_nn = r2_score(clean_signal[window_size:], filtered_nn)  


    metrics_nn['MSE'].append(mse_nn)
    metrics_nn['MAE'].append(mae_nn)
    metrics_nn['PSNR'].append(psnr_nn)
    metrics_nn['SNR'].append(snr_nn)
    metrics_nn['R2'].append(r2_nn)


    filtered_lms = filtered_lms[window_size:]  
    clean_signal_trimmed = clean_signal[window_size:]  

    mse_lms = mean_squared_error(clean_signal_trimmed, filtered_lms)  
    mae_lms = mean_absolute_error(clean_signal_trimmed, filtered_lms)  
    psnr_lms = calculate_psnr(filtered_lms, clean_signal_trimmed)  
    snr_lms = calculate_snr(filtered_lms, clean_signal_trimmed)  
    r2_lms = r2_score(clean_signal_trimmed, filtered_lms)  


    metrics_lms['MSE'].append(mse_lms)
    metrics_lms['MAE'].append(mae_lms)
    metrics_lms['PSNR'].append(psnr_lms)
    metrics_lms['SNR'].append(snr_lms)
    metrics_lms['R2'].append(r2_lms)


    filtered_nlms = filtered_nlms[window_size:]  
    clean_signal_trimmed = clean_signal[window_size:]  

    mse_nlms = mean_squared_error(clean_signal_trimmed, filtered_nlms)  
    mae_nlms = mean_absolute_error(clean_signal_trimmed, filtered_nlms)
    psnr_nlms = calculate_psnr(filtered_nlms, clean_signal_trimmed)
    snr_nlms = calculate_snr(filtered_nlms, clean_signal_trimmed)  
    r2_nlms = r2_score(clean_signal_trimmed, filtered_nlms)  


    metrics_nlms['MSE'].append(mse_nlms)
    metrics_nlms['MAE'].append(mae_nlms)
    metrics_nlms['PSNR'].append(psnr_nlms)
    metrics_nlms['SNR'].append(snr_nlms)
    metrics_nlms['R2'].append(r2_nlms)


    plt.subplot(len(clean_signals), 3, 3 * idx + 1)
    plt.plot(clean_signal, label='Clean Signal')
    plt.plot(noisy_signal, label='Noisy Signal')
    plt.plot(filtered_nn, label='Filtered by NN')
    plt.legend()
    plt.title(f"Signal {idx + 1}: NN Filtered")

    plt.subplot(len(clean_signals), 3, 3 * idx + 2)
    plt.plot(clean_signal, label='Clean Signal')
    plt.plot(noisy_signal, label='Noisy Signal')
    plt.plot(filtered_lms, label='Filtered by LMS')
    plt.legend()
    plt.title(f"Signal {idx + 1}: LMS Filtered")

    plt.subplot(len(clean_signals), 3, 3 * idx + 3)
    plt.plot(clean_signal, label='Clean Signal')
    plt.plot(noisy_signal, label='Noisy Signal')
    plt.plot(filtered_nlms, label='Filtered by NLMS')
    plt.legend()
    plt.title(f"Signal {idx + 1}: NLMS Filtered")


plt.tight_layout()
plt.show()


avg_metrics_nn = {metric: (np.mean(values), np.std(values)) for metric, values in metrics_nn.items()}
avg_metrics_lms = {metric: (np.mean(values), np.std(values)) for metric, values in metrics_lms.items()}
avg_metrics_nlms = {metric: (np.mean(values), np.std(values)) for metric, values in metrics_nlms.items()}


print("\nNN Metrics:")
for metric, (mean_val, std_val) in avg_metrics_nn.items():
    print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

print("\nLMS Metrics:")
for metric, (mean_val, std_val) in avg_metrics_lms.items():
    print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

print("\nNLMS Metrics:")
for metric, (mean_val, std_val) in avg_metrics_nlms.items():
    print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")


output_dirs = {
    'NN': 'data/output/filtered_nn',
    'LMS': 'data/output/filtered_lms',
    'NLMS': 'data/output/filtered_nlms'
}


for dir_name in output_dirs.values():
    os.makedirs(dir_name, exist_ok=True)


for idx in range(30):
    if idx >= len(clean_signals):  
        break

    clean_signal, noisy_signal = clean_signals[idx], noisy_signals[idx]

    # NN filtering
    with torch.no_grad():
        X_signal, _ = prepare_dataset([noisy_signal], [clean_signal], window_size)
        filtered_nn = model(torch.tensor(X_signal, dtype=torch.float32)).squeeze().detach().numpy()

    # LMS filtering
    filtered_lms, _ = lms_filter(noisy_signal, clean_signal, mu, window_size)

    # NLMS filtering
    filtered_nlms, _ = nlms_filter(noisy_signal, clean_signal, mu, window_size)

    # Save files
    file_name_nn = os.path.join(output_dirs['NN'], f"filtered_nn_{idx+1}.wav")
    file_name_lms = os.path.join(output_dirs['LMS'], f"filtered_lms_{idx+1}.wav")
    file_name_nlms = os.path.join(output_dirs['NLMS'], f"filtered_nlms_{idx+1}.wav")

    wavfile.write(file_name_nn, fs_clean, (filtered_nn * 32767).astype(np.int16))
    wavfile.write(file_name_lms, fs_clean, (filtered_lms * 32767).astype(np.int16))
    wavfile.write(file_name_nlms, fs_clean, (filtered_nlms * 32767).astype(np.int16))

    print(f"Saved filtered signals for audio {idx+1}")
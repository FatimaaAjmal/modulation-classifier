import numpy as np
import matplotlib.pyplot as plt


def generate_bpsk(n_samples):
    bits = np.random.randint(0, 2, n_samples)
    signal = 2 * bits - 1
    return signal


def generate_qpsk(n_samples):
    bits = np.random.randint(0, 2, (n_samples, 2))
    I = 2 * bits[:, 0] - 1
    Q = 2 * bits[:, 1] - 1
    signal = I + 1j * Q
    return signal


def generate_16qam(n_samples):
    bits = np.random.randint(0, 2, (n_samples, 4))
    I = 2 * (2 * bits[:, 0] - 1) + (2 * bits[:, 1] - 1)
    Q = 2 * (2 * bits[:, 2] - 1) + (2 * bits[:, 3] - 1)
    signal = I + 1j * Q
    return signal


def add_noise(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise


def generate_dataset(n_samples=1000, snr_db=20):
    dataset = []
    labels = []
    
    # Generate BPSK signals
    for _ in range(n_samples):
        signal = generate_bpsk(128)
        signal = add_noise(signal.astype(complex), snr_db)
        dataset.append([signal.real, signal.imag])
        labels.append(0)
    
    # Generate QPSK signals
    for _ in range(n_samples):
        signal = generate_qpsk(128)
        signal = add_noise(signal, snr_db)
        dataset.append([signal.real, signal.imag])
        labels.append(1)
    
    # Generate 16QAM signals
    for _ in range(n_samples):
        signal = generate_16qam(128)
        signal = add_noise(signal, snr_db)
        dataset.append([signal.real, signal.imag])
        labels.append(2)
    
    dataset = np.array(dataset, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    np.save('dataset.npy', dataset)
    np.save('labels.npy', labels)
    print(f"Dataset saved! Shape: {dataset.shape}")
    print(f"Labels saved! Shape: {labels.shape}")
    print("0=BPSK, 1=QPSK, 2=16QAM")

generate_dataset()
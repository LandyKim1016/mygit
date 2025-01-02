import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import pydub as pyd

def clip(file_path,min_time,max_time):

    sound = pyd.AudioSegment.from_file(file_path)
    clip = sound[min_time:max_time]
    clip.export('noise_0_4.wav',format= 'wav')


def stft_q(file_path, sampling_frequency, time_duration, save_path_real, save_path_img):

    file_path = file_path
    fs = sampling_frequency
    duration = time_duration

    y, sr = librosa.load(file_path, sr=fs, duration=duration)

    n_fft = fs # 주파수 분해능 1Hz
    hop_length = int(n_fft * 0.2) # 오버랩 80%
    window = 'hann'

    SQ = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)

    S_real = np.real(SQ)
    S_img = np.imag(SQ)

    S_amplitude = np.abs(SQ)
    reference = np.max(SQ) # dB의 reference 값을 실수부와 허수부의 동일한 값을 적용하기 위해서 amplitude의 최댓값으로 지정

    S_real_db = librosa.amplitude_to_db(S_real, ref=np.max)
    S_img_db = librosa.amplitude_to_db(S_img, ref=np.max)

    time_step = hop_length / fs 
    num_time_steps = S_real.shape[1]
    times = np.arange(0, (num_time_steps - 1)* time_step + time_step, time_step)

    frequency_step = fs / n_fft
    frequencies = np.arange(0, 2001, frequency_step)

    spectrogram_df1 = pd.DataFrame(S_real, index=frequencies, columns=times)
    spectrogram_df2 = pd.DataFrame(S_img, index=frequencies, columns=times)

    csv_output_path1 = save_path_real
    csv_output_path2 = save_path_img
    spectrogram_df1.to_csv(csv_output_path1)
    spectrogram_df2.to_csv(csv_output_path2)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(S_real, ref=reference), sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Real Spectrogram')
    plt.ylim([20,500])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(S_img, ref=reference), sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Imaginary Spectrogram')
    plt.ylim([20,500])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()

stft_q('raw_data/noise.wav', 4000, 20, 'spectrogram_no_real.csv','spectrogram_no_img.csv')
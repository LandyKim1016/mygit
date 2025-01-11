import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf

def stft_d(target_file_path, predicted_file_path ,samplin_frequency, time_duration,narrow_band_range, save_path):

    target = target_file_path
    predicted = predicted_file_path
    fs = samplin_frequency
    duration = time_duration

    yt, srt = librosa.load(target, sr=fs)
    yp, srp = librosa.load(predicted, sr=fs)

    print(yp.shape)
    print(srp)

    n_fft = fs # 주파수 분해능 1Hz
    hop_length = int(n_fft * 0.2) # 오버랩 80%
    window = 'hann'

    defferential = yt + yp
    print('yt' , yt)
    print('yp', yp)
    print('defferential shape', defferential.shape)

    S = librosa.stft(defferential, n_fft=n_fft, hop_length=hop_length, window=window)
    S1 = librosa.stft(yp, n_fft=n_fft, hop_length=hop_length, window=window)

    reference = librosa.stft(yt, n_fft=n_fft, hop_length=hop_length, window=window)

    S_amplitude = np.abs(S)
    S_amplitude1 = np.abs(S1)
    reference1 = np.abs(reference)
    print(S_amplitude1.shape)

    time_step = hop_length / fs 
    num_time_steps = S.shape[1]
    times = np.arange(0, (num_time_steps - 1)* time_step + time_step, time_step)

    frequency_step = fs / n_fft
    frequencies = np.arange(0, 2001, frequency_step)

    spectrogram_df = pd.DataFrame(S, index=frequencies, columns=times)


    csv_output_path = save_path

    spectrogram_df.to_csv(csv_output_path)

    vmin = np.min(librosa.amplitude_to_db(reference1,ref=20))
    vmax = np.max(librosa.amplitude_to_db(reference1,ref=20))
    print(S_amplitude.shape)
    print(S_amplitude1.shape)

    #data slice
    # 협대역 필터링

    narrow_band_range = narrow_band_range
    sr=4000  # 80-100Hz 대역 필터링

# 1. Extract narrow-band frequency range
    def extract_narrow_band(spectrogram, sr, freq_range):
        # 주파수 축 생성
        freq_bins = np.linspace(0, sr / 2, spectrogram.shape[0])
        lower_idx = np.searchsorted(freq_bins, freq_range[0])
        upper_idx = np.searchsorted(freq_bins, freq_range[1])
        # 주파수 대역 필터링
        return spectrogram[lower_idx:upper_idx, :]
    S_e = extract_narrow_band(S_amplitude, sr, narrow_band_range)
    S1_e = extract_narrow_band(S_amplitude1, sr, narrow_band_range)
    R_e = extract_narrow_band(reference1, sr, narrow_band_range)

    plt.figure(figsize=(10, 6))
    plt.imshow(librosa.amplitude_to_db(S_e,ref=20), aspect='auto', origin='lower', cmap='viridis',
           extent=[0, S_e.shape[1], narrow_band_range[0], narrow_band_range[1]], vmin=vmin,vmax=vmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('defferenctial spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency (Hz)')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.imshow(librosa.amplitude_to_db(R_e - S1_e,ref=20), aspect='auto', origin='lower', cmap='viridis',
           extent=[0, S_e.shape[1], narrow_band_range[0], narrow_band_range[1]], vmin=vmin,vmax=vmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('defferenctial spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency (Hz)')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.imshow(librosa.amplitude_to_db(R_e,ref=20), aspect='auto', origin='lower', cmap='viridis',
           extent=[0,R_e.shape[1], narrow_band_range[0], narrow_band_range[1]], vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Target Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def signal2fft(file1, file2, expected_sr=4000):
    """
    2개의 신호를 파일에서 로드하고,
    3번째 신호는 두 신호의 차(signal1 - signal2)를 만들어
    각각 FFT 후, 한 그래프에 dB 스케일 크기 스펙트럼으로 표시.

    주파수 분해능은 1Hz(즉, N=4000)로 가정.
    """
    # 1) 첫 번째, 두 번째 신호 로드
    data1, sr1 = sf.read(file1)
    data2, sr2 = sf.read(file2)
    
    # 샘플링 레이트 확인
    if sr1 != expected_sr or sr2 != expected_sr:
        raise ValueError(f"샘플링 레이트가 {expected_sr} Hz가 아닙니다."
                         f" file1 sr={sr1}, file2 sr={sr2}")

    # 다채널(스테레오 등)인 경우 모노(첫 채널)로 처리
    if data1.ndim > 1:
        data1 = data1[:, 0]
    if data2.ndim > 1:
        data2 = data2[:, 0]

    # 2) 신호 길이를 4000샘플(N=4000)로 맞춤
    def fix_length_to_n(data, n=4000):
        length = len(data)
        if length < n:
            # zero-padding
            pad_len = n - length
            data = np.pad(data, (0, pad_len), mode='constant')
        elif length > n:
            # 초과 샘플 제거
            data = data[:n]
        return data

    data1 = fix_length_to_n(data1, expected_sr)
    data2 = fix_length_to_n(data2, expected_sr)

    # 3) 두 신호의 차 = 세 번째 신호
    data_diff = data1 + data2

    # 4) FFT 수행 (실수 신호 -> rfft)
    fft1 = np.fft.rfft(data1)
    fft2 = np.fft.rfft(data2)
    fft_diff = np.fft.rfft(data_diff)

    # 5) 주파수 축 계산 (N=4000, sr=4000 → 분해능 1Hz)
    freqs = np.fft.rfftfreq(expected_sr, d=1.0/expected_sr)

    # 6) 크기 스펙트럼(dB) 계산
    #    log(0) 방지를 위해 아주 작은 값(1e-12) 추가
    mag1_db = 10 * np.log10(np.maximum(np.abs(fft1), 1e-12))
    mag2_db = 10 * np.log10(np.maximum(np.abs(fft2), 1e-12))
    mag_diff_db = 10 * np.log10(np.maximum(np.abs(fft_diff), 1e-12))

    def game(spectrum_target, spectrum_predict, fre_band):
    
        band = np.arange(fre_band[0],fre_band[1], 1)
        print(band)

        max = 0

        buffer = np.zeros((fre_band[1]-fre_band[0]))

        for i in band:
            buffer[i-fre_band[0]] = spectrum_target[i] - spectrum_predict[i]

        max = np.max(buffer)

        buffer1 = 0
        buffer2 = 0

        for i in band:
            if buffer1 < spectrum_target[i]:
                buffer1 = spectrum_target[i]
                buffer2 = i

        print(spectrum_target[buffer2])
        print(spectrum_predict[buffer2])

        d = spectrum_target[buffer2] - spectrum_predict[buffer2]

        print(f'최대 peak에서의 주파수{buffer2}Hz 에서의 감소량 {d}dB')
        print('최대 dB값의 감소량 : ', max,'dB')

    game(mag1_db, mag_diff_db, (80,120))

    # 7) 그래프 그리기 (dB 스케일)
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mag1_db, label='ANC off (dB)', alpha=0.8)
    #plt.plot(freqs, mag2_db, label='predict (dB)', alpha=0.8)
    plt.plot(freqs, mag_diff_db, label='ANC on (dB)', alpha=0.8)

    plt.title('Magnitude Spectrums')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.xlim(80, 120)
    plt.show()

    

#stft_d('raw_data/noise2.wav','reconstructed_audio1.wav', 4000, 40,(20,200), 'defferential1.wav')
#signal2fft(file1, file2, expected_sr=4000)
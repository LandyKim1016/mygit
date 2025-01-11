import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import librosa

# 기본 설정
SAMPLE_RATE = 44100  # 샘플링 레이트
DURATION = 1         # 실시간 업데이트 간격 (초)
FREQ_LIMIT = 2000    # 주파수 분석 범위 상한 (Hz)
RESOLUTION = 1       # 주파수 분해능 (Hz)

# n_fft 계산 (분해능 1Hz를 만족하기 위해)
n_fft = int(SAMPLE_RATE / RESOLUTION)

# 스트림 데이터 버퍼
buffer_size = SAMPLE_RATE * DURATION
audio_buffer = np.zeros(buffer_size)

# 오디오 데이터 업데이트 콜백
def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

# 스펙트로그램 업데이트 함수
def update_spectrogram(frame):
    global audio_buffer
    # STFT 계산
    D = librosa.stft(audio_buffer, n_fft=n_fft, hop_length=n_fft//4, win_length=n_fft)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    reference = np.max(np.abs(D))

    D_real = np.real(D)
    D_imag = np.imag(D)

    Dr_db = librosa.amplitude_to_db(D_real, ref=np.max)
    Di_db = librosa.amplitude_to_db(D_imag, ref=np.max)

    # 주파수 범위 제한
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=n_fft)
    freq_mask = freqs <= FREQ_LIMIT
    Dr_db_limited = Dr_db[freq_mask, :]
    Di_db_limited = Di_db[freq_mask, :]
    freqs_limited = freqs[freq_mask]

    

    # 스펙트로그램 출력
    ax_real.clear()
    img = ax_real.imshow(Dr_db_limited, aspect='auto', origin='lower',
                    extent=[0, DURATION, freqs_limited[0], freqs_limited[-1]],
                    cmap='viridis')
    ax_real.set_title('Real Spectrogram')
    ax_real.set_xlabel('Time (s)')
    ax_real.set_ylabel('Frequency (Hz)')
    #fig_real.colorbar(img, ax=ax_real, format="%+2.0f dB")
    
    ax_imag.clear()
    img = ax_imag.imshow(Di_db_limited, aspect='auto', origin='lower',
                    extent=[0, DURATION, freqs_limited[0], freqs_limited[-1]],
                    cmap='viridis')
    ax_imag.set_title('Imag Spectrogram')
    ax_imag.set_xlabel('Time (s)')
    ax_imag.set_ylabel('Frequency (Hz)')
    fig_real.colorbar(img, ax=ax_imag, format="%+2.0f dB")

# Matplotlib 설정
fig_real, ax_real = plt.subplots()
fig_imag, ax_imag = plt.subplots()

# 오디오 입력 스트림 시작
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE)
stream.start()

# 애니메이션 실행
ani = FuncAnimation(fig_imag, update_spectrogram, interval=DURATION * 1000)
plt.show()

# 스트림 종료
stream.stop()

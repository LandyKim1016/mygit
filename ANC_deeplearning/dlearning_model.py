import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Parameters
sampling_rate = 4000
n_fft = sampling_rate  # 1Hz 분해능
hop_length = int(n_fft * 0.2)  # 80% overlap
frame_size = 1
narrow_band_range = (95, 115)  # 80-100Hz 대역 필터링

# 1. Extract narrow-band frequency range
def extract_narrow_band(spectrogram, sr, freq_range):
    # 주파수 축 생성
    freq_bins = np.linspace(0, sr / 2, spectrogram.shape[0])
    lower_idx = np.searchsorted(freq_bins, freq_range[0])
    upper_idx = np.searchsorted(freq_bins, freq_range[1])
    # 주파수 대역 필터링
    return spectrogram[lower_idx:upper_idx, :]

# 2. Load audio and preprocess
def preprocess_audio(audio_path):
    # WAV 파일 로드
    y, sr = librosa.load(audio_path, sr=sampling_rate)
    # STFT 수행
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # 실수부와 허수부 분리
    real_part = np.real(stft)
    imag_part = np.imag(stft)
    # 정규화
    real_part = real_part / np.max(real_part)
    rmax = np.max(real_part)
    imag_part = imag_part / np.max(imag_part)
    imax = np.max(imag_part)
    # 협대역 필터링
    real_part = extract_narrow_band(real_part, sr, narrow_band_range)
    imag_part = extract_narrow_band(imag_part, sr, narrow_band_range)
    return real_part, imag_part, sr , rmax, imax

# 3. Slice spectrogram for model input
def slice_spectrogram(real_spectrogram, imag_spectrogram, frame_size=1):
    # 슬라이스된 입력 데이터 생성
    input_real = []
    input_imag = []
    for i in range(real_spectrogram.shape[1] - frame_size + 1):
        input_real.append(real_spectrogram[:, i:i + frame_size])
        input_imag.append(imag_spectrogram[:, i:i + frame_size])
    # 모델 입력 형식으로 변환
    input_real = np.stack(input_real, axis=0)[..., np.newaxis]  # Add channel dimension
    input_imag = np.stack(input_imag, axis=0)[..., np.newaxis]  # Add channel dimension
    print(input_real.shape)
    return input_real, input_imag

# 4. Predict using the trained model
def predict_audio(model_path, input_real, input_imag):
    # 모델 로드
    model = load_model(model_path)
    # 예측 수행
    predicted_real, predicted_imag = model.predict([input_real, input_imag])
    return predicted_real, predicted_imag

# 5. Reconstruct audio
def reconstruct_audio(predicted_real, predicted_imag, n_fft, hop_length):
    
    print('predicted frame shape :',predicted_real.shape)
    reconstructed_real = np.concatenate([frame.squeeze(axis=-1) for frame in predicted_real], axis=1)

    print('predicted frame shape :',predicted_imag.shape)
    reconstructed_imag = np.concatenate([frame.squeeze(axis=-1) for frame in predicted_imag], axis=1)

    reconstructed = reconstructed_real + 1j * reconstructed_imag
    print(reconstructed.dtype)

    spectrogram_buffer = np.zeros((2001,101))
    spectrogram_buffer = spectrogram_buffer.astype(np.complex128)
    spectrogram_buffer[96:116,:] += reconstructed

    print('reconstructed spectrogrma shape : ',spectrogram_buffer.shape)
    #print('reconstructed spectrogrma data : ', spectrogram_buffer[101,:])

    # 시간 도메인으로 변환
    reconstructed_audio = librosa.istft(spectrogram_buffer, hop_length=hop_length)
    return reconstructed_audio

# 6. Real-time processing workflow
import soundfile as sf

# Step 5: WAV 파일 저장
def process_realtime_audio(wav_path, model_path):
    # Step 1: 전처리
    real_part, imag_part, sr , real_max, imag_max= preprocess_audio(wav_path)

    print('sampling_fre : ', sr)
    
    # Step 2: 슬라이스 데이터 생성
    input_real, input_imag = slice_spectrogram(real_part, imag_part, frame_size=frame_size)

    print(input_real.shape)
    
    # Step 3: 모델 예측
    predicted_real, predicted_imag = predict_audio(model_path, input_real, input_imag)
    print(predicted_real.shape)

    # Step 4: 예측 결과 복원
    predicted_real = predicted_real * real_max*1000 # ##스케일 보정이 필요한 상태임. 아마도 model.save에서 추가적인 기능들이 저장되어서 어디까지 스케일 조정이 들어간지 모르는 상태태
    predicted_imag = predicted_imag * imag_max*1000
    
    reconstructed_audio = reconstruct_audio(predicted_real, predicted_imag, n_fft, hop_length)

    print('audio shape : ',reconstructed_audio.shape)
    
    # Step 5: WAV 파일로 저장 (soundfile 사용)
    sf.write('reconstructed_audio1.wav', reconstructed_audio, sr)
    print("Reconstructed audio saved as 'reconstructed_audio.wav'")

# Example usage
wav_path = 'raw_data/noise1.wav'  # 입력 WAV 파일 경로
model_path = 'my_model1.h5'  # 저장된 모델 경로
process_realtime_audio(wav_path, model_path)


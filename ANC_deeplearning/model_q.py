import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation ,concatenate
from sklearn.model_selection import train_test_split

# Parameters
frame_size = 1
narrow_band_range = (80, 100)  # Frequency range for narrow-band ANC in Hz

# 1. Load audio for input and target, generate spectrograms 복소푸리에 변환후 실수와 허수 스펙트로그램 분리 및 정규화 // 아마 dB는 음수처리 때문에 최종해석 때!!
def load_and_generate_input_target_spectrograms(input_path, target_path):
    # Input data
    y_input, sr_input = librosa.load(input_path, sr=None)
    stft_input = librosa.stft(y_input, n_fft=sr_input, hop_length=int(sr_input//5))
    y_input_real = np.real(stft_input)
    y_input_imag = np.real(stft_input)
    n_input_real = y_input_real / np.max(y_input_real) # 데이터 정규화
    n_input_imag = y_input_imag / np.max(y_input_imag)

    # Target data
    y_target, sr_target = librosa.load(target_path, sr=None)
    stft_target = librosa.stft(y_target, n_fft=sr_target, hop_length=int(sr_target//5))
    y_target_real = np.real(stft_target)
    y_target_imag = np.real(stft_target)
    n_target_real = y_target_real / np.max(y_target_real) # 데이터 정규화
    n_target_imag = y_target_imag / np.max(y_target_imag)
    
    return n_input_real, n_input_imag, n_target_real, n_target_imag, sr_input, sr_target ,np.max(y_input_real) ,np.max(y_input_imag),np.max(y_target_real) ,np.max(y_target_imag)

# 2. Extract narrow-band frequency range
def extract_narrow_band(spectrogram, sr, freq_range):
    freq_bins = np.linspace(0, sr / 2, spectrogram.shape[0])
    lower_idx = np.searchsorted(freq_bins, freq_range[0])
    upper_idx = np.searchsorted(freq_bins, freq_range[1])
    return spectrogram[lower_idx:upper_idx, :]

# 3. Preprocess spectrogram for input and target
def preprocess_input_target(real_spectrogram, imag_spectrogram):
    height_input, width_input = real_spectrogram.shape
    height_target, width_target = imag_spectrogram.shape

    # Process real data (1-frame slices)
    real_data = []
    for i in range(width_input - frame_size + 1):
        real_data.append(real_spectrogram[:, i + frame_size - 1])
    real_data = np.stack(real_data, axis=0)[..., np.newaxis]  # Add channel dimension
    real_data = np.expand_dims(real_data, axis=-1)  # Add channel dimension

    # Process imag data (1-frame slices)
    imag_data = []
    for i in range(width_target - frame_size + 1):
        imag_data.append(imag_spectrogram[:, i + frame_size - 1])
    imag_data = np.stack(imag_data, axis=0)[..., np.newaxis]  # Add channel dimension
    imag_data = np.expand_dims(imag_data, axis=-1)  # Add channel dimension


    return real_data, imag_data


# 4. Build parallel CRN model
def build_parallel_crn(input_shape):
    # Real branch
    input_real = Input(shape=input_shape)
    x_real = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(input_real)
    x_real = BatchNormalization()(x_real)
    x_real = Activation('tanh')(x_real)
    x_real = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same')(x_real)
    x_real = BatchNormalization()(x_real)
    x_real = Conv2D(1, (1, 1))(x_real)
    output_real = Activation('sigmoid')(x_real)

    # Imaginary branch
    input_imag = Input(shape=input_shape)
    x_imag = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(input_imag)
    x_imag = BatchNormalization()(x_imag)
    x_imag = Activation('tanh')(x_imag)
    x_imag = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same')(x_imag)
    x_imag = BatchNormalization()(x_imag)
    x_imag = Conv2D(1, (1, 1))(x_imag)
    output_imag = Activation('sigmoid')(x_imag)

    # Combine real and imaginary outputs
    combined_model = Model(inputs=[input_real, input_imag], outputs=[output_real, output_imag])
    combined_model.compile(optimizer='adam', loss='mse')
    return combined_model

# 5. Train and predict
def train_and_predict(model, input_real, input_imag, target_real, target_imag):
    model.fit([input_real, input_imag], [target_real, target_imag], epochs=200, batch_size=20, validation_split=0.2)
    predicted_real, predicted_imag = model.predict([input_real, input_imag])
    return predicted_real, predicted_imag

# 6. Reconstruct spectrogram
def reconstruct_spectrogram(predicted_real, predicted_imag):

    print('predicted frame shape :',predicted_real.shape)
    reconstructed_real = np.concatenate([frame.squeeze(axis=-1) for frame in predicted_real], axis=1)

    print('predicted frame shape :',predicted_imag.shape)
    reconstructed_imag = np.concatenate([frame.squeeze(axis=-1) for frame in predicted_imag], axis=1)

    reconstructed = reconstructed_real + 1j * reconstructed_imag

    return reconstructed

# Example workflow
input_path = 'raw_data/input.wav'
target_path = 'raw_data/target.wav'

# Step 1: Load and preprocess
input_real, input_imag, target_real, target_imag, sr_input, sr_target ,input_real_max ,input_imag_max,target_real_max, target_imag_max = load_and_generate_input_target_spectrograms(input_path, target_path)

# Step 2: Extract narrow-band components
input_real_e = extract_narrow_band(input_real, sr_input, narrow_band_range)
input_imag_e = extract_narrow_band(input_imag, sr_input, narrow_band_range)
target_real_e = extract_narrow_band(target_real, sr_target, narrow_band_range)
target_imag_e = extract_narrow_band(target_imag, sr_target, narrow_band_range)

# Step 3: Preprocess input and target data
input_real_s, input_imag_s = preprocess_input_target(input_real_e, input_imag_e)
target_real_s, target_imag_s = preprocess_input_target(target_real_e, target_imag_e)

# Ensure sampling rates match for consistency
assert sr_input == sr_target, "Sampling rates of input and target must match."

# Step 4: Build and train model
input_shape = input_real_s.shape[1:]  # (height, frame_size, 1)
model = build_parallel_crn(input_shape)
model.summary()
predicted_real, predicted_imag = train_and_predict(model, input_real_s, input_imag_s, target_real_s, target_imag_s)

# Step 5: Predict and reconstruct
#predicted = model.predict(input_data)
reconstructed_spectrogram = reconstruct_spectrogram(predicted_real * input_real_max, predicted_imag * input_imag_max)
reconstructed_spectrogram = np.abs(reconstructed_spectrogram)
reconstructed_spectrogram = librosa.amplitude_to_db(reconstructed_spectrogram, ref=20)

target_spectrogram_q = (target_real_e * target_real_max) + 1j * (target_imag_e * target_imag_max)
target_spectrogram_a = np.abs(target_spectrogram_q)
target_spectrogram_a = librosa.amplitude_to_db(target_spectrogram_a, ref=20)

# librosa.display.specshow() 함수는 사용하면 안될 듯? 왜냐하면  기존의 스펙트로그램의 스펙으로 협대역 예측으로 추출된 2차원의 데이터를 스펙트로그램으로 나타내기에는 부적절함.
# Step 6: Visualize results using imshow

vmin = min(np.min(reconstructed_spectrogram), np.min(target_spectrogram_a))
vmax = max(np.max(reconstructed_spectrogram), np.max(target_spectrogram_a))

plt.figure(figsize=(10, 6))
plt.imshow(reconstructed_spectrogram, aspect='auto', origin='lower', cmap='viridis',
           extent=[0, reconstructed_spectrogram.shape[1], 80, 100], vmin=vmin,vmax=vmax)
plt.colorbar(format='%+2.0f dB')
plt.title('Reconstructed Target Spectrogram (80-100 Hz)')
plt.xlabel('Time Frames')
plt.ylabel('Frequency (Hz)')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(target_spectrogram_a
           , aspect='auto', origin='lower', cmap='viridis',
           extent=[0, reconstructed_spectrogram.shape[1], 80, 100], vmin=vmin, vmax=vmax)
plt.colorbar(format='%+2.0f dB')
plt.title('Target Spectrogram (80-100 Hz)')
plt.xlabel('Time Frames')
plt.ylabel('Frequency (Hz)')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(reconstructed_spectrogram - target_spectrogram_a, aspect='auto', origin='lower', cmap='viridis',
           extent=[0, reconstructed_spectrogram.shape[1], 80, 100], vmin=vmin, vmax=vmax)
plt.colorbar(format='%+2.0f dB')
plt.title('ANC Spectrogram (80-100 Hz)')
plt.xlabel('Time Frames')
plt.ylabel('Frequency (Hz)')
plt.show()
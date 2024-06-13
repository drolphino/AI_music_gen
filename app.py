import gradio as gr
import torch
from diffusers import DiffusionPipeline
from pydub import AudioSegment
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from accelerate import Accelerator
from typing import Tuple
import typing as T
from PIL import Image
import scipy.io.wavfile as wavfile
import tempfile
import os
import torchaudio

# Initialize the Accelerator
accelerator = Accelerator()

# Load the fine-tuned model
fine_tuned_model_path = "fine-tuned-riffusion-model"
pipe3 = DiffusionPipeline.from_pretrained(fine_tuned_model_path)

# Check if a GPU is available, otherwise use CPU
device = accelerator.device if torch.cuda.is_available() else "cpu"

# Move the model to the appropriate device
pipe3 = pipe3.to(device)

def waveform_from_spectrogram(
    Sxx: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    num_samples: int,
    sample_rate: int,
    mel_scale: bool = True,
    n_mels: int = 512,
    num_griffin_lim_iters: int = 32,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Reconstruct a waveform from a spectrogram.
    This is an approximate inverse of spectrogram_from_waveform, using the Griffin-Lim algorithm
    to approximate the phase.
    """
    Sxx_torch = torch.from_numpy(Sxx).to(device)

    if mel_scale:
        mel_inv_scaler = torchaudio.transforms.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk"
        ).to(device)

        Sxx_torch = mel_inv_scaler(Sxx_torch)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        n_iter=num_griffin_lim_iters,
    ).to(device)

    waveform = griffin_lim(Sxx_torch).cpu().numpy()

    return waveform

def audio_from_spectrogram_image(image: Image.Image) -> T.Tuple[BytesIO, float]:
    """
    Reconstruct a WAV audio clip from a spectrogram image. Also returns the duration in seconds.
    """
    max_volume = 50
    power_for_image = 0.25
    Sxx = spectrogram_from_image(image, max_volume=max_volume, power_for_image=power_for_image)

    sample_rate = 44100  # [Hz]
    clip_duration_ms = 5000  # [ms]

    bins_per_image = 512
    n_mels = 512

    # FFT parameters
    window_duration_ms = 100  # [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10  # [ms]

    # Derived parameters
    num_samples = int(image.width / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    samples = waveform_from_spectrogram(
        Sxx=Sxx,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        num_samples=num_samples,
        sample_rate=sample_rate,
        mel_scale=True,
        n_mels=n_mels,
        num_griffin_lim_iters=32,
        device=device
    )

    wav_bytes = BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples.astype(np.int16))
    wav_bytes.seek(0)

    duration_s = float(len(samples)) / sample_rate

    return wav_bytes, duration_s

def spectrogram_from_image(
    image: Image.Image, max_volume: float = 50, power_for_image: float = 0.25
) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.
    """
    data = np.array(image).astype(np.float32)

    data = data[::-1, :, 0]

    data = 255 - data

    data = data * max_volume / 255

    data = np.power(data, 1 / power_for_image)

    return data

def predict2(prompt, negative_prompt):
    spec = pipe3(
        prompt,
        negative_prompt=negative_prompt,
        width=768,
        num_inference_steps=50
    ).images[0]

    wav_bytes, duration_s = audio_from_spectrogram_image(spec)
    wav_bytes.seek(0)
    return wav_bytes, spec

def generate_melody(prompt, negative_prompt):
    wav_bytes, spec = predict2(prompt, negative_prompt)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
        temp_audio_file.write(wav_bytes.read())
        temp_audio_file_path = temp_audio_file.name

    audio = AudioSegment.from_file(temp_audio_file_path, format="wav")

    audio_file = "output.wav"
    audio.export(audio_file, format="wav")

    audio_data = BytesIO()
    audio.export(audio_data, format="wav")
    audio_data.seek(0)

    samples = np.array(audio.get_array_of_samples())
    plt.figure(figsize=(10, 4))
    plt.plot(samples)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid()

    waveform_image_path = 'waveform.png'
    plt.savefig(waveform_image_path)
    plt.close()

    os.remove(temp_audio_file_path)

    return audio_file, waveform_image_path

iface = gr.Interface(
    fn=generate_melody,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt")
    ],
    outputs=[
        gr.Audio(label="Generated Audio"),
        gr.Image(label="Waveform")
    ],
    title="Melody Generator",
    description="Enter a prompt and a negative prompt to generate a melody."
)

iface.launch()


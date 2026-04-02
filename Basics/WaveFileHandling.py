import wave
import numpy as np

def ReadWaveAsNumpyArray(Filename):
    """Reads a wave file and returns the audio data as a numpy array along with the sample rate.

    Args:
        Filename (str): Path to the wave file.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The audio data as a numpy array.
            - int: The sample rate of the audio data.
    """
    with wave.open(Filename, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        raw_data = wf.readframes(num_frames)

        if sample_width == 1:
            dtype = np.uint8  # 8-bit PCM
        elif sample_width == 2:
            dtype = np.int16  # 16-bit PCM
        elif sample_width == 4:
            dtype = np.int32  # 32-bit PCM
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        audio_data = np.frombuffer(raw_data, dtype=dtype) / (2**(8 * sample_width - 1) - 1)

        if num_channels > 1:
            audio_data = audio_data.reshape(-1, num_channels)
            # extract first channel only
            audio_data = audio_data[:, 0]
    return audio_data, sample_rate

def WriteWaveAsNumpyArray(x, samplerate, Filename):
    ### source:
    ### https://stackoverflow.com/questions/40782159/writing-wav-file-using-python-numpy-array-and-wave-module
    # Convert to (little-endian) 16 bit integers.
    audio = (x * (2 ** 15 - 1)).astype("<h")

    with wave.open(Filename, "w") as f:
        # 2 Channels.
        f.setnchannels(1)
        # 2 bytes per sample.
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(audio.tobytes())
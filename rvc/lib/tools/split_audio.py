import numpy as np
import librosa


def format_time(seconds):
    """Convert seconds to mm:ss format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def process_audio(audio, sr=16000, silence_thresh=-35, min_silence_len=500, min_chunk_len=300000):
    """
    Splits an audio signal into segments at silence points.

    Parameters:
    - audio (np.ndarray): The audio signal to split.
    - sr (int): The sample rate of the input audio (default is 16000).
    - silence_thresh (int): Silence threshold in dB (default -35dB)
    - min_silence_len (int): Minimum silence duration in ms (default 500ms)
    - min_chunk_len (int): Minimum chunk duration in ms (default 300000ms = 5 minutes)

    Returns:
    - list of np.ndarray: A list of audio segments.
    - np.ndarray: The intervals where the audio was split.
    """
    # Convert ms to samples
    frame_length = int(min_silence_len / 1000 * sr)
    min_chunk_samples = int(min_chunk_len / 1000 * sr)
    hop_length = frame_length // 4  # Smaller hop length for better precision

    # Get intervals of non-silent audio
    intervals = librosa.effects.split(
        audio, 
        top_db=-silence_thresh,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Merge intervals that are too close or split if too long
    merged_intervals = []
    current_start = intervals[0][0]
    current_end = intervals[0][1]

    for start, end in intervals[1:]:
        gap_duration = start - current_end
        chunk_duration = end - current_start

        # If we've accumulated more than min_chunk_samples and found a silence, split
        if chunk_duration >= min_chunk_samples and gap_duration >= frame_length:
            merged_intervals.append([current_start, current_end])
            current_start = start
            current_end = end
        else:
            current_end = end

    # Add the final chunk
    merged_intervals.append([current_start, current_end])
    merged_intervals = np.array(merged_intervals)

    # Create audio segments
    audio_segments = [audio[start:end] for start, end in merged_intervals]
    
    print(f"Split points found at (seconds): {[i/sr for i in merged_intervals.flatten()]}")
    print(f"Split points in mm:ss format: {[format_time(i/sr) for i in merged_intervals.flatten()]}")
    print(f"Chunk durations (minutes): {[(end-start)/(sr*60) for start, end in merged_intervals]}")
    
    return audio_segments, merged_intervals


def merge_audio(audio_segments_org, audio_segments_new, intervals, sr_orig, sr_new):
    """
    Merges audio segments back into a single audio signal, filling gaps with silence.
    Assumes audio segments are already at sr_new.

    Parameters:
    - audio_segments_org (list of np.ndarray): The non-silent audio segments (at sr_orig).
    - audio_segments_new (list of np.ndarray): The non-silent audio segments (at sr_new).
    - intervals (np.ndarray): The intervals used for splitting the original audio.
    - sr_orig (int): The sample rate of the original audio
    - sr_new (int): The sample rate of the model
    Returns:
    - np.ndarray: The merged audio signal with silent gaps restored.
    """
    merged_audio = np.array([], dtype=audio_segments_new[0].dtype)
    sr_ratio = sr_new / sr_orig

    for i, (start, end) in enumerate(intervals):

        start_new = int(start * sr_ratio)
        end_new = int(end * sr_ratio)

        original_duration = len(audio_segments_org[i]) / sr_orig
        new_duration = len(audio_segments_new[i]) / sr_new
        duration_diff = new_duration - original_duration

        silence_samples = int(abs(duration_diff) * sr_new)
        silence_compensation = np.zeros(
            silence_samples, dtype=audio_segments_new[0].dtype
        )

        if i == 0 and start_new > 0:
            initial_silence = np.zeros(start_new, dtype=audio_segments_new[0].dtype)
            merged_audio = np.concatenate((merged_audio, initial_silence))

        if duration_diff > 0:
            merged_audio = np.concatenate((merged_audio, silence_compensation))

        merged_audio = np.concatenate((merged_audio, audio_segments_new[i]))

        if duration_diff < 0:
            merged_audio = np.concatenate((merged_audio, silence_compensation))

        if i < len(intervals) - 1:
            next_start_new = int(intervals[i + 1][0] * sr_ratio)
            silence_duration = next_start_new - end_new
            if silence_duration > 0:
                silence = np.zeros(silence_duration, dtype=audio_segments_new[0].dtype)
                merged_audio = np.concatenate((merged_audio, silence))

    return merged_audio

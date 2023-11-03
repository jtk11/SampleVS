import wave
import aifc
import os
import random
import numpy as np
import pygame
from pydub import AudioSegment

def load_wav(filename):

    with wave.open(filename, 'r') as f:
        frames = f.readframes(-1)
        sound_info = np.frombuffer(frames, 'int16')
        return sound_info

def load_sample(filename):
    if filename.lower().endswith(".wav"):
        with wave.open(filename, 'r') as f:
            frames = f.readframes(-1)
            return np.frombuffer(frames, 'int16')
    elif filename.lower().endswith(".aif") or filename.lower().endswith(".aiff"):
        with aifc.open(filename, 'r') as f:
            frames = f.readframes(-1)
            return np.frombuffer(frames, 'int16')

#sample_folder = "/Users/macbookpro/Documents/Samples/Body Mechanik Library/Drop Mekanik"
#sample_folder ="/Users/macbookpro/Documents/Samples/Poke 2 copy_all_audio"  # Replace with the path to your sample folder
#sample_folder = "/Users/macbookpro/Documents/Samples/VS/PPG style"
#sample_folder = "/Users/macbookpro/Documents/Samples/Drop Interpolated_all_audio"
sample_folder = "/Users/macbookpro/Documents/Samples/Insect Sounds"
#sample_folder ="/Users/macbookpro/Documents/TextToSample/generated"
#sample_folder = "/Users/macbookpro/Documents/Samples/My Samples/New"
# sample_folder = "/Users/macbookpro/Documents/Samples/FairlightOG/Fairlight Drop"

def get_random_samples_from_folder():
    all_files = [f for f in os.listdir(sample_folder) if os.path.isfile(os.path.join(sample_folder, f))]
    sample_files = [f for f in all_files if f.lower().endswith(('.wav', '.aif', '.aiff'))]
    return random.sample(sample_files, 4)
    
def set_channel_volumes(x, y):
    """
    Set the volume of each channel based on the ball's x and y position.
    This simulates the Prophet VS's joystick crossfading/morphing behavior.
    """
    width, height = screen_width, screen_height

    # Calculate relative position in the 2D space (0 to 1 for both x and y)
    rel_x = x / width
    rel_y = y / height

    # Bilinear interpolation for volume levels
    volumes = [
        (1-rel_x) * (1-rel_y),  # Top-left quadrant
        rel_x * (1-rel_y),      # Top-right quadrant
        (1-rel_x) * rel_y,      # Bottom-left quadrant
        rel_x * rel_y           # Bottom-right quadrant
    ]

    for i, channel in enumerate(channels):
        channel.set_volume(volumes[i])

def pitch_shift_drop_sample(audio_file, semitones):
    segment = None

    file_extension = os.path.splitext(audio_file)[1].lower()
    
    if file_extension == ".wav":
        segment = AudioSegment.from_wav(audio_file)
    elif file_extension in [".aif", ".aiff"]:
        segment = AudioSegment.from_file(audio_file, format="aiff")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    shift_factor = 2 ** (semitones / 12.0)
    shifted_rate = segment.frame_rate * shift_factor

    pitched = segment._spawn(segment.raw_data, overrides={
        "frame_rate": int(shifted_rate)
    }).set_frame_rate(segment.frame_rate)

    return pitched

def detune_sample(sample_path):
    """
    Detune a sample.
    
    Args:
    - sample_path: path to the audio sample
    
    Returns:
    - numpy array of processed audio
    """
    semitones_shift = random.uniform(-2, 2)  # Random shift between -2 to 2 semitones
    pitched_segment = pitch_shift_drop_sample(sample_path, semitones_shift)
    pitched_array = np.array(pitched_segment.get_array_of_samples())

    # Reapply granular effect if it was active
    if GRANULAR_MODE:
        grain_count = random.randint(1, 30)
        return apply_granular_effect(pitched_array, grain_count)
    
    return pitched_array

def oscillate_samples(samples_list):
    # Parameters
    snippet_duration = 1/220  # Adjusted for 220 Hz tuning
    zone_duration = 0.05  # Zone duration in seconds
    overlap_ratio = .68  # 50% overlap

    output_samples_list = []

    for data in samples_list:
        # Assume the sample rate is always 44100; adjust if needed
        sample_rate = 44100

        snippet_length_samples = int(snippet_duration * sample_rate)
        zone_length_samples = int(zone_duration * sample_rate)
        overlap_samples = int(overlap_ratio * zone_length_samples)

        # Initialize an array to hold the output data
        out_data = np.zeros(len(data), dtype=np.int16)
        snippet_count = 0

        # Loop over the data
        for i in range(0, len(data) - zone_length_samples, zone_length_samples - overlap_samples):
            # Get the current snippet
            snippet = data[i:i + snippet_length_samples]

            # Repeat the snippet until it fills a zone
            repeated_snippet = np.tile(snippet, zone_length_samples // snippet_length_samples)

            # Trim or extend the repeated snippet to match the zone length
            if len(repeated_snippet) > zone_length_samples:
                repeated_snippet = repeated_snippet[:zone_length_samples]
            else:
                repeated_snippet = np.pad(repeated_snippet, (0, zone_length_samples - len(repeated_snippet)))

            # Apply crossfade in overlapping areas
            if i != 0 and i + zone_length_samples - overlap_samples < len(data):
                # Apply linear crossfade ramp
                ramp = np.linspace(1, 0, overlap_samples)
                out_data[i:i + overlap_samples] = out_data[i:i + overlap_samples] * ramp + repeated_snippet[:overlap_samples] * (1 - ramp)
                out_data[i + overlap_samples:i + zone_length_samples] += repeated_snippet[overlap_samples:]
            else:
                out_data[i:i + zone_length_samples] += repeated_snippet

            snippet_count += 1

        # Add the processed data to the output list
        output_samples_list.append(out_data)

    return output_samples_list

GRANULAR_MODE = False  # Global variable to keep track of granular mode

def apply_granular_effect(audio_array, grain_count):
    """
    Apply granular effects to an audio array.

    Args:
    - audio_array: numpy array of audio data
    - grain_count: number of grains
    
    Returns:
    - numpy array of processed audio
    """
    grainsize = random.randint(500, 10000)  # Random grain size

    # Adjust grain size if necessary
    if len(audio_array) < grainsize:
        grainsize = len(audio_array)

    start_pos = random.randint(0, len(audio_array) - grainsize)  # Random starting position
    
    granulated_array = []
    for _ in range(grain_count):
        grain = audio_array[start_pos: start_pos + grainsize]
        
        # Apply Hanning window
        hanning_window = np.hanning(len(grain))
        grain = (grain * hanning_window).astype(np.int16)  # Convert back to int16 after multiplying

        granulated_array.extend(grain)
        start_pos += grainsize  # Move to the next grain
        
        # If we reach the end of the array, loop back to the beginning
        if start_pos + grainsize > len(audio_array):
            start_pos = 0

    return np.array(granulated_array)


# Modify the detune function to keep granular effects
def detune_with_granular_check(sample_path):
    """
    Detune a sample and reapply granular effects if needed.
    
    Args:
    - sample_path: path to the audio sample
    
    Returns:
    - numpy array of processed audio
    """
    semitones_shift = random.uniform(-2, 2)  # Random shift between -2 to 2 semitones
    pitched_segment = pitch_shift_drop_sample(sample_path, semitones_shift)
    pitched_array = np.array(pitched_segment.get_array_of_samples())

    # Reapply granular effect if it was active
    if GRANULAR_MODE:
        grain_count = random.randint(1, 30)
        return apply_granular_effect(pitched_array, grain_count)
    
    return pitched_array


class Trajectory:
    def __init__(self):
        self.nodes = [(random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(5)]
        self.current_node = 0
        self.progress = 0.0

    def update(self, dt, move_speed):
        """Update the ball position based on the trajectory."""
        start_pos = pygame.Vector2(*self.nodes[self.current_node])
        end_pos = pygame.Vector2(*self.nodes[(self.current_node + 1) % len(self.nodes)])  # Loops to the beginning when reaching the end

        distance = start_pos.distance_to(end_pos)
        move_time = distance / move_speed
        
        self.progress += dt / move_time

        # Clamp the progress value to the range [0, 1] to prevent ValueError
        clamped_progress = max(0, min(1, self.progress))
    
        # Using lerp (linear interpolation) with clamped progress
        new_pos = start_pos.lerp(end_pos, clamped_progress)  

        # Ensure the progress value doesn't exceed the range [0, 1] before using it in lerp
        if self.progress >= 1:
            self.progress = 0.0
            self.current_node = (self.current_node + 1) % len(self.nodes)  # Loops to the beginning when reaching the end

        return new_pos.x, new_pos.y



active_sample_paths = [
    "/Users/macbookpro/Documents/Samples/Body Mechanik Library/Drop Mekanik/Pad B Evil MSV_-24st.wav",
    "/Users/macbookpro/Documents/Samples/Body Mechanik Library/Drop Mekanik/Lead D# Collapser MNRK_-24st.wav",
    "/Users/macbookpro/Documents/Samples/Body Mechanik Library/Drop Mekanik/Tom IronTek 1_-48st.wav",
    "/Users/macbookpro/Documents/Samples/Body Mechanik Library/Drop Mekanik/Perc[125] Cm Horrific_-48st.wav"
]

samples = [load_wav(path) for path in active_sample_paths]
pygame.init()
screen_width, screen_height = 400, 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Prophet VS Joystick Emulator')

channels = []
pygame.mixer.init(frequency=44100)

for path in active_sample_paths:
    sound = pygame.mixer.Sound(path)
    channel = pygame.mixer.Channel(active_sample_paths.index(path))
    channel.play(sound, -1)
    channels.append(channel)

def draw_waveforms():
    # Normalize and scale the waveforms to fit into each quadrant
    for i, sample in enumerate(samples):
        normalized = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        if i == 0:  # Top-left quadrant
            x_start, y_start = 0, 0
        elif i == 1:  # Top-right quadrant
            x_start, y_start = screen_width // 2, 0
        elif i == 2:  # Bottom-left quadrant
            x_start, y_start = 0, screen_height // 2
        else:  # Bottom-right quadrant
            x_start, y_start = screen_width // 2, screen_height // 2
        
        for j, val in enumerate(normalized):
            if j < screen_width // 2:  # to ensure it fits within quadrant width
                y_pos = int((1.0 - val) * (screen_height // 2))
                screen.set_at((j + x_start, y_pos + y_start), (255, 255, 255))

x, y = screen_width // 2, screen_height // 2
running = True
trajectory = None
move_speed = 3000.0  # Pixels per second


def array_to_pygame_format(np_array):
    """Convert a numpy array to a pygame sound object."""
    # Check if array is mono by seeing if it's 1-dimensional
    if len(np_array.shape) == 1:
        # Convert mono array to stereo by duplicating the mono channel
        np_array = np.column_stack([np_array, np_array])
    return pygame.mixer.Sound(pygame.sndarray.make_sound(np_array))

while running:
    screen.fill((0, 0, 0))
    # Draw quadrant lines
    pygame.draw.line(screen, (255, 255, 255), (screen_width // 2, 0), (screen_width // 2, screen_height))
    pygame.draw.line(screen, (255, 255, 255), (0, screen_height // 2), (screen_width, screen_height // 2))

    draw_waveforms()

    dt = pygame.time.Clock().tick(60) / 1000.0  # Get the elapsed time since last frame in seconds
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
            x, y = event.pos
            set_channel_volumes(x, y)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # Random replacement
                active_sample_paths = [os.path.join(sample_folder, path) for path in get_random_samples_from_folder()]
                samples = [load_sample(path) for path in active_sample_paths]
                # Reload sounds in pygame
                for i, path in enumerate(active_sample_paths):
                    sound = pygame.mixer.Sound(path)
                    channels[i].stop()
                    channels[i].play(sound, -1)
            elif event.key == pygame.K_d:  # Detune current samples
                for i, path in enumerate(active_sample_paths):
                    processed_array = detune_sample(path)
                    
                    # Update the samples and play the processed sound
                    samples[i] = processed_array
                    sound = array_to_pygame_format(processed_array)
                    channels[i].stop()
                    channels[i].play(sound, -1)
            elif event.key == pygame.K_g:  # Granular synthesis
                GRANULAR_MODE = not GRANULAR_MODE
                for i, sample in enumerate(samples):
                    grain_count = random.randint(1, 30)  # You can also use random.uniform(1, 30) for floating numbers.
                    granulated_array = apply_granular_effect(sample, grain_count)
                    
                    # Update the samples and play the granulated sound
                    samples[i] = granulated_array
                    sound = array_to_pygame_format(granulated_array)
                    channels[i].stop()
                    channels[i].play(sound, -1)
            elif event.key == pygame.K_f:  # Generate a random trajectory
                trajectory = Trajectory()
            elif event.key == pygame.K_o:  # Oscillate/Resynth effect
                processed_samples = oscillate_samples(samples)
                
                # Stop currently playing samples
                for channel in channels:
                    channel.stop()

                # Reload sounds in pygame with oscillated samples
                for i, processed_sample_array in enumerate(processed_samples):
                    sound = array_to_pygame_format(processed_sample_array)  # Convert numpy array to pygame Sound object
                    channels[i].play(sound, -1)

    if trajectory:
        x, y = trajectory.update(dt, move_speed)
        set_channel_volumes(x, y)

    pygame.draw.circle(screen, (0, 0, 255), (x, y), 10)
    pygame.display.flip()

pygame.quit()

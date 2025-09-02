#!/usr/bin/env python3
"""
Generate MUSE demo audio files (WAV and MIDI)
Showcases golden ratio melodies and Fibonacci rhythms
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.muse.core.predictive_music_engine import PredictiveMusicEngine, FrequencySignature
import numpy as np
import wave
import struct
import io
from mido import Message, MidiFile, MidiTrack, bpm2tempo

def generate_wav_demo(output_file="muse_golden_ratio_demo.wav"):
    """Generate a WAV file demonstrating golden ratio melody"""
    print("🎵 Generating golden ratio melody WAV...")
    
    engine = PredictiveMusicEngine()
    
    # Generate a golden ratio melody starting at A440
    melody_freqs = engine.golden_ratio_melody(440.0, length=12)
    
    # Convert to tuples with duration (since golden_ratio_melody returns just frequencies)
    melody = [(freq, 0.5) for freq in melody_freqs]  # 0.5 beats per note
    
    # Audio parameters
    framerate = 44100
    duration_per_note = 0.4  # seconds
    
    # Synthesize the melody
    audio_data = []
    
    for i, (freq, duration) in enumerate(melody):
        # Create note samples
        t = np.linspace(0, duration_per_note, int(framerate * duration_per_note), False)
        
        # Use a richer sound with harmonics
        fundamental = 0.4 * np.sin(2 * np.pi * freq * t)
        second_harmonic = 0.2 * np.sin(4 * np.pi * freq * t)
        third_harmonic = 0.1 * np.sin(6 * np.pi * freq * t)
        
        # Add envelope (ADSR)
        envelope = np.ones_like(t)
        attack_samples = int(0.05 * framerate)
        decay_samples = int(0.1 * framerate)
        release_samples = int(0.05 * framerate)
        
        # Attack
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        # Decay
        envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, 0.7, decay_samples)
        # Release
        envelope[-release_samples:] = np.linspace(0.7, 0, release_samples)
        
        # Combine harmonics with envelope
        wave_data = envelope * (fundamental + second_harmonic + third_harmonic)
        audio_data.extend(wave_data)
        
        # Add a tiny gap between notes
        silence = np.zeros(int(0.05 * framerate))
        audio_data.extend(silence)
    
    # Normalize to prevent clipping
    audio_data = np.array(audio_data)
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
    
    # Write WAV file
    with wave.open(output_file, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(framerate)
        
        for sample in audio_data:
            f.writeframes(struct.pack('h', int(sample * 32767)))
    
    print(f"✅ WAV saved: {output_file} ({len(audio_data)/framerate:.1f} seconds)")
    return output_file

def generate_midi_demo(output_file="muse_fibonacci_rhythm_demo.mid"):
    """Generate a MIDI file demonstrating Fibonacci rhythms"""
    print("🎹 Generating Fibonacci rhythm MIDI...")
    
    engine = PredictiveMusicEngine()
    
    # Generate Fibonacci rhythm pattern (returns first 8 values)
    fibonacci_rhythm = engine.fibonacci_rhythm(base_duration=0.5)
    
    # Generate a scale for the melody (EUTERPE mode - Mixolydian)
    from backend.muse.core.predictive_music_engine import ScaleMode
    scale = engine.generate_scale(440.0, ScaleMode.MIXOLYDIAN, octaves=2)
    
    # Create phrase by combining scale notes with Fibonacci rhythms
    # Extend rhythm pattern if needed
    phrase = []
    for i in range(16):
        duration = fibonacci_rhythm[i % len(fibonacci_rhythm)]
        note_freq = scale[i % len(scale)]
        phrase.append((note_freq, duration))
    
    # Create MIDI file
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    tempo_bpm = 108  # Golden ratio tempo
    track.append(Message('program_change', program=0, time=0))  # Piano
    from mido import MetaMessage
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm)))
    
    # Convert frequencies to MIDI notes
    def freq_to_midi(freq):
        """Convert frequency to MIDI note number"""
        if freq <= 0:
            return 60  # Default to middle C
        return int(round(69 + 12 * np.log2(freq / 440.0)))
    
    ticks_per_beat = mid.ticks_per_beat
    
    # Add notes to track
    for freq, duration_beats in phrase:
        midi_note = max(0, min(127, freq_to_midi(freq)))
        velocity = 80 + int(20 * np.random.random())  # Slight velocity variation
        
        # Note on
        track.append(Message('note_on', note=midi_note, velocity=velocity, time=0))
        
        # Note off after duration
        ticks = int(duration_beats * ticks_per_beat)
        track.append(Message('note_off', note=midi_note, velocity=0, time=ticks))
    
    # Save MIDI file
    mid.save(output_file)
    
    print(f"✅ MIDI saved: {output_file}")
    return output_file

def generate_comparison_demo():
    """Generate a special demo comparing golden ratio vs equal temperament"""
    print("\n🔬 Generating comparison demo...")
    
    engine = PredictiveMusicEngine()
    
    # Golden ratio intervals
    golden_freqs = engine.golden_ratio_melody(440.0, length=8)
    golden_melody = [(freq, 0.5) for freq in golden_freqs]
    
    # Equal temperament scale (for comparison)
    equal_temp_scale = [440.0 * (2 ** (i/12)) for i in range(8)]
    equal_melody = [(freq, 0.5) for freq in equal_temp_scale]
    
    # Generate both as WAV
    framerate = 44100
    duration_per_note = 0.3
    
    def synthesize_melody(melody, label):
        audio_data = []
        for freq, _ in melody:
            t = np.linspace(0, duration_per_note, int(framerate * duration_per_note), False)
            wave_data = 0.5 * np.sin(2 * np.pi * freq * t)
            
            # Simple envelope
            envelope = np.ones_like(t)
            envelope[:100] = np.linspace(0, 1, 100)
            envelope[-100:] = np.linspace(1, 0, 100)
            
            audio_data.extend(envelope * wave_data)
            audio_data.extend(np.zeros(int(0.1 * framerate)))  # Gap
        
        return np.array(audio_data)
    
    # Synthesize both
    golden_audio = synthesize_melody(golden_melody, "Golden Ratio")
    equal_audio = synthesize_melody(equal_melody, "Equal Temperament")
    
    # Add silence between them
    silence = np.zeros(int(1.0 * framerate))
    
    # Combine: golden ratio, silence, equal temperament
    combined = np.concatenate([golden_audio, silence, equal_audio])
    combined = combined / np.max(np.abs(combined)) * 0.8
    
    # Save
    output_file = "muse_comparison_demo.wav"
    with wave.open(output_file, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(framerate)
        
        for sample in combined:
            f.writeframes(struct.pack('h', int(sample * 32767)))
    
    print(f"✅ Comparison saved: {output_file}")
    print("   First half: Golden Ratio melody (natural, flowing)")
    print("   Second half: Equal Temperament scale (mechanical, rigid)")
    
    return output_file

if __name__ == "__main__":
    print("=" * 60)
    print("🎭 MUSE Platform - Demo Audio Generator")
    print("=" * 60)
    
    # Generate all demos
    try:
        wav_file = generate_wav_demo()
        midi_file = generate_midi_demo()
        comparison_file = generate_comparison_demo()
        
        print("\n" + "=" * 60)
        print("✨ All demos generated successfully!")
        print("=" * 60)
        print("\nFiles created:")
        print(f"  • {wav_file} - Golden ratio melody")
        print(f"  • {midi_file} - Fibonacci rhythm patterns")
        print(f"  • {comparison_file} - Golden vs Equal Temperament")
        print("\nShare these with your reviewer to demonstrate the")
        print("mathematical beauty of MUSE's music generation!")
        
    except Exception as e:
        print(f"\n❌ Error generating demos: {e}")
        print("Make sure you're in the muse-platform directory and")
        print("have installed all requirements:")
        print("  pip install numpy mido")
        sys.exit(1)
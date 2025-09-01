/**
 * MUSE Sound Engine Component
 * 
 * Real-time audio synthesis based on frequency signatures
 * Uses Web Audio API to generate actual sound from mathematical patterns
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Play, Pause, Volume2, Zap, Music } from 'lucide-react';
import { useMuseStore } from '../stores/useMuseStore';
import { useMuseAPI } from '../hooks/useMuseAPI';

interface MusicalPhrase {
  notes: number[];
  durations: number[];
  dynamics: number[];
  archetype: string;
  sacred_ratio: number;
}

interface OscillatorNode extends AudioScheduledSourceNode {
  frequency: AudioParam;
  type: OscillatorType;
}

export const MuseSoundEngine: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [currentNoteIndex, setCurrentNoteIndex] = useState(0);
  const [waveform, setWaveform] = useState<OscillatorType>('sine');
  const [tempo, setTempo] = useState(120); // BPM
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const oscillatorsRef = useRef<Map<string, OscillatorNode>>(new Map());
  const analyserRef = useRef<AnalyserNode | null>(null);
  const visualizerCanvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  const { frequencySignature } = useMuseStore();
  const { generateMusicalPhrase } = useMuseAPI();
  
  const [currentPhrase, setCurrentPhrase] = useState<MusicalPhrase | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Initialize Web Audio API
  useEffect(() => {
    if (typeof window !== 'undefined' && !audioContextRef.current) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      audioContextRef.current = new AudioContextClass();
      
      // Create master gain node
      masterGainRef.current = audioContextRef.current.createGain();
      masterGainRef.current.gain.value = volume;
      masterGainRef.current.connect(audioContextRef.current.destination);
      
      // Create analyser for visualization
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      analyserRef.current.connect(masterGainRef.current);
    }
    
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Update master volume
  useEffect(() => {
    if (masterGainRef.current) {
      masterGainRef.current.gain.value = volume;
    }
  }, [volume]);

  // Sacred geometry oscillator creation
  const createSacredOscillator = useCallback((
    frequency: number,
    startTime: number,
    duration: number,
    dynamics: number
  ): OscillatorNode | null => {
    if (!audioContextRef.current || !analyserRef.current) return null;
    
    const oscillator = audioContextRef.current.createOscillator();
    const gainNode = audioContextRef.current.createGain();
    
    // Apply sacred geometry to waveform
    oscillator.type = waveform;
    oscillator.frequency.value = frequency;
    
    // Golden ratio envelope
    const phi = 1.618033988749895;
    const attackTime = duration / (phi * 2);
    const decayTime = duration / phi;
    
    // ADSR envelope
    gainNode.gain.setValueAtTime(0, startTime);
    gainNode.gain.linearRampToValueAtTime(dynamics, startTime + attackTime);
    gainNode.gain.exponentialRampToValueAtTime(dynamics * 0.6, startTime + decayTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, startTime + duration);
    
    // Connect nodes
    oscillator.connect(gainNode);
    gainNode.connect(analyserRef.current);
    
    // Schedule playback
    oscillator.start(startTime);
    oscillator.stop(startTime + duration);
    
    return oscillator;
  }, [waveform]);

  // Generate new musical phrase
  const handleGeneratePhrase = useCallback(async () => {
    if (!frequencySignature) {
      console.warn('No frequency signature available');
      return;
    }
    
    setIsGenerating(true);
    try {
      const phrase = await generateMusicalPhrase(frequencySignature);
      setCurrentPhrase(phrase);
      setCurrentNoteIndex(0);
    } catch (error) {
      console.error('Failed to generate phrase:', error);
    } finally {
      setIsGenerating(false);
    }
  }, [frequencySignature, generateMusicalPhrase]);

  // Play the current phrase
  const playPhrase = useCallback(() => {
    if (!currentPhrase || !audioContextRef.current) return;
    
    const now = audioContextRef.current.currentTime;
    let currentTime = now;
    
    // Clear previous oscillators
    oscillatorsRef.current.forEach(osc => {
      try {
        osc.stop();
      } catch (e) {
        // Oscillator already stopped
      }
    });
    oscillatorsRef.current.clear();
    
    // Schedule all notes
    currentPhrase.notes.forEach((frequency, index) => {
      const duration = currentPhrase.durations[index] * (60 / tempo);
      const dynamics = currentPhrase.dynamics[index];
      
      const oscillator = createSacredOscillator(
        frequency,
        currentTime,
        duration,
        dynamics
      );
      
      if (oscillator) {
        oscillatorsRef.current.set(`note-${index}`, oscillator);
        
        // Update current note index for visualization
        setTimeout(() => {
          setCurrentNoteIndex(index);
        }, (currentTime - now) * 1000);
      }
      
      currentTime += duration;
    });
    
    // Stop playing after all notes are done
    setTimeout(() => {
      setIsPlaying(false);
    }, (currentTime - now) * 1000);
  }, [currentPhrase, tempo, createSacredOscillator]);

  // Toggle playback
  const togglePlayback = useCallback(() => {
    if (!audioContextRef.current) return;
    
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
    
    if (isPlaying) {
      // Stop all oscillators
      oscillatorsRef.current.forEach(osc => {
        try {
          osc.stop();
        } catch (e) {
          // Already stopped
        }
      });
      oscillatorsRef.current.clear();
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      playPhrase();
    }
  }, [isPlaying, playPhrase]);

  // Visualizer animation
  useEffect(() => {
    if (!analyserRef.current || !visualizerCanvasRef.current) return;
    
    const canvas = visualizerCanvasRef.current;
    const canvasContext = canvas.getContext('2d');
    if (!canvasContext) return;
    
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw);
      
      analyser.getByteFrequencyData(dataArray);
      
      canvasContext.fillStyle = 'rgb(20, 20, 30)';
      canvasContext.fillRect(0, 0, canvas.width, canvas.height);
      
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;
      
      // Apply golden ratio to visualization
      const phi = 1.618033988749895;
      
      for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArray[i] / 255) * canvas.height * 0.8;
        
        // Color based on frequency range with golden ratio
        const hue = (i / bufferLength) * 360 / phi;
        canvasContext.fillStyle = `hsl(${hue}, 70%, 50%)`;
        
        canvasContext.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
      }
    };
    
    draw();
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying]);

  // Play single frequency (for testing)
  const playFrequency = useCallback((frequency: number) => {
    if (!audioContextRef.current || !analyserRef.current) return;
    
    const oscillator = audioContextRef.current.createOscillator();
    const gainNode = audioContextRef.current.createGain();
    
    oscillator.frequency.value = frequency;
    oscillator.type = waveform;
    
    gainNode.gain.value = volume * 0.5;
    
    oscillator.connect(gainNode);
    gainNode.connect(analyserRef.current);
    
    oscillator.start();
    oscillator.stop(audioContextRef.current.currentTime + 0.5);
  }, [waveform, volume]);

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Music className="w-6 h-6" />
          MUSE Sound Engine
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Visualizer */}
        <div className="relative bg-gray-900 rounded-lg overflow-hidden">
          <canvas
            ref={visualizerCanvasRef}
            width={800}
            height={200}
            className="w-full h-48"
          />
          {currentPhrase && (
            <div className="absolute top-2 left-2 text-white text-sm bg-black/50 px-2 py-1 rounded">
              Note {currentNoteIndex + 1} / {currentPhrase.notes.length}
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Playback Controls */}
          <div className="space-y-4">
            <div className="flex gap-2">
              <Button
                onClick={handleGeneratePhrase}
                disabled={isGenerating || !frequencySignature}
                className="flex-1"
              >
                <Zap className="w-4 h-4 mr-2" />
                {isGenerating ? 'Generating...' : 'Generate Phrase'}
              </Button>
              <Button
                onClick={togglePlayback}
                disabled={!currentPhrase}
                variant={isPlaying ? 'destructive' : 'default'}
              >
                {isPlaying ? (
                  <Pause className="w-4 h-4" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
              </Button>
            </div>

            {/* Volume Control */}
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <Volume2 className="w-4 h-4" />
                Volume: {Math.round(volume * 100)}%
              </label>
              <Slider
                value={[volume]}
                onValueChange={([v]) => setVolume(v)}
                min={0}
                max={1}
                step={0.01}
                className="w-full"
              />
            </div>

            {/* Tempo Control */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Tempo: {tempo} BPM
              </label>
              <Slider
                value={[tempo]}
                onValueChange={([t]) => setTempo(t)}
                min={60}
                max={180}
                step={1}
                className="w-full"
              />
            </div>
          </div>

          {/* Waveform & Testing */}
          <div className="space-y-4">
            {/* Waveform Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Waveform</label>
              <div className="grid grid-cols-2 gap-2">
                {(['sine', 'square', 'sawtooth', 'triangle'] as OscillatorType[]).map(type => (
                  <Button
                    key={type}
                    size="sm"
                    variant={waveform === type ? 'default' : 'outline'}
                    onClick={() => setWaveform(type)}
                  >
                    {type}
                  </Button>
                ))}
              </div>
            </div>

            {/* Test Frequencies */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Test Sacred Frequencies</label>
              <div className="grid grid-cols-3 gap-2">
                <Button size="sm" variant="outline" onClick={() => playFrequency(432)}>
                  432 Hz
                </Button>
                <Button size="sm" variant="outline" onClick={() => playFrequency(528)}>
                  528 Hz
                </Button>
                <Button size="sm" variant="outline" onClick={() => playFrequency(639)}>
                  639 Hz
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Phrase Info */}
        {currentPhrase && (
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium mb-2">Current Phrase</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              <div>
                <span className="text-gray-500">Archetype:</span>{' '}
                <span className="font-medium">{currentPhrase.archetype}</span>
              </div>
              <div>
                <span className="text-gray-500">Sacred Ratio:</span>{' '}
                <span className="font-medium">{currentPhrase.sacred_ratio.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-gray-500">Notes:</span>{' '}
                <span className="font-medium">{currentPhrase.notes.length}</span>
              </div>
              <div>
                <span className="text-gray-500">Duration:</span>{' '}
                <span className="font-medium">
                  {currentPhrase.durations.reduce((a, b) => a + b, 0).toFixed(1)}s
                </span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
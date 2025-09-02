/**
 * MUSE Magic Button - Instant Music Creation
 * 
 * Zero-friction entry point that creates beautiful music immediately
 * while progressively revealing the mathematical depth underneath
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Music, Heart, Brain, Zap } from 'lucide-react';

interface UserAura {
  timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  mouseVelocity: number;
  clickPosition: { x: number; y: number };
  seasonalEnergy: number;
  browserEntropy: string;
}

interface Props {
  onMusicCreated?: () => void;
}

export const MuseMagicButton: React.FC<Props> = ({ onMusicCreated }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [hasCreated, setHasCreated] = useState(false);
  const [currentPhase, setCurrentPhase] = useState<'idle' | 'discovering' | 'creating' | 'playing'>('idle');
  const [userEnergy, setUserEnergy] = useState(0);
  const [particles, setParticles] = useState<Array<{id: number, x: number, y: number}>>([]);
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const mouseTrailRef = useRef<Array<{x: number, y: number, time: number}>>([]);
  
  // Detect user's "aura" from environmental and behavioral signals
  const detectUserAura = useCallback((): UserAura => {
    const hour = new Date().getHours();
    const timeOfDay = 
      hour < 6 ? 'night' :
      hour < 12 ? 'morning' :
      hour < 17 ? 'afternoon' :
      hour < 21 ? 'evening' : 'night';
    
    // Calculate mouse velocity from trail
    const velocity = mouseTrailRef.current.length > 1 
      ? Math.sqrt(
          Math.pow(mouseTrailRef.current[mouseTrailRef.current.length-1].x - mouseTrailRef.current[0].x, 2) +
          Math.pow(mouseTrailRef.current[mouseTrailRef.current.length-1].y - mouseTrailRef.current[0].y, 2)
        ) / mouseTrailRef.current.length
      : 0.5;
    
    // Seasonal energy based on month
    const month = new Date().getMonth();
    const seasonalEnergy = Math.sin((month / 12) * Math.PI * 2) * 0.5 + 0.5;
    
    // Generate browser-based entropy
    const entropy = btoa(String(Date.now() + Math.random())).substring(0, 8);
    
    return {
      timeOfDay,
      mouseVelocity: velocity,
      clickPosition: mouseTrailRef.current[mouseTrailRef.current.length - 1] || {x: 0.5, y: 0.5},
      seasonalEnergy,
      browserEntropy: entropy
    };
  }, []);
  
  // Map aura to musical parameters WITHOUT exposing complexity
  const auraToMusicalParams = (aura: UserAura) => {
    const energyMap = {
      'morning': { archetype: 'EUTERPE', energy: 0.7, tempo: 120 },
      'afternoon': { archetype: 'THALIA', energy: 0.8, tempo: 140 },
      'evening': { archetype: 'ERATO', energy: 0.5, tempo: 90 },
      'night': { archetype: 'PSYCHE', energy: 0.3, tempo: 60 }
    };
    
    const params = energyMap[aura.timeOfDay];
    
    return {
      ...params,
      complexity: Math.min(aura.mouseVelocity, 1),
      emotionalTone: aura.clickPosition.y, // Higher = happier
      spatialWidth: aura.clickPosition.x,  // Left/right = narrow/wide
      uniqueSeed: aura.browserEntropy
    };
  };
  
  // Create instant musical magic
  const createInstantMagic = async () => {
    setIsGenerating(true);
    setCurrentPhase('discovering');
    
    // Initialize audio context on user interaction
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    
    // Detect user's current state
    const aura = detectUserAura();
    const musicalParams = auraToMusicalParams(aura);
    
    // Visual feedback - spawn particles
    const newParticles = Array.from({length: 20}, (_, i) => ({
      id: Date.now() + i,
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight
    }));
    setParticles(newParticles);
    
    // Simulate discovery phase
    await new Promise(resolve => setTimeout(resolve, 1500));
    setCurrentPhase('creating');
    
    // Generate and play music
    await playMagicalMusic(musicalParams);
    
    setCurrentPhase('playing');
    setIsGenerating(false);
    setHasCreated(true);
    
    // Notify parent component
    if (onMusicCreated) {
      onMusicCreated();
    }
    
    // Clean up particles after animation
    setTimeout(() => setParticles([]), 3000);
  };
  
  // Play the discovered music with rich, pleasant sounds
  const playMagicalMusic = async (params: any) => {
    if (!audioContextRef.current) return;
    
    const ctx = audioContextRef.current;
    const now = ctx.currentTime;
    
    // Create a rich, layered sound (not just sine waves)
    const createRichOscillator = (freq: number, startTime: number, duration: number) => {
      // Main tone
      const osc1 = ctx.createOscillator();
      osc1.frequency.value = freq;
      osc1.type = 'sine';
      
      // Harmonic richness
      const osc2 = ctx.createOscillator();
      osc2.frequency.value = freq * 2;
      osc2.type = 'triangle';
      
      // Sub bass
      const osc3 = ctx.createOscillator();
      osc3.frequency.value = freq / 2;
      osc3.type = 'sine';
      
      // Gains for mixing
      const gain1 = ctx.createGain();
      const gain2 = ctx.createGain();
      const gain3 = ctx.createGain();
      const masterGain = ctx.createGain();
      
      gain1.gain.value = 0.5;
      gain2.gain.value = 0.2;
      gain3.gain.value = 0.3;
      
      // ADSR envelope
      masterGain.gain.setValueAtTime(0, startTime);
      masterGain.gain.linearRampToValueAtTime(0.3, startTime + 0.05); // Attack
      masterGain.gain.exponentialRampToValueAtTime(0.2, startTime + 0.2); // Decay
      masterGain.gain.exponentialRampToValueAtTime(0.01, startTime + duration); // Release
      
      // Connect
      osc1.connect(gain1);
      osc2.connect(gain2);
      osc3.connect(gain3);
      gain1.connect(masterGain);
      gain2.connect(masterGain);
      gain3.connect(masterGain);
      masterGain.connect(ctx.destination);
      
      // Add reverb simulation with delay
      const delay = ctx.createDelay();
      delay.delayTime.value = 0.03;
      const delayGain = ctx.createGain();
      delayGain.gain.value = 0.3;
      masterGain.connect(delay);
      delay.connect(delayGain);
      delayGain.connect(ctx.destination);
      
      // Schedule
      osc1.start(startTime);
      osc2.start(startTime);
      osc3.start(startTime);
      osc1.stop(startTime + duration);
      osc2.stop(startTime + duration);
      osc3.stop(startTime + duration);
    };
    
    // Generate a pleasant melody based on parameters
    const baseFreq = 440 * (0.5 + params.emotionalTone); // 220-660 Hz range
    const scale = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8]; // Major scale ratios
    
    // Play an ascending, hopeful phrase
    for (let i = 0; i < 8; i++) {
      const freq = baseFreq * scale[i % scale.length];
      const startTime = now + (i * 0.3);
      const duration = 0.5 + (Math.sin(i) * 0.2);
      
      createRichOscillator(freq, startTime, duration);
    }
    
    // Add a resolution chord at the end
    setTimeout(() => {
      const chordTime = now + 2.5;
      createRichOscillator(baseFreq, chordTime, 1.5);
      createRichOscillator(baseFreq * 5/4, chordTime, 1.5); // Major third
      createRichOscillator(baseFreq * 3/2, chordTime, 1.5); // Perfect fifth
    }, 2400);
  };
  
  // Track mouse movement for aura detection
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const x = e.clientX / window.innerWidth;
      const y = e.clientY / window.innerHeight;
      
      mouseTrailRef.current.push({ x, y, time: Date.now() });
      
      // Keep only last 10 positions
      if (mouseTrailRef.current.length > 10) {
        mouseTrailRef.current.shift();
      }
      
      // Update energy visualization
      setUserEnergy(Math.sqrt(x * x + y * y) / Math.sqrt(2));
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);
  
  return (
    <div className="relative min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 overflow-hidden">
      {/* Animated background particles */}
      <AnimatePresence>
        {particles.map(particle => (
          <motion.div
            key={particle.id}
            className="absolute w-2 h-2 bg-yellow-300 rounded-full opacity-60"
            initial={{ x: particle.x, y: particle.y, scale: 0 }}
            animate={{ 
              x: particle.x + (Math.random() - 0.5) * 200,
              y: particle.y - 200,
              scale: [0, 1.5, 0],
              opacity: [0, 1, 0]
            }}
            exit={{ opacity: 0 }}
            transition={{ duration: 3, ease: "easeOut" }}
          />
        ))}
      </AnimatePresence>
      
      {/* Energy field visualization */}
      <div 
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `radial-gradient(circle at 50% 50%, 
            rgba(147, 51, 234, ${userEnergy * 0.2}) 0%, 
            transparent 70%)`
        }}
      />
      
      {/* Main content */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4">
        {/* Title with dynamic glow */}
        <motion.h1 
          className="text-5xl md:text-7xl font-thin text-white mb-4 text-center"
          animate={{ 
            textShadow: `0 0 ${20 + userEnergy * 30}px rgba(147, 51, 234, 0.5)` 
          }}
        >
          {hasCreated ? 'Your Music Lives' : 'Discover Your Sound'}
        </motion.h1>
        
        {/* Subtitle */}
        <p className="text-xl text-purple-200 mb-12 text-center max-w-md">
          {hasCreated 
            ? 'Every movement creates a new universe of sound'
            : 'One touch reveals the music within you'}
        </p>
        
        {/* The Magic Button */}
        <motion.button
          onClick={createInstantMagic}
          disabled={isGenerating}
          className={`
            relative px-8 py-4 md:px-12 md:py-6 rounded-full
            text-lg md:text-xl font-medium transition-all duration-300
            ${isGenerating 
              ? 'bg-purple-800 text-purple-300 cursor-wait' 
              : 'bg-gradient-to-r from-yellow-400 to-orange-500 text-black hover:scale-105 hover:shadow-2xl hover:shadow-yellow-500/25'
            }
          `}
          whileHover={!isGenerating ? { scale: 1.05 } : {}}
          whileTap={!isGenerating ? { scale: 0.95 } : {}}
        >
          {/* Button content */}
          <span className="relative z-10 flex items-center gap-3">
            {currentPhase === 'idle' && (
              <>
                <Sparkles className="w-6 h-6" />
                {hasCreated ? 'Create Another' : 'Create My First Song'}
              </>
            )}
            {currentPhase === 'discovering' && (
              <>
                <Brain className="w-6 h-6 animate-pulse" />
                Reading Your Aura...
              </>
            )}
            {currentPhase === 'creating' && (
              <>
                <Zap className="w-6 h-6 animate-bounce" />
                Discovering Music...
              </>
            )}
            {currentPhase === 'playing' && (
              <>
                <Music className="w-6 h-6 animate-pulse" />
                Playing Your Song
              </>
            )}
          </span>
          
          {/* Animated ring */}
          {isGenerating && (
            <motion.div
              className="absolute inset-0 rounded-full border-4 border-purple-400"
              initial={{ scale: 1, opacity: 1 }}
              animate={{ scale: 1.5, opacity: 0 }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
          )}
        </motion.button>
        
        {/* Progressive complexity reveal */}
        {hasCreated && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-12 text-center"
          >
            <p className="text-purple-300 mb-4">
              Want to understand the magic?
            </p>
            <button className="text-yellow-400 hover:text-yellow-300 underline">
              Show me how it works →
            </button>
          </motion.div>
        )}
      </div>
    </div>
  );
};
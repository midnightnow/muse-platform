/**
 * MUSE Magic Landing Page - Zero-Friction Entry Point
 * 
 * The new landing experience that prioritizes instant gratification
 * and progressive complexity disclosure.
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { MuseMagicButton } from '../components/MuseMagicButton';
import { MusePersonalityQuiz } from '../components/MusePersonalityQuiz';
import { Music, Sparkles, Zap, Heart, Code } from 'lucide-react';

type ViewMode = 'magic' | 'quiz' | 'dashboard';

export const MagicLandingPage: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('magic');
  const [userArchetype, setUserArchetype] = useState<string | null>(null);
  const [userFrequency, setUserFrequency] = useState<number | null>(null);
  const [hasCreatedFirstSong, setHasCreatedFirstSong] = useState(false);
  const navigate = useNavigate();

  const handleQuizComplete = (archetype: string, frequency: number, scores: Record<string, number>) => {
    setUserArchetype(archetype);
    setUserFrequency(frequency);
    
    // Store in localStorage for persistence
    localStorage.setItem('muse_archetype', archetype);
    localStorage.setItem('muse_frequency', String(frequency));
    localStorage.setItem('muse_scores', JSON.stringify(scores));
    
    // Navigate to dashboard after a brief delay
    setTimeout(() => {
      navigate('/dashboard');
    }, 2000);
  };

  const handleExploreMore = () => {
    setViewMode('quiz');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-purple-950 to-black">
      <AnimatePresence mode="wait">
        {viewMode === 'magic' && (
          <motion.div
            key="magic"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative"
          >
            {/* Magic Button View */}
            <MuseMagicButton onMusicCreated={() => setHasCreatedFirstSong(true)} />
            
            {/* Progressive Options */}
            <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 space-y-3">
              {/* Show playground option after first song */}
              {hasCreatedFirstSong && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="text-center"
                >
                  <button
                    onClick={() => navigate('/playground')}
                    className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full text-white hover:from-purple-600 hover:to-pink-600 transition-all duration-300 flex items-center gap-2 shadow-lg hover:shadow-xl"
                  >
                    <Code className="w-4 h-4" />
                    Explore the Sacred Geometry
                  </button>
                </motion.div>
              )}
              
              {/* Archetype discovery option */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: hasCreatedFirstSong ? 0.5 : 2 }}
              >
                <button
                  onClick={handleExploreMore}
                  className="px-6 py-3 bg-white/10 backdrop-blur-sm rounded-full text-white hover:bg-white/20 transition-all duration-300 flex items-center gap-2"
                >
                  <Sparkles className="w-4 h-4" />
                  Discover Your Musical Archetype
                </button>
              </motion.div>
            </div>
            
            {/* Floating testimonials */}
            <div className="fixed top-8 right-8 max-w-xs space-y-4 pointer-events-none">
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 3 }}
                className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10"
              >
                <p className="text-sm text-white/70 italic">
                  "I never thought I could create music. This changed everything."
                </p>
                <p className="text-xs text-white/50 mt-2">- Sarah, first-time creator</p>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 4 }}
                className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10"
              >
                <p className="text-sm text-white/70 italic">
                  "The mathematics behind the music is fascinating!"
                </p>
                <p className="text-xs text-white/50 mt-2">- Alex, mathematician</p>
              </motion.div>
            </div>
            
            {/* Stats overlay */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 5 }}
              className="fixed bottom-8 right-8 flex gap-8 text-white/50 text-sm"
            >
              <div className="flex items-center gap-2">
                <Music className="w-4 h-4" />
                <span>12,847 songs created</span>
              </div>
              <div className="flex items-center gap-2">
                <Heart className="w-4 h-4" />
                <span>98% love it</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4" />
                <span>< 60s to first song</span>
              </div>
            </motion.div>
          </motion.div>
        )}
        
        {viewMode === 'quiz' && (
          <motion.div
            key="quiz"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center justify-center min-h-screen"
          >
            <div className="w-full max-w-4xl">
              {/* Quiz Header */}
              <motion.div
                initial={{ y: -20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className="text-center mb-8"
              >
                <h1 className="text-4xl font-thin text-white mb-2">
                  Discover Your Musical DNA
                </h1>
                <p className="text-gray-400">
                  Answer a few simple questions to unlock your unique sound
                </p>
              </motion.div>
              
              {/* Quiz Component */}
              <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-white/10 p-8">
                <MusePersonalityQuiz onComplete={handleQuizComplete} />
              </div>
              
              {/* Skip Option */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1 }}
                className="text-center mt-6"
              >
                <button
                  onClick={() => navigate('/dashboard')}
                  className="text-gray-500 hover:text-gray-300 text-sm underline"
                >
                  Skip for now
                </button>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Background ambient animation */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0">
          {Array.from({ length: 50 }).map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-purple-400/20 rounded-full"
              initial={{
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
              }}
              animate={{
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
              }}
              transition={{
                duration: 20 + Math.random() * 20,
                repeat: Infinity,
                ease: "linear",
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default MagicLandingPage;
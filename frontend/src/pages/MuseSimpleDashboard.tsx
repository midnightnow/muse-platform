/**
 * MUSE Simple Dashboard - Post-Magic Button/Quiz Experience
 * 
 * Shows the user their discovered archetype and provides simple
 * next steps for continued music creation.
 */

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { 
  Sparkles, 
  Music, 
  Users, 
  Play, 
  Zap, 
  Heart,
  Share2,
  Download,
  ChevronRight
} from 'lucide-react';

const archetypeInfo: Record<string, {
  color: string;
  emoji: string;
  description: string;
  traits: string[];
  musicalStyle: string;
}> = {
  CALLIOPE: {
    color: 'from-purple-500 to-indigo-600',
    emoji: '📜',
    description: 'Epic storyteller, voice of heroes',
    traits: ['Bold', 'Inspiring', 'Dramatic'],
    musicalStyle: 'Epic orchestral, heroic themes'
  },
  CLIO: {
    color: 'from-amber-500 to-orange-600',
    emoji: '📚',
    description: 'Keeper of history, wisdom of time',
    traits: ['Wise', 'Thoughtful', 'Timeless'],
    musicalStyle: 'Classical, contemplative melodies'
  },
  ERATO: {
    color: 'from-pink-500 to-rose-600',
    emoji: '💕',
    description: 'Romantic soul, love\'s melody',
    traits: ['Passionate', 'Tender', 'Emotional'],
    musicalStyle: 'Romantic ballads, soft harmonies'
  },
  EUTERPE: {
    color: 'from-green-500 to-emerald-600',
    emoji: '🎵',
    description: 'Musical joy, nature\'s harmony',
    traits: ['Joyful', 'Natural', 'Harmonious'],
    musicalStyle: 'Folk melodies, nature sounds'
  },
  MELPOMENE: {
    color: 'from-red-500 to-red-700',
    emoji: '🎭',
    description: 'Deep emotions, transformative power',
    traits: ['Intense', 'Transformative', 'Powerful'],
    musicalStyle: 'Dramatic minor keys, emotional depth'
  },
  POLYHYMNIA: {
    color: 'from-violet-500 to-purple-700',
    emoji: '🙏',
    description: 'Sacred expression, divine connection',
    traits: ['Spiritual', 'Sacred', 'Contemplative'],
    musicalStyle: 'Hymns, sacred music, meditation'
  },
  TERPSICHORE: {
    color: 'from-cyan-500 to-blue-600',
    emoji: '💃',
    description: 'Dance of life, rhythmic flow',
    traits: ['Rhythmic', 'Dynamic', 'Energetic'],
    musicalStyle: 'Dance beats, rhythmic patterns'
  },
  THALIA: {
    color: 'from-yellow-400 to-amber-500',
    emoji: '😊',
    description: 'Joyful creativity, playful spirit',
    traits: ['Playful', 'Creative', 'Lighthearted'],
    musicalStyle: 'Upbeat pop, playful melodies'
  },
  URANIA: {
    color: 'from-indigo-500 to-blue-700',
    emoji: '🌌',
    description: 'Cosmic wisdom, celestial patterns',
    traits: ['Cosmic', 'Mathematical', 'Visionary'],
    musicalStyle: 'Ambient space music, ethereal sounds'
  },
  MNEMOSYNE: {
    color: 'from-gray-500 to-slate-700',
    emoji: '💭',
    description: 'Memory keeper, ancestral wisdom',
    traits: ['Nostalgic', 'Wise', 'Reflective'],
    musicalStyle: 'Nostalgic themes, memory motifs'
  },
  PSYCHE: {
    color: 'from-purple-600 to-pink-600',
    emoji: '🦋',
    description: 'Soul\'s journey, inner light',
    traits: ['Intuitive', 'Soulful', 'Transformative'],
    musicalStyle: 'Soul music, inner journey themes'
  },
  SOPHIA: {
    color: 'from-white to-gray-300',
    emoji: '✨',
    description: 'Ultimate wisdom, pure understanding',
    traits: ['Wise', 'Pure', 'Enlightened'],
    musicalStyle: 'Crystalline tones, pure harmony'
  }
};

export const MuseSimpleDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [userArchetype, setUserArchetype] = useState<string>('EUTERPE');
  const [userFrequency, setUserFrequency] = useState<number>(594);
  const [songsCreated, setSongsCreated] = useState(0);
  const [showShare, setShowShare] = useState(false);
  
  useEffect(() => {
    // Load user's archetype from localStorage
    const archetype = localStorage.getItem('muse_archetype');
    const frequency = localStorage.getItem('muse_frequency');
    const songs = localStorage.getItem('muse_songs_created');
    
    if (archetype && frequency) {
      setUserArchetype(archetype);
      setUserFrequency(Number(frequency));
    }
    
    if (songs) {
      setSongsCreated(Number(songs));
    }
  }, []);
  
  const archetype = archetypeInfo[userArchetype] || archetypeInfo.EUTERPE;
  
  const handleCreateMore = () => {
    navigate('/');
  };
  
  const handleExploreDepth = () => {
    // Show the complex features progressively
    navigate('/discovery');
  };
  
  const handleShare = () => {
    setShowShare(true);
    // Copy to clipboard
    const shareText = `I'm a ${userArchetype} archetype on MUSE! My frequency is ${userFrequency}Hz. Create your musical DNA at muse.platform`;
    navigator.clipboard.writeText(shareText);
    setTimeout(() => setShowShare(false), 2000);
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-purple-950 to-black text-white">
      {/* Hero Section - Your Musical DNA */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center pt-20 pb-12 px-4"
      >
        <h1 className="text-5xl font-thin mb-2">Your Musical DNA</h1>
        <p className="text-gray-400 text-lg">
          Discovered through mathematical resonance
        </p>
      </motion.div>
      
      {/* Archetype Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        className="max-w-2xl mx-auto px-4 mb-12"
      >
        <div className={`bg-gradient-to-br ${archetype.color} p-1 rounded-2xl`}>
          <div className="bg-black/80 backdrop-blur rounded-2xl p-8">
            <div className="text-center mb-6">
              <span className="text-6xl mb-4 block">{archetype.emoji}</span>
              <h2 className="text-3xl font-medium mb-2">{userArchetype}</h2>
              <p className="text-lg opacity-90 mb-4">{archetype.description}</p>
              
              {/* Frequency Display */}
              <div className="inline-flex items-center gap-2 bg-white/10 rounded-full px-4 py-2 mb-6">
                <Zap className="w-4 h-4" />
                <span className="font-mono text-lg">{userFrequency} Hz</span>
              </div>
              
              {/* Traits */}
              <div className="flex justify-center gap-3 mb-6">
                {archetype.traits.map(trait => (
                  <span key={trait} className="px-3 py-1 bg-white/10 rounded-full text-sm">
                    {trait}
                  </span>
                ))}
              </div>
              
              {/* Musical Style */}
              <p className="text-sm opacity-75">
                <Music className="w-4 h-4 inline mr-2" />
                {archetype.musicalStyle}
              </p>
            </div>
            
            {/* Stats */}
            <div className="grid grid-cols-3 gap-4 pt-6 border-t border-white/20">
              <div className="text-center">
                <span className="text-2xl font-bold">{songsCreated}</span>
                <p className="text-sm opacity-75">Songs Created</p>
              </div>
              <div className="text-center">
                <span className="text-2xl font-bold">∞</span>
                <p className="text-sm opacity-75">Possibilities</p>
              </div>
              <div className="text-center">
                <span className="text-2xl font-bold">1</span>
                <p className="text-sm opacity-75">Unique You</p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
      
      {/* Action Buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="max-w-2xl mx-auto px-4 space-y-4"
      >
        {/* Primary Action - Create More Music */}
        <button
          onClick={handleCreateMore}
          className="w-full bg-gradient-to-r from-yellow-400 to-orange-500 text-black font-medium py-4 rounded-xl hover:scale-105 transition-transform flex items-center justify-center gap-3"
        >
          <Play className="w-5 h-5" />
          Create More Music
        </button>
        
        {/* Secondary Actions */}
        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={handleShare}
            className="bg-white/10 backdrop-blur py-3 rounded-xl hover:bg-white/20 transition-colors flex items-center justify-center gap-2"
          >
            <Share2 className="w-4 h-4" />
            Share DNA
          </button>
          
          <button
            className="bg-white/10 backdrop-blur py-3 rounded-xl hover:bg-white/20 transition-colors flex items-center justify-center gap-2"
          >
            <Download className="w-4 h-4" />
            Download Song
          </button>
        </div>
        
        {/* Explore Deeper */}
        <button
          onClick={handleExploreDepth}
          className="w-full bg-transparent border border-white/20 py-3 rounded-xl hover:bg-white/5 transition-colors flex items-center justify-center gap-2 group"
        >
          <Sparkles className="w-4 h-4" />
          Explore the Mathematics Behind Your Music
          <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
        </button>
      </motion.div>
      
      {/* Community Teaser */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="max-w-2xl mx-auto px-4 mt-12 mb-8"
      >
        <div className="bg-white/5 backdrop-blur rounded-xl p-6 text-center">
          <Users className="w-8 h-8 mx-auto mb-3 text-purple-400" />
          <h3 className="text-lg font-medium mb-2">Find Your Resonance Tribe</h3>
          <p className="text-sm opacity-75 mb-4">
            12,847 creators share your frequency range. Connect with kindred musical souls.
          </p>
          <button className="text-purple-400 hover:text-purple-300 text-sm underline">
            Explore Community →
          </button>
        </div>
      </motion.div>
      
      {/* Share Toast */}
      <AnimatePresence>
        {showShare && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-8 left-1/2 transform -translate-x-1/2 bg-green-500 text-white px-6 py-3 rounded-full shadow-lg"
          >
            ✓ Copied to clipboard!
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Subtle Footer */}
      <div className="text-center py-8 text-gray-600 text-sm">
        <p>MUSE · Mathematical Universal Sacred Expression</p>
        <p className="mt-1">Discovering pre-existing music in the Platonic realm</p>
      </div>
    </div>
  );
};

export default MuseSimpleDashboard;
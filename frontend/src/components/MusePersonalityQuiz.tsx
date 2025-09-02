/**
 * MUSE Personality Quiz - Simple Questions to Complex Archetypes
 * 
 * Maps everyday preferences and feelings to the 12 Muse archetypes
 * without requiring users to understand the underlying theory.
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronLeft, Sparkles } from 'lucide-react';

interface QuizQuestion {
  id: string;
  question: string;
  answers: Array<{
    text: string;
    archetypes: Record<string, number>; // Maps archetypes to weights
    emoji?: string;
  }>;
}

interface ArchetypeScore {
  archetype: string;
  score: number;
  frequency: number;
  color: string;
  description: string;
}

const QUIZ_QUESTIONS: QuizQuestion[] = [
  {
    id: 'energy',
    question: 'How are you feeling right now?',
    answers: [
      { 
        text: 'Energetic and ready to go', 
        emoji: '⚡',
        archetypes: { THALIA: 2, TERPSICHORE: 2, EUTERPE: 1 }
      },
      { 
        text: 'Calm and peaceful', 
        emoji: '🌊',
        archetypes: { ERATO: 2, PSYCHE: 2, MNEMOSYNE: 1 }
      },
      { 
        text: 'Thoughtful and reflective', 
        emoji: '🤔',
        archetypes: { CALLIOPE: 2, URANIA: 2, POLYHYMNIA: 1 }
      },
      { 
        text: 'Creative and inspired', 
        emoji: '✨',
        archetypes: { MELPOMENE: 2, CLIO: 1, SOPHIA: 1 }
      }
    ]
  },
  {
    id: 'time',
    question: 'What time of day do you feel most creative?',
    answers: [
      { 
        text: 'Early morning sunrise', 
        emoji: '🌅',
        archetypes: { EUTERPE: 2, TERPSICHORE: 1, THALIA: 1 }
      },
      { 
        text: 'Bright afternoon', 
        emoji: '☀️',
        archetypes: { THALIA: 2, CALLIOPE: 1, CLIO: 1 }
      },
      { 
        text: 'Golden hour evening', 
        emoji: '🌇',
        archetypes: { ERATO: 2, MELPOMENE: 1, POLYHYMNIA: 1 }
      },
      { 
        text: 'Quiet late night', 
        emoji: '🌙',
        archetypes: { PSYCHE: 2, URANIA: 2, SOPHIA: 1 }
      }
    ]
  },
  {
    id: 'music',
    question: 'What kind of music speaks to your soul?',
    answers: [
      { 
        text: 'Epic and powerful', 
        emoji: '🎸',
        archetypes: { CALLIOPE: 2, MELPOMENE: 2, CLIO: 1 }
      },
      { 
        text: 'Soft and romantic', 
        emoji: '💕',
        archetypes: { ERATO: 3, EUTERPE: 1, PSYCHE: 1 }
      },
      { 
        text: 'Rhythmic and danceable', 
        emoji: '💃',
        archetypes: { TERPSICHORE: 3, THALIA: 1, EUTERPE: 1 }
      },
      { 
        text: 'Mystical and ethereal', 
        emoji: '🔮',
        archetypes: { URANIA: 2, PSYCHE: 2, SOPHIA: 1 }
      }
    ]
  },
  {
    id: 'create',
    question: 'When you create something, you prefer to:',
    answers: [
      { 
        text: 'Plan everything carefully', 
        emoji: '📐',
        archetypes: { URANIA: 2, CALLIOPE: 2, CLIO: 1 }
      },
      { 
        text: 'Follow your intuition', 
        emoji: '🌟',
        archetypes: { PSYCHE: 2, ERATO: 2, SOPHIA: 1 }
      },
      { 
        text: 'Experiment playfully', 
        emoji: '🎨',
        archetypes: { THALIA: 2, TERPSICHORE: 1, EUTERPE: 1 }
      },
      { 
        text: 'Express deep emotions', 
        emoji: '❤️',
        archetypes: { MELPOMENE: 2, ERATO: 1, POLYHYMNIA: 1 }
      }
    ]
  },
  {
    id: 'learn',
    question: 'What fascinates you most?',
    answers: [
      { 
        text: 'The stories of humanity', 
        emoji: '📚',
        archetypes: { CLIO: 3, CALLIOPE: 1, MNEMOSYNE: 1 }
      },
      { 
        text: 'The mysteries of the cosmos', 
        emoji: '🌌',
        archetypes: { URANIA: 3, SOPHIA: 2, PSYCHE: 1 }
      },
      { 
        text: 'The beauty of connection', 
        emoji: '🤝',
        archetypes: { ERATO: 2, POLYHYMNIA: 2, MNEMOSYNE: 1 }
      },
      { 
        text: 'The joy of movement', 
        emoji: '🏃',
        archetypes: { TERPSICHORE: 3, THALIA: 1, EUTERPE: 1 }
      }
    ]
  },
  {
    id: 'environment',
    question: 'Where do you feel most at peace?',
    answers: [
      { 
        text: 'In nature, under open skies', 
        emoji: '🌲',
        archetypes: { URANIA: 2, EUTERPE: 2, PSYCHE: 1 }
      },
      { 
        text: 'In a cozy, warm space', 
        emoji: '🏠',
        archetypes: { ERATO: 2, MNEMOSYNE: 2, POLYHYMNIA: 1 }
      },
      { 
        text: 'In bustling, vibrant places', 
        emoji: '🎭',
        archetypes: { THALIA: 2, TERPSICHORE: 2, CLIO: 1 }
      },
      { 
        text: 'In quiet, sacred spaces', 
        emoji: '🕊️',
        archetypes: { POLYHYMNIA: 2, SOPHIA: 2, PSYCHE: 1 }
      }
    ]
  }
];

const ARCHETYPE_INFO: Record<string, ArchetypeScore> = {
  CALLIOPE: {
    archetype: 'CALLIOPE',
    score: 0,
    frequency: 432,
    color: 'from-purple-500 to-indigo-600',
    description: 'Epic storyteller, voice of heroes'
  },
  CLIO: {
    archetype: 'CLIO',
    score: 0,
    frequency: 456,
    color: 'from-amber-500 to-orange-600',
    description: 'Keeper of history, wisdom of time'
  },
  ERATO: {
    archetype: 'ERATO',
    score: 0,
    frequency: 528,
    color: 'from-pink-500 to-rose-600',
    description: 'Romantic soul, love\'s melody'
  },
  EUTERPE: {
    archetype: 'EUTERPE',
    score: 0,
    frequency: 594,
    color: 'from-green-500 to-emerald-600',
    description: 'Musical joy, nature\'s harmony'
  },
  MELPOMENE: {
    archetype: 'MELPOMENE',
    score: 0,
    frequency: 639,
    color: 'from-red-500 to-red-700',
    description: 'Deep emotions, transformative power'
  },
  POLYHYMNIA: {
    archetype: 'POLYHYMNIA',
    score: 0,
    frequency: 693,
    color: 'from-violet-500 to-purple-700',
    description: 'Sacred expression, divine connection'
  },
  TERPSICHORE: {
    archetype: 'TERPSICHORE',
    score: 0,
    frequency: 741,
    color: 'from-cyan-500 to-blue-600',
    description: 'Dance of life, rhythmic flow'
  },
  THALIA: {
    archetype: 'THALIA',
    score: 0,
    frequency: 825,
    color: 'from-yellow-400 to-amber-500',
    description: 'Joyful creativity, playful spirit'
  },
  URANIA: {
    archetype: 'URANIA',
    score: 0,
    frequency: 888,
    color: 'from-indigo-500 to-blue-700',
    description: 'Cosmic wisdom, celestial patterns'
  },
  MNEMOSYNE: {
    archetype: 'MNEMOSYNE',
    score: 0,
    frequency: 912,
    color: 'from-gray-500 to-slate-700',
    description: 'Memory keeper, ancestral wisdom'
  },
  PSYCHE: {
    archetype: 'PSYCHE',
    score: 0,
    frequency: 936,
    color: 'from-purple-600 to-pink-600',
    description: 'Soul\'s journey, inner light'
  },
  SOPHIA: {
    archetype: 'SOPHIA',
    score: 0,
    frequency: 963,
    color: 'from-white to-gray-300',
    description: 'Ultimate wisdom, pure understanding'
  }
};

interface Props {
  onComplete: (primaryArchetype: string, frequency: number, scores: Record<string, number>) => void;
}

export const MusePersonalityQuiz: React.FC<Props> = ({ onComplete }) => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<Array<number>>([]);
  const [showResults, setShowResults] = useState(false);
  const [archetypeScores, setArchetypeScores] = useState<Record<string, number>>({});

  const handleAnswer = (answerIndex: number) => {
    const newAnswers = [...answers];
    newAnswers[currentQuestion] = answerIndex;
    setAnswers(newAnswers);

    // Calculate scores incrementally
    const scores = { ...ARCHETYPE_INFO };
    const answer = QUIZ_QUESTIONS[currentQuestion].answers[answerIndex];
    
    Object.entries(answer.archetypes).forEach(([archetype, weight]) => {
      if (scores[archetype]) {
        scores[archetype].score += weight;
      }
    });

    // Move to next question or show results
    if (currentQuestion < QUIZ_QUESTIONS.length - 1) {
      setTimeout(() => setCurrentQuestion(currentQuestion + 1), 300);
    } else {
      calculateFinalScores(newAnswers);
    }
  };

  const calculateFinalScores = (allAnswers: number[]) => {
    const scores = { ...ARCHETYPE_INFO };
    
    // Calculate total scores
    allAnswers.forEach((answerIndex, questionIndex) => {
      if (answerIndex !== undefined) {
        const answer = QUIZ_QUESTIONS[questionIndex].answers[answerIndex];
        Object.entries(answer.archetypes).forEach(([archetype, weight]) => {
          if (scores[archetype]) {
            scores[archetype].score += weight;
          }
        });
      }
    });

    // Find primary archetype
    const sortedArchetypes = Object.values(scores).sort((a, b) => b.score - a.score);
    const primaryArchetype = sortedArchetypes[0];
    
    // Create simple scores object for parent
    const simpleScores: Record<string, number> = {};
    Object.entries(scores).forEach(([key, value]) => {
      simpleScores[key] = value.score;
    });
    
    setArchetypeScores(simpleScores);
    setShowResults(true);
    
    // Trigger completion after showing results
    setTimeout(() => {
      onComplete(primaryArchetype.archetype, primaryArchetype.frequency, simpleScores);
    }, 3000);
  };

  const goBack = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  const question = QUIZ_QUESTIONS[currentQuestion];
  const progress = ((currentQuestion + 1) / QUIZ_QUESTIONS.length) * 100;

  if (showResults) {
    const sortedArchetypes = Object.values(ARCHETYPE_INFO)
      .map(info => ({ ...info, score: archetypeScores[info.archetype] || 0 }))
      .sort((a, b) => b.score - a.score);
    const primary = sortedArchetypes[0];
    
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-2xl mx-auto p-8"
      >
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", delay: 0.2 }}
            className="inline-block mb-4"
          >
            <Sparkles className="w-16 h-16 text-yellow-400" />
          </motion.div>
          
          <h2 className="text-3xl font-light mb-2">Your Musical Essence</h2>
          <p className="text-gray-400">We've discovered your unique frequency</p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className={`bg-gradient-to-r ${primary.color} p-6 rounded-2xl text-white mb-6`}
        >
          <h3 className="text-2xl font-medium mb-2">{primary.archetype}</h3>
          <p className="mb-4 opacity-90">{primary.description}</p>
          <div className="flex items-center justify-between">
            <span className="text-sm opacity-75">Your Frequency</span>
            <span className="text-2xl font-mono">{primary.frequency} Hz</span>
          </div>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="text-center text-gray-400"
        >
          Preparing your personalized musical experience...
        </motion.p>
      </motion.div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto p-8">
      {/* Progress Bar */}
      <div className="mb-8">
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
        <p className="text-sm text-gray-400 mt-2">
          Question {currentQuestion + 1} of {QUIZ_QUESTIONS.length}
        </p>
      </div>

      {/* Question */}
      <AnimatePresence mode="wait">
        <motion.div
          key={question.id}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          className="mb-8"
        >
          <h2 className="text-2xl font-light mb-8">{question.question}</h2>
          
          <div className="space-y-3">
            {question.answers.map((answer, index) => (
              <motion.button
                key={index}
                onClick={() => handleAnswer(index)}
                className="w-full p-4 text-left rounded-xl border border-gray-700 hover:border-purple-500 hover:bg-purple-500/10 transition-all duration-200 group"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-3">
                    <span className="text-2xl">{answer.emoji}</span>
                    <span className="group-hover:text-white transition-colors">
                      {answer.text}
                    </span>
                  </span>
                  <ChevronRight className="w-5 h-5 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </motion.button>
            ))}
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Navigation */}
      <div className="flex justify-between mt-8">
        <button
          onClick={goBack}
          disabled={currentQuestion === 0}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            currentQuestion === 0 
              ? 'opacity-50 cursor-not-allowed' 
              : 'hover:bg-gray-800'
          }`}
        >
          <ChevronLeft className="w-4 h-4" />
          Back
        </button>
        
        <div className="text-sm text-gray-400">
          Press any option to continue
        </div>
      </div>
    </div>
  );
};
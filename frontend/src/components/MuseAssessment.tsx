/**
 * MUSE Assessment Component
 * 
 * Personality assessment interface that generates frequency signatures
 * based on user responses to archetypal questions.
 */

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Target, ChevronRight, ChevronLeft, Sparkles } from 'lucide-react'
import { useCompleteAssessment } from '@/hooks/useMuseAPI'
import { useMuseStore, useMuseActions } from '@/stores/useMuseStore'
import { PersonalityAssessmentRequest, MuseArchetype } from '@/types'

const MuseAssessment: React.FC = () => {
  const navigate = useNavigate()
  const { user } = useMuseStore()
  const { setSignature, showSuccess } = useMuseActions()
  const { execute: completeAssessment, loading } = useCompleteAssessment()
  
  const [currentStep, setCurrentStep] = useState(0)
  const [responses, setResponses] = useState<Record<string, any>>({})
  
  // Assessment questions organized by category
  const assessmentSteps = [
    {
      title: 'Creative Preferences',
      description: 'Tell us about your creative inclinations',
      questions: [
        {
          id: 'preferred_themes',
          type: 'multiple',
          question: 'Which themes resonate most with you?',
          options: [
            'Love and relationships',
            'Nature and cosmos',
            'History and tradition',
            'Art and beauty',
            'Wisdom and knowledge',
            'Adventure and heroism',
            'Spirituality and transcendence',
            'Joy and celebration',
            'Tragedy and loss',
            'Music and rhythm',
            'Dance and movement',
            'Craft and skill',
          ],
        },
        {
          id: 'creative_forms',
          type: 'multiple',
          question: 'Which creative forms appeal to you?',
          options: [
            'Epic poetry',
            'Lyric poetry',
            'Sonnets',
            'Haiku',
            'Free verse',
            'Narrative poetry',
            'Hymns and prayers',
            'Song lyrics',
            'Prose poetry',
            'Experimental forms',
          ],
        },
      ],
    },
    {
      title: 'Personality Traits',
      description: 'Help us understand your archetypal nature',
      questions: [
        {
          id: 'inspiration_source',
          type: 'scale',
          question: 'I find inspiration primarily through:',
          scale: {
            left: 'Inner reflection',
            right: 'External experiences',
            range: [1, 7],
          },
        },
        {
          id: 'creative_process',
          type: 'scale',
          question: 'When creating, I prefer:',
          scale: {
            left: 'Structured approach',
            right: 'Spontaneous flow',
            range: [1, 7],
          },
        },
        {
          id: 'emotional_depth',
          type: 'scale',
          question: 'I am drawn to works that are:',
          scale: {
            left: 'Intellectually complex',
            right: 'Emotionally profound',
            range: [1, 7],
          },
        },
        {
          id: 'artistic_style',
          type: 'scale',
          question: 'My ideal creative expression is:',
          scale: {
            left: 'Classical and timeless',
            right: 'Modern and innovative',
            range: [1, 7],
          },
        },
      ],
    },
    {
      title: 'Mathematical Affinities',
      description: 'Discover your connection to sacred mathematics',
      questions: [
        {
          id: 'pattern_preference',
          type: 'choice',
          question: 'Which pattern speaks to you most?',
          options: [
            { value: 'golden_ratio', label: 'Golden Ratio (φ) - Perfect proportions' },
            { value: 'fibonacci', label: 'Fibonacci Sequence - Natural growth' },
            { value: 'pi', label: 'Pi (π) - Infinite circles' },
            { value: 'euler', label: 'Euler\'s Number (e) - Exponential beauty' },
          ],
        },
        {
          id: 'geometry_preference',
          type: 'choice',
          question: 'Which geometric form resonates with you?',
          options: [
            { value: 'spiral', label: 'Spiral - Growth and evolution' },
            { value: 'circle', label: 'Circle - Unity and wholeness' },
            { value: 'triangle', label: 'Triangle - Stability and harmony' },
            { value: 'pentagon', label: 'Pentagon - Sacred proportions' },
          ],
        },
      ],
    },
  ]
  
  const currentStepData = assessmentSteps[currentStep]
  const isLastStep = currentStep === assessmentSteps.length - 1
  
  const handleResponse = (questionId: string, value: any) => {
    setResponses({
      ...responses,
      [questionId]: value,
    })
  }
  
  const handleNext = () => {
    if (isLastStep) {
      handleSubmit()
    } else {
      setCurrentStep(currentStep + 1)
    }
  }
  
  const handleSubmit = async () => {
    if (!user) return
    
    try {
      const assessmentData: PersonalityAssessmentRequest = {
        user_id: user.id,
        creative_preferences: {
          preferred_themes: responses.preferred_themes || [],
          creative_forms: responses.creative_forms || [],
        },
        personality_traits: {
          inspiration_source: responses.inspiration_source || 4,
          creative_process: responses.creative_process || 4,
          emotional_depth: responses.emotional_depth || 4,
          artistic_style: responses.artistic_style || 4,
        },
        mathematical_affinity: {
          pattern_preference: responses.pattern_preference || 'golden_ratio',
          geometry_preference: responses.geometry_preference || 'spiral',
        },
        preferred_forms: responses.creative_forms || [],
        favorite_themes: responses.preferred_themes || [],
        discovery_style: 'balanced',
      }
      
      const signature = await completeAssessment(assessmentData)
      setSignature(signature)
      showSuccess('Assessment complete! Your frequency signature has been generated.')
      navigate('/signature')
    } catch (error) {
      console.error('Assessment failed:', error)
    }
  }
  
  const canProceed = () => {
    const currentQuestions = currentStepData.questions
    return currentQuestions.every(q => responses[q.id] !== undefined)
  }
  
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 mx-auto bg-primary/10 rounded-full flex items-center justify-center mb-4">
            <Target className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-3xl font-display font-bold mb-2">
            Archetypal Assessment
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Discover your unique frequency signature by answering questions about your
            creative preferences, personality traits, and mathematical affinities.
          </p>
        </div>
        
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-muted-foreground">
              Step {currentStep + 1} of {assessmentSteps.length}
            </span>
            <span className="text-sm text-muted-foreground">
              {Math.round(((currentStep + 1) / assessmentSteps.length) * 100)}%
            </span>
          </div>
          <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-primary"
              initial={{ width: 0 }}
              animate={{ width: `${((currentStep + 1) / assessmentSteps.length) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
        
        {/* Current Step */}
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="bg-card border border-border rounded-lg p-8"
        >
          <div className="text-center mb-8">
            <h2 className="text-2xl font-semibold mb-2">
              {currentStepData.title}
            </h2>
            <p className="text-muted-foreground">
              {currentStepData.description}
            </p>
          </div>
          
          <div className="space-y-8">
            {currentStepData.questions.map((question) => (
              <div key={question.id} className="space-y-4">
                <h3 className="text-lg font-medium">{question.question}</h3>
                
                {question.type === 'multiple' && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {question.options?.map((option) => (
                      <label
                        key={option}
                        className="flex items-center space-x-2 p-3 border border-border rounded-lg hover:border-primary/50 cursor-pointer transition-colors"
                      >
                        <input
                          type="checkbox"
                          className="text-primary"
                          checked={(responses[question.id] || []).includes(option)}
                          onChange={(e) => {
                            const current = responses[question.id] || []
                            if (e.target.checked) {
                              handleResponse(question.id, [...current, option])
                            } else {
                              handleResponse(question.id, current.filter((x: string) => x !== option))
                            }
                          }}
                        />
                        <span>{option}</span>
                      </label>
                    ))}
                  </div>
                )}
                
                {question.type === 'scale' && (
                  <div className="space-y-4">
                    <div className="flex justify-between text-sm text-muted-foreground">
                      <span>{question.scale?.left}</span>
                      <span>{question.scale?.right}</span>
                    </div>
                    <div className="flex justify-center">
                      <div className="flex space-x-2">
                        {Array.from({ length: question.scale?.range?.[1] || 7 }, (_, i) => (
                          <button
                            key={i}
                            onClick={() => handleResponse(question.id, i + 1)}
                            className={`w-8 h-8 rounded-full border-2 transition-colors ${
                              responses[question.id] === i + 1
                                ? 'bg-primary border-primary text-primary-foreground'
                                : 'border-border hover:border-primary/50'
                            }`}
                          >
                            {i + 1}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                
                {question.type === 'choice' && (
                  <div className="space-y-2">
                    {question.options?.map((option) => (
                      <label
                        key={option.value}
                        className="flex items-center space-x-3 p-4 border border-border rounded-lg hover:border-primary/50 cursor-pointer transition-colors"
                      >
                        <input
                          type="radio"
                          name={question.id}
                          value={option.value}
                          checked={responses[question.id] === option.value}
                          onChange={(e) => handleResponse(question.id, e.target.value)}
                          className="text-primary"
                        />
                        <span className="font-medium">{option.label}</span>
                      </label>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </motion.div>
        
        {/* Navigation */}
        <div className="flex justify-between items-center mt-8">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className="inline-flex items-center px-4 py-2 border border-input rounded-lg hover:bg-accent hover:text-accent-foreground transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-4 h-4 mr-2" />
            Previous
          </button>
          
          <button
            onClick={handleNext}
            disabled={!canProceed() || loading.isLoading}
            className="inline-flex items-center px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading.isLoading ? (
              <>
                <Sparkles className="w-4 h-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : isLastStep ? (
              <>
                Complete Assessment
                <Sparkles className="w-4 h-4 ml-2" />
              </>
            ) : (
              <>
                Next
                <ChevronRight className="w-4 h-4 ml-2" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

export default MuseAssessment
/**
 * MUSE Discovery Interface
 * 
 * Real-time creative discovery interface with constraint optimization
 * and live fitness score monitoring.
 */

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Compass, Play, Pause, RotateCcw, Target, Activity } from 'lucide-react'
import { useDiscoverySession } from '@/hooks/useMuseAPI'
import { useMuseStore } from '@/stores/useMuseStore'
import { DiscoverySessionRequest } from '@/types'

const MuseDiscoveryInterface: React.FC = () => {
  const { user } = useMuseStore()
  const {
    session,
    loading,
    error,
    startSession,
    continueSession,
    completeSession,
  } = useDiscoverySession()
  
  const [discoveryForm, setDiscoveryForm] = useState({
    theme: '',
    form_type: 'auto',
    mode: 'individual' as const,
    constraints: {},
  })
  
  const handleStartDiscovery = async () => {
    if (!user) return
    
    const request: DiscoverySessionRequest = {
      user_id: user.id,
      theme: discoveryForm.theme,
      form_type: discoveryForm.form_type,
      mode: discoveryForm.mode,
      constraints: discoveryForm.constraints,
    }
    
    await startSession(request)
  }
  
  const handleContinue = async () => {
    await continueSession()
  }
  
  const handleComplete = async () => {
    const result = await completeSession()
    if (result) {
      console.log('Discovery completed:', result)
    }
  }
  
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 mx-auto bg-primary/10 rounded-full flex items-center justify-center mb-4">
            <Compass className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-3xl font-display font-bold mb-2">
            Creative Discovery
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Discover creative works that exist in the mathematical realm through
            archetypal frequency optimization and sacred geometry.
          </p>
        </div>
        
        {!session ? (
          /* Discovery Setup */
          <div className="max-w-2xl mx-auto">
            <div className="bg-card border border-border rounded-lg p-8">
              <h2 className="text-xl font-semibold mb-6">Start New Discovery</h2>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Theme or Subject
                  </label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-input rounded-md bg-background"
                    placeholder="Enter your creative theme..."
                    value={discoveryForm.theme}
                    onChange={(e) => setDiscoveryForm({
                      ...discoveryForm,
                      theme: e.target.value,
                    })}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Form Type
                  </label>
                  <select
                    className="w-full px-3 py-2 border border-input rounded-md bg-background"
                    value={discoveryForm.form_type}
                    onChange={(e) => setDiscoveryForm({
                      ...discoveryForm,
                      form_type: e.target.value,
                    })}
                  >
                    <option value="auto">Auto-detect</option>
                    <option value="sonnet">Sonnet</option>
                    <option value="haiku">Haiku</option>
                    <option value="free_verse">Free Verse</option>
                    <option value="epic">Epic</option>
                    <option value="lyric">Lyric</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Discovery Mode
                  </label>
                  <select
                    className="w-full px-3 py-2 border border-input rounded-md bg-background"
                    value={discoveryForm.mode}
                    onChange={(e) => setDiscoveryForm({
                      ...discoveryForm,
                      mode: e.target.value as any,
                    })}
                  >
                    <option value="individual">Individual</option>
                    <option value="guided">Guided</option>
                    <option value="experimental">Experimental</option>
                  </select>
                </div>
                
                <button
                  onClick={handleStartDiscovery}
                  disabled={!discoveryForm.theme || loading.isLoading}
                  className="w-full px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading.isLoading ? (
                    <>
                      <Activity className="w-4 h-4 mr-2 animate-spin inline" />
                      Initializing Discovery...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2 inline" />
                      Begin Discovery
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* Active Discovery Session */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Session Status */}
            <div className="bg-card border border-border rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Discovery Session</h2>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-muted-foreground">Progress</span>
                    <span className="text-sm font-medium">
                      {Math.round(session.progress * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-primary"
                      initial={{ width: 0 }}
                      animate={{ width: `${session.progress * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-sm text-muted-foreground">Phase</span>
                    <p className="font-medium capitalize">{session.phase}</p>
                  </div>
                  <div>
                    <span className="text-sm text-muted-foreground">Iteration</span>
                    <p className="font-medium">
                      {session.current_iteration} / {session.max_iterations}
                    </p>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h3 className="font-medium">Fitness Scores</h3>
                  <div className="space-y-1">
                    {Object.entries(session.fitness_scores).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="capitalize">{key.replace('_', ' ')}</span>
                        <span className="font-medium">
                          {Math.round(value * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Current Discovery */}
            <div className="bg-card border border-border rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Current Discovery</h2>
              
              <div className="space-y-4">
                <div className="bg-accent/10 border border-accent/20 rounded-lg p-4">
                  <h3 className="font-medium mb-2">Theme: {session.theme}</h3>
                  <div className="text-sm text-muted-foreground">
                    {session.current_discovery.content || 'Generating...'}
                  </div>
                </div>
                
                <div className="flex space-x-2">
                  <button
                    onClick={handleContinue}
                    disabled={loading.isLoading}
                    className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
                  >
                    {loading.isLoading ? (
                      <>
                        <Activity className="w-4 h-4 mr-2 animate-spin inline" />
                        Optimizing...
                      </>
                    ) : (
                      <>
                        <Target className="w-4 h-4 mr-2 inline" />
                        Continue
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={handleComplete}
                    disabled={loading.isLoading}
                    className="flex-1 px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 transition-colors disabled:opacity-50"
                  >
                    Complete
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {error.hasError && (
          <div className="mt-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
            <p className="text-destructive">{error.message}</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default MuseDiscoveryInterface
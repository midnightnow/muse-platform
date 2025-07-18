/**
 * MUSE Creative Session Component
 * 
 * Manages individual creative discovery sessions with real-time updates
 * and session management features.
 */

import React from 'react'
import { useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Activity, Target, Users, Clock, TrendingUp } from 'lucide-react'
import { useDiscoverySession } from '@/hooks/useMuseAPI'
import { useRealtimeDiscovery } from '@/hooks/useMuseAPI'

const MuseCreativeSession: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>()
  const { 
    session, 
    loading, 
    error, 
    continueSession, 
    completeSession,
    refreshSession 
  } = useDiscoverySession(sessionId)
  
  const { updates, currentUpdate } = useRealtimeDiscovery(sessionId)
  
  if (loading.isLoading && !session) {
    return (
      <div className="min-h-screen bg-background p-6 flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-16 h-16 mx-auto text-primary mb-4 animate-spin" />
          <h2 className="text-2xl font-semibold mb-2">Loading Session</h2>
          <p className="text-muted-foreground">
            Retrieving your creative discovery session...
          </p>
        </div>
      </div>
    )
  }
  
  if (error.hasError) {
    return (
      <div className="min-h-screen bg-background p-6 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto bg-destructive/10 rounded-full flex items-center justify-center mb-4">
            <Target className="w-8 h-8 text-destructive" />
          </div>
          <h2 className="text-2xl font-semibold mb-2">Session Error</h2>
          <p className="text-muted-foreground mb-4">
            {error.message}
          </p>
          <button
            onClick={refreshSession}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }
  
  if (!session) {
    return (
      <div className="min-h-screen bg-background p-6 flex items-center justify-center">
        <div className="text-center">
          <Target className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
          <h2 className="text-2xl font-semibold mb-2">Session Not Found</h2>
          <p className="text-muted-foreground">
            The requested discovery session could not be found.
          </p>
        </div>
      </div>
    )
  }
  
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-display font-bold mb-2">
                {session.theme}
              </h1>
              <p className="text-muted-foreground">
                {session.form_type} • {session.mode} mode
              </p>
            </div>
            <div className="text-right">
              <div className="text-sm text-muted-foreground">Session ID</div>
              <div className="font-mono text-sm">{session.session_id}</div>
            </div>
          </div>
        </div>
        
        {/* Progress Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-card border border-border rounded-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <Target className="w-5 h-5 text-primary" />
              <span className="text-2xl font-bold">
                {Math.round(session.progress * 100)}%
              </span>
            </div>
            <h3 className="font-medium">Progress</h3>
            <p className="text-sm text-muted-foreground">
              Phase: {session.phase}
            </p>
          </div>
          
          <div className="bg-card border border-border rounded-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <Activity className="w-5 h-5 text-green-600" />
              <span className="text-2xl font-bold">
                {session.current_iteration}
              </span>
            </div>
            <h3 className="font-medium">Iterations</h3>
            <p className="text-sm text-muted-foreground">
              of {session.max_iterations}
            </p>
          </div>
          
          <div className="bg-card border border-border rounded-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              <span className="text-2xl font-bold">
                {Math.round((session.fitness_scores.overall_fitness || 0) * 100)}%
              </span>
            </div>
            <h3 className="font-medium">Fitness</h3>
            <p className="text-sm text-muted-foreground">
              Overall Score
            </p>
          </div>
          
          <div className="bg-card border border-border rounded-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <Clock className="w-5 h-5 text-purple-600" />
              <span className="text-2xl font-bold">
                {Math.round((Date.now() - new Date(session.created_at).getTime()) / 60000)}
              </span>
            </div>
            <h3 className="font-medium">Duration</h3>
            <p className="text-sm text-muted-foreground">
              minutes
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Current Discovery */}
          <div className="bg-card border border-border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Current Discovery</h2>
            
            <div className="space-y-4">
              <div className="bg-accent/10 border border-accent/20 rounded-lg p-4">
                <div className="prose prose-sm max-w-none">
                  {session.current_discovery.content || 'Generating discovery...'}
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">Fitness Scores</h4>
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
                
                <div>
                  <h4 className="font-medium mb-2">Metadata</h4>
                  <div className="space-y-1 text-sm">
                    {Object.entries(session.current_discovery.metadata || {}).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="capitalize">{key.replace('_', ' ')}</span>
                        <span className="font-medium">
                          {typeof value === 'number' ? value.toFixed(2) : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Discovery Path */}
          <div className="bg-card border border-border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Discovery Path</h2>
            
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {session.discovery_path.map((step, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="border border-border rounded-lg p-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center text-xs text-primary-foreground font-medium">
                        {step.iteration}
                      </div>
                      <span className="font-medium">
                        Iteration {step.iteration}
                      </span>
                    </div>
                    <span className="text-sm font-medium">
                      {Math.round(step.fitness_score * 100)}%
                    </span>
                  </div>
                  
                  <p className="text-sm text-muted-foreground mb-2">
                    {step.content_preview}
                  </p>
                  
                  <div className="flex justify-between items-center text-xs text-muted-foreground">
                    <span>{step.optimization_notes}</span>
                    <span>{new Date(step.timestamp).toLocaleTimeString()}</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Actions */}
        <div className="mt-8 flex justify-center space-x-4">
          <button
            onClick={continueSession}
            disabled={loading.isLoading}
            className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {loading.isLoading ? (
              <>
                <Activity className="w-4 h-4 mr-2 animate-spin inline" />
                Optimizing...
              </>
            ) : (
              <>
                <Target className="w-4 h-4 mr-2 inline" />
                Continue Discovery
              </>
            )}
          </button>
          
          <button
            onClick={completeSession}
            disabled={loading.isLoading}
            className="px-6 py-3 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 transition-colors disabled:opacity-50"
          >
            Complete Session
          </button>
        </div>
        
        {/* Real-time Updates */}
        {currentUpdate && (
          <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="font-medium text-blue-900 mb-2">Live Update</h3>
            <p className="text-sm text-blue-800">
              Phase: {currentUpdate.phase} | 
              Progress: {Math.round(currentUpdate.progress * 100)}% | 
              Iteration: {currentUpdate.current_iteration}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default MuseCreativeSession
/**
 * Loading States Component
 * 
 * Provides loading animations with archetypal theming and sacred geometry
 * patterns. Displays different loading states based on the current phase.
 */

import React from 'react'
import { motion } from 'framer-motion'
import { Loader2, Sparkles, Target, Zap, CheckCircle } from 'lucide-react'
import { LoadingState } from '@/types'

interface LoadingStatesProps {
  isLoading: boolean
  phase?: string
  progress?: number
  message?: string
  className?: string
}

const LoadingStates: React.FC<LoadingStatesProps> = ({
  isLoading,
  phase = 'loading',
  progress = 0,
  message = 'Loading...',
  className = '',
}) => {
  if (!isLoading) return null

  const getPhaseIcon = () => {
    switch (phase) {
      case 'initialization':
        return <Sparkles className="w-6 h-6" />
      case 'assessment':
        return <Target className="w-6 h-6" />
      case 'optimization':
        return <Zap className="w-6 h-6" />
      case 'completion':
        return <CheckCircle className="w-6 h-6" />
      default:
        return <Loader2 className="w-6 h-6 animate-spin" />
    }
  }

  const getPhaseMessage = () => {
    switch (phase) {
      case 'initialization':
        return 'Initializing archetypal frequencies...'
      case 'assessment':
        return 'Calculating frequency signature...'
      case 'optimization':
        return 'Optimizing sacred geometry...'
      case 'completion':
        return 'Finalizing discovery...'
      case 'fetching':
        return 'Retrieving data...'
      case 'saving':
        return 'Saving to the mathematical realm...'
      default:
        return message
    }
  }

  return (
    <div className={`flex flex-col items-center justify-center space-y-6 ${className}`}>
      {/* Sacred geometry loading animation */}
      <div className="relative">
        {/* Outer ring */}
        <motion.div
          className="w-24 h-24 border-2 border-primary/20 rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />
        
        {/* Golden ratio ring */}
        <motion.div
          className="absolute inset-2 border-2 border-primary/40 rounded-full"
          animate={{ rotate: -360 }}
          transition={{ duration: 4.854, repeat: Infinity, ease: "linear" }} // 3 * φ seconds
        />
        
        {/* Inner ring */}
        <motion.div
          className="absolute inset-4 border-2 border-primary/60 rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        />
        
        {/* Center icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            className="text-primary"
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [0.8, 1, 0.8] 
            }}
            transition={{ 
              duration: 2, 
              repeat: Infinity, 
              ease: "easeInOut" 
            }}
          >
            {getPhaseIcon()}
          </motion.div>
        </div>
      </div>
      
      {/* Progress bar */}
      {progress > 0 && (
        <div className="w-64 h-2 bg-secondary rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-primary rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
      )}
      
      {/* Loading message */}
      <div className="text-center">
        <motion.h3
          className="text-lg font-medium mb-2"
          animate={{ opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        >
          {getPhaseMessage()}
        </motion.h3>
        
        {progress > 0 && (
          <p className="text-sm text-muted-foreground">
            {Math.round(progress)}% Complete
          </p>
        )}
      </div>
      
      {/* Sacred geometry pattern background */}
      <div className="absolute inset-0 -z-10 opacity-5">
        <div className="w-full h-full bg-sacred-pattern animate-sacred-pulse" />
      </div>
    </div>
  )
}

// Skeleton loading component for content
export const SkeletonLoader: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`animate-pulse ${className}`}>
    <div className="bg-muted rounded-md h-4 w-3/4 mb-2"></div>
    <div className="bg-muted rounded-md h-4 w-1/2 mb-2"></div>
    <div className="bg-muted rounded-md h-4 w-2/3"></div>
  </div>
)

// Card skeleton for creation cards
export const CreationCardSkeleton: React.FC = () => (
  <div className="bg-card border border-border rounded-lg p-4 animate-pulse">
    <div className="flex items-center space-x-3 mb-3">
      <div className="w-8 h-8 bg-muted rounded-full"></div>
      <div className="flex-1">
        <div className="bg-muted rounded-md h-4 w-1/4 mb-1"></div>
        <div className="bg-muted rounded-md h-3 w-1/3"></div>
      </div>
    </div>
    <div className="space-y-2 mb-3">
      <div className="bg-muted rounded-md h-4 w-full"></div>
      <div className="bg-muted rounded-md h-4 w-3/4"></div>
      <div className="bg-muted rounded-md h-4 w-1/2"></div>
    </div>
    <div className="flex items-center justify-between">
      <div className="bg-muted rounded-md h-3 w-1/4"></div>
      <div className="flex space-x-2">
        <div className="bg-muted rounded-md h-6 w-12"></div>
        <div className="bg-muted rounded-md h-6 w-12"></div>
      </div>
    </div>
  </div>
)

// Frequency signature skeleton
export const FrequencySignatureSkeleton: React.FC = () => (
  <div className="bg-card border border-border rounded-lg p-6 animate-pulse">
    <div className="flex items-center space-x-3 mb-6">
      <div className="w-12 h-12 bg-muted rounded-full"></div>
      <div className="flex-1">
        <div className="bg-muted rounded-md h-5 w-1/3 mb-2"></div>
        <div className="bg-muted rounded-md h-4 w-1/2"></div>
      </div>
    </div>
    
    <div className="space-y-4">
      <div className="bg-muted rounded-md h-32 w-full"></div>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-muted rounded-md h-20"></div>
        <div className="bg-muted rounded-md h-20"></div>
      </div>
    </div>
  </div>
)

// Discovery session skeleton
export const DiscoverySessionSkeleton: React.FC = () => (
  <div className="bg-card border border-border rounded-lg p-6 animate-pulse">
    <div className="flex items-center justify-between mb-4">
      <div className="bg-muted rounded-md h-6 w-1/4"></div>
      <div className="bg-muted rounded-md h-4 w-1/6"></div>
    </div>
    
    <div className="bg-muted rounded-md h-2 w-full mb-4"></div>
    
    <div className="space-y-3 mb-4">
      <div className="bg-muted rounded-md h-4 w-full"></div>
      <div className="bg-muted rounded-md h-4 w-3/4"></div>
      <div className="bg-muted rounded-md h-4 w-1/2"></div>
    </div>
    
    <div className="flex items-center justify-between">
      <div className="bg-muted rounded-md h-8 w-20"></div>
      <div className="flex space-x-2">
        <div className="bg-muted rounded-md h-8 w-24"></div>
        <div className="bg-muted rounded-md h-8 w-24"></div>
      </div>
    </div>
  </div>
)

export default LoadingStates
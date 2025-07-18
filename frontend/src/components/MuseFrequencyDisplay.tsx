/**
 * MUSE Frequency Display Component
 * 
 * Visualizes the user's frequency signature with 3D spiral coordinates,
 * radar charts, and archetypal breakdown.
 */

import React from 'react'
import { useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Target, 
  Sparkles, 
  TrendingUp, 
  Settings,
  Activity,
  Zap,
  Eye
} from 'lucide-react'
import { useMuseStore } from '@/stores/useMuseStore'
import { useFrequencySignature } from '@/hooks/useMuseAPI'
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts'

const MuseFrequencyDisplay: React.FC = () => {
  const { signatureId } = useParams()
  const { currentSignature, user } = useMuseStore()
  
  // Use provided signature ID or current user's signature
  const { data: signature } = useFrequencySignature(signatureId)
  const displaySignature = signature || currentSignature
  
  if (!displaySignature) {
    return (
      <div className="min-h-screen bg-background p-6 flex items-center justify-center">
        <div className="text-center">
          <Target className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
          <h2 className="text-2xl font-semibold mb-2">No Frequency Signature</h2>
          <p className="text-muted-foreground">
            Complete your assessment to generate your archetypal frequency signature.
          </p>
        </div>
      </div>
    )
  }
  
  // Prepare data for radar chart
  const radarData = Object.entries(displaySignature.harmonic_blend).map(([muse, value]) => ({
    muse: muse.replace('_', ' '),
    value: value * 100,
  }))
  
  const characteristics = displaySignature.characteristics || {}
  const performance = displaySignature.performance_metrics || {}
  
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 mx-auto bg-primary/10 rounded-full flex items-center justify-center mb-4">
            <Sparkles className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-3xl font-display font-bold mb-2">
            Frequency Signature
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Your unique archetypal frequency signature based on sacred mathematics
            and the twelve classical Muses.
          </p>
        </div>
        
        {/* Primary Muse Display */}
        <div className="bg-card border border-border rounded-lg p-8 mb-8">
          <div className="text-center">
            <div className="w-24 h-24 mx-auto bg-primary/10 rounded-full flex items-center justify-center mb-4">
              <Target className="w-12 h-12 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold mb-2">
              {displaySignature.primary_muse}
            </h2>
            <p className="text-muted-foreground mb-4">
              Primary Archetypal Frequency
            </p>
            <div className="flex justify-center items-center space-x-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-primary">
                  {Math.round((displaySignature.harmonic_blend[displaySignature.primary_muse] || 0) * 100)}%
                </div>
                <div className="text-sm text-muted-foreground">Dominance</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-secondary">
                  {displaySignature.secondary_muse}
                </div>
                <div className="text-sm text-muted-foreground">Secondary</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Archetypal Radar Chart */}
          <div className="bg-card border border-border rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Archetypal Breakdown</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid className="stroke-border" />
                  <PolarAngleAxis 
                    dataKey="muse" 
                    className="text-xs fill-muted-foreground"
                  />
                  <PolarRadiusAxis 
                    angle={30} 
                    domain={[0, 100]} 
                    className="text-xs fill-muted-foreground"
                  />
                  <Radar
                    name="Frequency"
                    dataKey="value"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.1}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Sacred Ratios */}
          <div className="bg-card border border-border rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Sacred Ratios</h3>
            <div className="space-y-4">
              {Object.entries(displaySignature.sacred_ratios).map(([ratio, value]) => (
                <div key={ratio} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium capitalize">
                      {ratio === 'phi' ? 'φ (Golden Ratio)' : 
                       ratio === 'pi' ? 'π (Pi)' : 
                       ratio === 'euler' ? 'e (Euler)' : 
                       ratio.replace('_', ' ')}
                    </span>
                    <span className="text-sm font-mono">
                      {value.toFixed(6)}
                    </span>
                  </div>
                  <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary transition-all duration-500"
                      style={{ width: `${Math.min(value * 50, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Characteristics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-8">
          {[
            {
              label: 'Specialization',
              value: characteristics.specialization_index || 0,
              icon: <Target className="w-5 h-5" />,
              color: 'text-blue-600',
            },
            {
              label: 'Diversity',
              value: characteristics.diversity_index || 0,
              icon: <Activity className="w-5 h-5" />,
              color: 'text-green-600',
            },
            {
              label: 'Coherence',
              value: characteristics.coherence_score || 0,
              icon: <Zap className="w-5 h-5" />,
              color: 'text-purple-600',
            },
            {
              label: 'Uniqueness',
              value: characteristics.uniqueness_score || 0,
              icon: <Eye className="w-5 h-5" />,
              color: 'text-orange-600',
            },
          ].map((item) => (
            <div key={item.label} className="bg-card border border-border rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <div className={`p-2 rounded-lg bg-opacity-10 ${item.color}`}>
                  {item.icon}
                </div>
                <div className="text-2xl font-bold">
                  {Math.round(item.value * 100)}%
                </div>
              </div>
              <h3 className="font-medium">{item.label}</h3>
              <div className="w-full bg-secondary h-2 rounded-full overflow-hidden mt-2">
                <motion.div
                  className="h-full bg-primary"
                  initial={{ width: 0 }}
                  animate={{ width: `${item.value * 100}%` }}
                  transition={{ duration: 1, delay: 0.5 }}
                />
              </div>
            </div>
          ))}
        </div>
        
        {/* Performance Metrics */}
        <div className="bg-card border border-border rounded-lg p-6 mt-8">
          <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary mb-2">
                {Math.round((performance.discovery_success_rate || 0) * 100)}%
              </div>
              <div className="text-sm text-muted-foreground">
                Discovery Success Rate
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-secondary mb-2">
                {Math.round((performance.average_fitness_score || 0) * 100)}%
              </div>
              <div className="text-sm text-muted-foreground">
                Average Fitness Score
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-accent mb-2">
                {Math.round((performance.user_satisfaction_score || 0) * 100)}%
              </div>
              <div className="text-sm text-muted-foreground">
                User Satisfaction
              </div>
            </div>
          </div>
        </div>
        
        {/* Spiral Coordinates (placeholder for 3D visualization) */}
        <div className="bg-card border border-border rounded-lg p-6 mt-8">
          <h3 className="text-lg font-semibold mb-4">Spiral Coordinates</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Radius</span>
                <span className="text-sm font-mono">
                  {displaySignature.spiral_coordinates.radius.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Angle</span>
                <span className="text-sm font-mono">
                  {displaySignature.spiral_coordinates.angle.toFixed(2)}°
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Height</span>
                <span className="text-sm font-mono">
                  {displaySignature.spiral_coordinates.height.toFixed(2)}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Spiral Type</span>
                <span className="text-sm font-medium capitalize">
                  {displaySignature.spiral_coordinates.spiral_type}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Turn Count</span>
                <span className="text-sm font-mono">
                  {displaySignature.spiral_coordinates.turn_count.toFixed(1)}
                </span>
              </div>
            </div>
          </div>
          
          {/* Placeholder for 3D spiral visualization */}
          <div className="mt-4 h-64 bg-accent/5 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-primary/10 rounded-full flex items-center justify-center mb-4">
                <TrendingUp className="w-8 h-8 text-primary" />
              </div>
              <p className="text-muted-foreground">
                3D Spiral Visualization
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                Interactive visualization coming soon
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MuseFrequencyDisplay
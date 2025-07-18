/**
 * MUSE Dashboard Page
 * 
 * The main dashboard showing user overview, recent discoveries,
 * frequency signature status, and quick actions.
 */

import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Sparkles,
  Target,
  TrendingUp,
  Users,
  Calendar,
  ChevronRight,
  Plus,
  Activity,
  Award,
  Heart,
} from 'lucide-react'
import { useMuseStore } from '@/stores/useMuseStore'
import { useUserSessions } from '@/hooks/useMuseAPI'
import { cn } from '@/utils/cn'

const MuseDashboardPage: React.FC = () => {
  const { user, currentSignature, theme } = useMuseStore()
  const { data: sessions, loading } = useUserSessions(user?.id)
  
  // Mock data for demonstration
  const stats = [
    {
      label: 'Total Discoveries',
      value: user?.total_discoveries || 0,
      change: '+2 this week',
      icon: <Sparkles className="w-5 h-5" />,
      color: 'text-blue-600',
    },
    {
      label: 'Avg. Fitness Score',
      value: `${Math.round((user?.average_fitness_score || 0) * 100)}%`,
      change: '+5% this month',
      icon: <Target className="w-5 h-5" />,
      color: 'text-green-600',
    },
    {
      label: 'Community Resonance',
      value: '847',
      change: '+12% this month',
      icon: <Heart className="w-5 h-5" />,
      color: 'text-pink-600',
    },
    {
      label: 'Active Collaborations',
      value: '3',
      change: '+1 this week',
      icon: <Users className="w-5 h-5" />,
      color: 'text-purple-600',
    },
  ]
  
  const quickActions = [
    {
      title: 'Start Discovery',
      description: 'Begin a new creative discovery session',
      href: '/discovery',
      icon: <Sparkles className="w-6 h-6" />,
      color: 'bg-blue-500',
    },
    {
      title: 'View Signature',
      description: 'Explore your frequency signature',
      href: '/signature',
      icon: <Target className="w-6 h-6" />,
      color: 'bg-green-500',
    },
    {
      title: 'Community',
      description: 'Connect with kindred spirits',
      href: '/community',
      icon: <Users className="w-6 h-6" />,
      color: 'bg-purple-500',
    },
    {
      title: 'Profile',
      description: 'Edit your profile and preferences',
      href: '/profile',
      icon: <Award className="w-6 h-6" />,
      color: 'bg-orange-500',
    },
  ]
  
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-3xl font-display font-bold mb-2">
              Welcome back, {user?.display_name || user?.username}
            </h1>
            <p className="text-muted-foreground">
              Your creative journey guided by {currentSignature?.primary_muse || 'the Muses'}
            </p>
          </motion.div>
        </div>
        
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="bg-card border border-border rounded-lg p-6 hover:shadow-lg transition-shadow"
            >
              <div className="flex items-center justify-between mb-2">
                <div className={cn('p-2 rounded-lg bg-opacity-10', stat.color)}>
                  {stat.icon}
                </div>
                <span className="text-2xl font-bold">{stat.value}</span>
              </div>
              <h3 className="text-sm font-medium text-muted-foreground mb-1">
                {stat.label}
              </h3>
              <p className="text-xs text-green-600">{stat.change}</p>
            </motion.div>
          ))}
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="lg:col-span-2"
          >
            <div className="bg-card border border-border rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {quickActions.map((action, index) => (
                  <Link
                    key={action.title}
                    to={action.href}
                    className="group p-4 border border-border rounded-lg hover:border-primary/50 hover:shadow-md transition-all"
                  >
                    <div className="flex items-center space-x-3 mb-2">
                      <div className={cn('p-2 rounded-lg text-white', action.color)}>
                        {action.icon}
                      </div>
                      <h3 className="font-medium group-hover:text-primary transition-colors">
                        {action.title}
                      </h3>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      {action.description}
                    </p>
                    <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                  </Link>
                ))}
              </div>
            </div>
          </motion.div>
          
          {/* Frequency Signature Preview */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <div className="bg-card border border-border rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Your Signature</h2>
                <Link
                  to="/signature"
                  className="text-sm text-primary hover:text-primary/80 transition-colors"
                >
                  View Details
                </Link>
              </div>
              
              {currentSignature ? (
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="w-16 h-16 mx-auto bg-primary/10 rounded-full flex items-center justify-center mb-2">
                      <Sparkles className="w-8 h-8 text-primary" />
                    </div>
                    <h3 className="font-semibold text-lg">
                      {currentSignature.primary_muse}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      Primary Archetypal Frequency
                    </p>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Coherence Score</span>
                      <span className="font-medium">
                        {Math.round((currentSignature.characteristics?.coherence_score || 0) * 100)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Uniqueness</span>
                      <span className="font-medium">
                        {Math.round((currentSignature.characteristics?.uniqueness_score || 0) * 100)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Diversity Index</span>
                      <span className="font-medium">
                        {Math.round((currentSignature.characteristics?.diversity_index || 0) * 100)}%
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Target className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Complete your assessment to view your frequency signature
                  </p>
                  <Link
                    to="/assessment"
                    className="inline-block mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors text-sm"
                  >
                    Start Assessment
                  </Link>
                </div>
              )}
            </div>
          </motion.div>
        </div>
        
        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-8"
        >
          <div className="bg-card border border-border rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Recent Discoveries</h2>
              <Link
                to="/discovery"
                className="text-sm text-primary hover:text-primary/80 transition-colors"
              >
                View All
              </Link>
            </div>
            
            {loading.isLoading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="animate-pulse">
                    <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
                    <div className="h-3 bg-muted rounded w-1/2"></div>
                  </div>
                ))}
              </div>
            ) : sessions && sessions.length > 0 ? (
              <div className="space-y-4">
                {sessions.slice(0, 3).map((session) => (
                  <div
                    key={session.session_id}
                    className="flex items-center justify-between p-4 border border-border rounded-lg hover:border-primary/50 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                        <Activity className="w-4 h-4 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-medium">{session.theme}</h3>
                        <p className="text-sm text-muted-foreground">
                          {session.form_type} • {session.phase}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">
                        {Math.round(session.fitness_scores?.overall_fitness || 0)}%
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(session.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Sparkles className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-sm text-muted-foreground mb-4">
                  No discoveries yet. Start your first creative journey!
                </p>
                <Link
                  to="/discovery"
                  className="inline-flex items-center px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors text-sm"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Start Discovery
                </Link>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default MuseDashboardPage
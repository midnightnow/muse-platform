/**
 * MUSE Profile Page
 * 
 * User profile page with frequency signature display,
 * creation gallery, and social statistics.
 */

import React, { useState } from 'react'
import { useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  User,
  Settings,
  Heart,
  MessageSquare,
  Share2,
  Target,
  Activity,
  Users,
  Calendar,
  Edit,
  Eye,
  Sparkles,
} from 'lucide-react'
import { useMuseStore } from '@/stores/useMuseStore'
import { useUserProfile, useUserCreations } from '@/hooks/useMuseAPI'
import { cn } from '@/utils/cn'

const MuseProfilePage: React.FC = () => {
  const { userId } = useParams<{ userId: string }>()
  const { user: currentUser } = useMuseStore()
  const [activeTab, setActiveTab] = useState<'creations' | 'signature' | 'stats'>('creations')
  
  // Use provided userId or current user's ID
  const profileUserId = userId || currentUser?.id
  const isOwnProfile = !userId || userId === currentUser?.id
  
  const { data: profile, loading: profileLoading } = useUserProfile(profileUserId)
  const { data: creationsData, loading: creationsLoading } = useUserCreations(profileUserId)
  
  const displayProfile = profile || currentUser
  
  if (profileLoading.isLoading && !displayProfile) {
    return (
      <div className="min-h-screen bg-background p-6 flex items-center justify-center">
        <div className="text-center">
          <User className="w-16 h-16 mx-auto text-primary mb-4 animate-spin" />
          <h2 className="text-2xl font-semibold mb-2">Loading Profile</h2>
          <p className="text-muted-foreground">
            Retrieving profile information...
          </p>
        </div>
      </div>
    )
  }
  
  if (!displayProfile) {
    return (
      <div className="min-h-screen bg-background p-6 flex items-center justify-center">
        <div className="text-center">
          <User className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
          <h2 className="text-2xl font-semibold mb-2">Profile Not Found</h2>
          <p className="text-muted-foreground">
            The requested profile could not be found.
          </p>
        </div>
      </div>
    )
  }
  
  const creations = creationsData?.data || []
  const stats = [
    {
      label: 'Total Creations',
      value: displayProfile.creations_count || 0,
      icon: <Sparkles className="w-5 h-5" />,
    },
    {
      label: 'Total Discoveries',
      value: displayProfile.total_discoveries || 0,
      icon: <Target className="w-5 h-5" />,
    },
    {
      label: 'Followers',
      value: displayProfile.followers_count || 0,
      icon: <Users className="w-5 h-5" />,
    },
    {
      label: 'Following',
      value: displayProfile.following_count || 0,
      icon: <Eye className="w-5 h-5" />,
    },
  ]
  
  const tabs = [
    { id: 'creations', label: 'Creations', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'signature', label: 'Signature', icon: <Target className="w-4 h-4" /> },
    { id: 'stats', label: 'Statistics', icon: <Activity className="w-4 h-4" /> },
  ]
  
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto">
        {/* Profile Header */}
        <div className="bg-card border border-border rounded-lg p-8 mb-8">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-6">
              <div className="w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center">
                <span className="text-2xl font-display font-bold text-primary">
                  {displayProfile.username?.charAt(0).toUpperCase()}
                </span>
              </div>
              <div>
                <h1 className="text-3xl font-display font-bold mb-2">
                  {displayProfile.display_name || displayProfile.username}
                </h1>
                <p className="text-muted-foreground mb-4">
                  {displayProfile.bio || 'No bio provided'}
                </p>
                <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                  <div className="flex items-center space-x-1">
                    <Calendar className="w-4 h-4" />
                    <span>
                      Joined {new Date(displayProfile.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  {displayProfile.primary_muse && (
                    <div className="flex items-center space-x-1">
                      <Target className="w-4 h-4" />
                      <span>
                        Guided by {displayProfile.primary_muse}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            {isOwnProfile && (
              <button className="flex items-center space-x-2 px-4 py-2 border border-input rounded-lg hover:bg-accent transition-colors">
                <Edit className="w-4 h-4" />
                <span>Edit Profile</span>
              </button>
            )}
          </div>
        </div>
        
        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-card border border-border rounded-lg p-6 text-center"
            >
              <div className="flex justify-center mb-3">
                <div className="p-3 bg-primary/10 rounded-full text-primary">
                  {stat.icon}
                </div>
              </div>
              <div className="text-2xl font-bold mb-1">{stat.value}</div>
              <div className="text-sm text-muted-foreground">{stat.label}</div>
            </motion.div>
          ))}
        </div>
        
        {/* Tabs */}
        <div className="flex justify-center mb-8">
          <div className="bg-card border border-border rounded-lg p-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  'flex items-center space-x-2 px-4 py-2 rounded-md transition-colors',
                  activeTab === tab.id
                    ? 'bg-primary text-primary-foreground'
                    : 'hover:bg-accent hover:text-accent-foreground'
                )}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
        
        {/* Tab Content */}
        <div className="space-y-6">
          {activeTab === 'creations' && (
            <div className="space-y-6">
              {creationsLoading.isLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="bg-card border border-border rounded-lg p-6 animate-pulse">
                      <div className="space-y-2 mb-4">
                        <div className="h-4 bg-muted rounded w-3/4"></div>
                        <div className="h-4 bg-muted rounded w-1/2"></div>
                      </div>
                      <div className="space-y-2">
                        <div className="h-3 bg-muted rounded"></div>
                        <div className="h-3 bg-muted rounded w-2/3"></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : creations.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {creations.map((creation, index) => (
                    <motion.div
                      key={creation.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-card border border-border rounded-lg p-6 hover:shadow-lg transition-shadow"
                    >
                      <div className="mb-4">
                        <h3 className="font-semibold mb-2">{creation.title}</h3>
                        <div className="text-sm text-muted-foreground mb-2">
                          {creation.form_type} • {creation.primary_theme}
                        </div>
                        <div className="prose prose-sm max-w-none">
                          {creation.content_preview}
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center space-x-4">
                          <div className="flex items-center space-x-1">
                            <Heart className="w-4 h-4 text-pink-500" />
                            <span>{creation.likes_count}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <MessageSquare className="w-4 h-4 text-blue-500" />
                            <span>{creation.comments_count}</span>
                          </div>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Target className="w-4 h-4 text-green-500" />
                          <span>{Math.round(creation.mathematical_fitness * 100)}%</span>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Sparkles className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Creations Yet</h3>
                  <p className="text-muted-foreground">
                    {isOwnProfile 
                      ? 'Start your creative journey by beginning a discovery session.'
                      : 'This user hasn\'t shared any creations yet.'
                    }
                  </p>
                </div>
              )}
            </div>
          )}
          
          {activeTab === 'signature' && (
            <div className="space-y-6">
              {displayProfile.harmonic_blend ? (
                <div className="bg-card border border-border rounded-lg p-6">
                  <h3 className="text-lg font-semibold mb-4">Frequency Signature</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium mb-3">Archetypal Blend</h4>
                      <div className="space-y-2">
                        {Object.entries(displayProfile.harmonic_blend).map(([muse, value]) => (
                          <div key={muse} className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span className="capitalize">{muse.toLowerCase()}</span>
                              <span className="font-medium">{Math.round(value * 100)}%</span>
                            </div>
                            <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-primary transition-all duration-500"
                                style={{ width: `${value * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium mb-3">Sacred Ratios</h4>
                      <div className="space-y-2">
                        {displayProfile.sacred_ratios && Object.entries(displayProfile.sacred_ratios).map(([ratio, value]) => (
                          <div key={ratio} className="flex justify-between text-sm">
                            <span className="capitalize">
                              {ratio === 'phi' ? 'φ (Golden Ratio)' : 
                               ratio === 'pi' ? 'π (Pi)' : 
                               ratio === 'euler' ? 'e (Euler)' : 
                               ratio.replace('_', ' ')}
                            </span>
                            <span className="font-mono">{value.toFixed(4)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <Target className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Frequency Signature</h3>
                  <p className="text-muted-foreground">
                    {isOwnProfile 
                      ? 'Complete your assessment to generate your frequency signature.'
                      : 'This user hasn\'t completed their assessment yet.'
                    }
                  </p>
                </div>
              )}
            </div>
          )}
          
          {activeTab === 'stats' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-card border border-border rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Discovery Performance</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Average Fitness Score</span>
                    <span className="text-lg font-semibold">
                      {Math.round((displayProfile.average_fitness_score || 0) * 100)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Total Discoveries</span>
                    <span className="text-lg font-semibold">
                      {displayProfile.total_discoveries || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Success Rate</span>
                    <span className="text-lg font-semibold">
                      85% {/* placeholder */}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="bg-card border border-border rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Preferences</h3>
                <div className="space-y-4">
                  <div>
                    <span className="text-sm text-muted-foreground block mb-2">Preferred Forms</span>
                    <div className="flex flex-wrap gap-2">
                      {displayProfile.preferred_forms?.map((form) => (
                        <span key={form} className="px-2 py-1 bg-primary/10 text-primary rounded-md text-xs">
                          {form}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <span className="text-sm text-muted-foreground block mb-2">Favorite Themes</span>
                    <div className="flex flex-wrap gap-2">
                      {displayProfile.favorite_themes?.map((theme) => (
                        <span key={theme} className="px-2 py-1 bg-secondary/50 text-secondary-foreground rounded-md text-xs">
                          {theme}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <span className="text-sm text-muted-foreground block mb-2">Discovery Style</span>
                    <span className="px-2 py-1 bg-accent/50 text-accent-foreground rounded-md text-xs capitalize">
                      {displayProfile.discovery_style}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default MuseProfilePage
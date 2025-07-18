/**
 * MUSE Community Page
 * 
 * Main community hub with resonant feed, kindred spirits discovery,
 * and collaborative creation features.
 */

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Users,
  Heart,
  MessageSquare,
  Share2,
  Filter,
  Search,
  Plus,
  TrendingUp,
  Star,
  Target,
} from 'lucide-react'
import { useMuseStore } from '@/stores/useMuseStore'
import { useResonantFeed, useKindredSpirits } from '@/hooks/useMuseAPI'
import { cn } from '@/utils/cn'

const MuseCommunityPage: React.FC = () => {
  const { user } = useMuseStore()
  const [activeTab, setActiveTab] = useState<'feed' | 'spirits' | 'collaborate'>('feed')
  
  const { feed, loading: feedLoading, loadMore } = useResonantFeed(user?.id)
  const { data: kindredSpirits, loading: spiritsLoading } = useKindredSpirits(user?.id)
  
  const tabs = [
    { id: 'feed', label: 'Resonant Feed', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'spirits', label: 'Kindred Spirits', icon: <Users className="w-4 h-4" /> },
    { id: 'collaborate', label: 'Collaborate', icon: <Plus className="w-4 h-4" /> },
  ]
  
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 mx-auto bg-primary/10 rounded-full flex items-center justify-center mb-4">
            <Users className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-3xl font-display font-bold mb-2">
            MUSE Community
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Connect with creators who share your archetypal frequencies and
            discover works through mathematical resonance.
          </p>
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
        
        {/* Content */}
        <div className="space-y-6">
          {activeTab === 'feed' && (
            <div className="space-y-6">
              {/* Feed Filters */}
              <div className="bg-card border border-border rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <Filter className="w-5 h-5 text-muted-foreground" />
                    <span className="text-sm font-medium">Filters</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      placeholder="Search creations..."
                      className="px-3 py-2 border border-input rounded-md bg-background text-sm"
                    />
                    <button className="p-2 border border-input rounded-md hover:bg-accent transition-colors">
                      <Search className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
              
              {/* Feed Items */}
              <div className="space-y-4">
                {feedLoading.isLoading ? (
                  <div className="space-y-4">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="bg-card border border-border rounded-lg p-6 animate-pulse">
                        <div className="flex items-center space-x-3 mb-4">
                          <div className="w-8 h-8 bg-muted rounded-full"></div>
                          <div className="flex-1">
                            <div className="h-4 bg-muted rounded w-1/4 mb-2"></div>
                            <div className="h-3 bg-muted rounded w-1/3"></div>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <div className="h-4 bg-muted rounded"></div>
                          <div className="h-4 bg-muted rounded w-3/4"></div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : feed.length > 0 ? (
                  <>
                    {feed.map((item, index) => (
                      <motion.div
                        key={`${item.creation.id}-${index}`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-card border border-border rounded-lg p-6 hover:shadow-lg transition-shadow"
                      >
                        {/* Creator Info */}
                        <div className="flex items-center space-x-3 mb-4">
                          <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                            <span className="text-sm font-medium text-primary">
                              {item.creation.creator_username?.charAt(0).toUpperCase()}
                            </span>
                          </div>
                          <div>
                            <h4 className="font-medium">{item.creation.creator_display_name || item.creation.creator_username}</h4>
                            <p className="text-sm text-muted-foreground">
                              {item.creation.creator_primary_muse} • {new Date(item.creation.created_at).toLocaleDateString()}
                            </p>
                          </div>
                          <div className="ml-auto">
                            <div className="flex items-center space-x-1 text-sm">
                              <Heart className="w-4 h-4 text-pink-500" />
                              <span className="font-medium">{Math.round(item.resonance_score * 100)}%</span>
                            </div>
                          </div>
                        </div>
                        
                        {/* Content */}
                        <div className="mb-4">
                          <h3 className="font-semibold mb-2">{item.creation.title}</h3>
                          <div className="prose prose-sm max-w-none">
                            {item.creation.content_preview}
                          </div>
                        </div>
                        
                        {/* Metadata */}
                        <div className="flex items-center justify-between text-sm text-muted-foreground mb-4">
                          <div className="flex items-center space-x-4">
                            <span>{item.creation.form_type}</span>
                            <span>•</span>
                            <span>{item.creation.primary_theme}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Target className="w-4 h-4" />
                            <span>{Math.round(item.creation.mathematical_fitness * 100)}%</span>
                          </div>
                        </div>
                        
                        {/* Actions */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <button className="flex items-center space-x-1 text-sm hover:text-primary transition-colors">
                              <Heart className="w-4 h-4" />
                              <span>{item.creation.likes_count}</span>
                            </button>
                            <button className="flex items-center space-x-1 text-sm hover:text-primary transition-colors">
                              <MessageSquare className="w-4 h-4" />
                              <span>{item.creation.comments_count}</span>
                            </button>
                            <button className="flex items-center space-x-1 text-sm hover:text-primary transition-colors">
                              <Share2 className="w-4 h-4" />
                              <span>Share</span>
                            </button>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Resonance: {item.resonance_reasons.join(', ')}
                          </div>
                        </div>
                      </motion.div>
                    ))}
                    
                    {/* Load More */}
                    <div className="text-center">
                      <button
                        onClick={loadMore}
                        className="px-6 py-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 transition-colors"
                      >
                        Load More
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-12">
                    <Users className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-semibold mb-2">No Resonant Creations</h3>
                    <p className="text-muted-foreground">
                      Check back soon for creations that match your frequency signature.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {activeTab === 'spirits' && (
            <div className="space-y-6">
              {spiritsLoading.isLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {[1, 2, 3, 4, 5, 6].map((i) => (
                    <div key={i} className="bg-card border border-border rounded-lg p-6 animate-pulse">
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="w-12 h-12 bg-muted rounded-full"></div>
                        <div className="flex-1">
                          <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
                          <div className="h-3 bg-muted rounded w-1/2"></div>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="h-3 bg-muted rounded"></div>
                        <div className="h-3 bg-muted rounded w-2/3"></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : kindredSpirits && kindredSpirits.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {kindredSpirits.map((spirit, index) => (
                    <motion.div
                      key={spirit.user_id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-card border border-border rounded-lg p-6 hover:shadow-lg transition-shadow"
                    >
                      <div className="flex items-center space-x-3 mb-4">
                        <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
                          <span className="text-lg font-medium text-primary">
                            {spirit.username?.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <div>
                          <h4 className="font-medium">{spirit.display_name || spirit.username}</h4>
                          <p className="text-sm text-muted-foreground">
                            {spirit.primary_muse} • {spirit.secondary_muse}
                          </p>
                        </div>
                      </div>
                      
                      <div className="mb-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">Resonance Score</span>
                          <span className="text-sm font-bold text-primary">
                            {Math.round(spirit.resonance_score * 100)}%
                          </span>
                        </div>
                        <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-primary transition-all duration-500"
                            style={{ width: `${spirit.resonance_score * 100}%` }}
                          />
                        </div>
                      </div>
                      
                      <div className="space-y-2 mb-4">
                        <div className="text-sm">
                          <span className="text-muted-foreground">Shared themes:</span>
                          <span className="ml-1">{spirit.shared_themes.join(', ')}</span>
                        </div>
                        <div className="text-sm">
                          <span className="text-muted-foreground">Compatibility:</span>
                          <span className="ml-1">{spirit.compatibility_reasons.join(', ')}</span>
                        </div>
                      </div>
                      
                      <div className="flex space-x-2">
                        <button className="flex-1 px-3 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors text-sm">
                          Connect
                        </button>
                        <button className="flex-1 px-3 py-2 border border-input rounded-md hover:bg-accent transition-colors text-sm">
                          View Profile
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Users className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Kindred Spirits Found</h3>
                  <p className="text-muted-foreground">
                    We're still calculating resonance with other creators. Check back soon!
                  </p>
                </div>
              )}
            </div>
          )}
          
          {activeTab === 'collaborate' && (
            <div className="text-center py-12">
              <Plus className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">Collaborative Creation</h3>
              <p className="text-muted-foreground mb-6">
                Multi-user creative discovery sessions coming soon!
              </p>
              <button className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
                Join Beta Program
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default MuseCommunityPage
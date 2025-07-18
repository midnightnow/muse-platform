/**
 * MUSE Platform Navigation Component
 * 
 * Provides the main navigation sidebar with archetypal theming
 * and dynamic menu items based on user state.
 */

import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Home,
  Compass,
  Palette,
  Users,
  User,
  Settings,
  LogOut,
  Sparkles,
  Activity,
  MessageSquare,
  Target,
  Zap,
} from 'lucide-react'
import { useMuseStore, useMuseActions } from '@/stores/useMuseStore'
import { cn } from '@/utils/cn'

const Navigation: React.FC = () => {
  const { user, currentSignature, theme, notifications } = useMuseStore()
  const { logout } = useMuseActions()
  const location = useLocation()
  
  const unreadCount = notifications.filter(n => !n.read).length
  
  // Navigation items based on user state
  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: Home,
      description: 'Overview of your creative journey',
    },
    {
      name: 'Discovery',
      href: '/discovery',
      icon: Compass,
      description: 'Start new creative discovery sessions',
    },
    {
      name: 'Frequency Signature',
      href: '/signature',
      icon: Sparkles,
      description: 'View and tune your archetypal signature',
    },
    {
      name: 'Community',
      href: '/community',
      icon: Users,
      description: 'Connect with kindred spirits',
    },
    {
      name: 'Profile',
      href: '/profile',
      icon: User,
      description: 'Your profile and creations',
    },
  ]
  
  // Quick stats for the sidebar
  const stats = [
    {
      label: 'Primary Muse',
      value: currentSignature?.primary_muse || 'Unknown',
      icon: Target,
    },
    {
      label: 'Discoveries',
      value: user?.total_discoveries || 0,
      icon: Activity,
    },
    {
      label: 'Resonance',
      value: `${Math.round((user?.average_fitness_score || 0) * 100)}%`,
      icon: Zap,
    },
  ]
  
  const handleLogout = () => {
    logout()
  }
  
  return (
    <motion.nav
      initial={{ x: -256 }}
      animate={{ x: 0 }}
      className="fixed inset-y-0 left-0 z-50 w-64 bg-card border-r border-border"
    >
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-lg font-display font-semibold">MUSE</h1>
              <p className="text-xs text-muted-foreground">
                Computational Platonism
              </p>
            </div>
          </div>
          
          {/* Notifications badge */}
          {unreadCount > 0 && (
            <div className="w-5 h-5 bg-destructive text-destructive-foreground rounded-full flex items-center justify-center text-xs font-medium">
              {unreadCount > 9 ? '9+' : unreadCount}
            </div>
          )}
        </div>
        
        {/* User info */}
        <div className="p-6 border-b border-border">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center">
              <span className="text-sm font-medium text-primary">
                {user?.username?.charAt(0).toUpperCase() || 'U'}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">
                {user?.display_name || user?.username}
              </p>
              <p className="text-xs text-muted-foreground truncate">
                {currentSignature?.primary_muse && (
                  <span className="inline-flex items-center space-x-1">
                    <span>Guided by</span>
                    <span className="font-medium text-primary">
                      {currentSignature.primary_muse}
                    </span>
                  </span>
                )}
              </p>
            </div>
          </div>
        </div>
        
        {/* Navigation items */}
        <div className="flex-1 p-4 space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.href || 
                           (item.href !== '/' && location.pathname.startsWith(item.href))
            
            return (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) =>
                  cn(
                    'flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'text-foreground hover:bg-accent hover:text-accent-foreground'
                  )
                }
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                <span className="flex-1">{item.name}</span>
              </NavLink>
            )
          })}
        </div>
        
        {/* Stats */}
        <div className="p-4 border-t border-border">
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
            Quick Stats
          </h3>
          <div className="space-y-2">
            {stats.map((stat) => {
              const Icon = stat.icon
              return (
                <div
                  key={stat.label}
                  className="flex items-center justify-between text-sm"
                >
                  <div className="flex items-center space-x-2">
                    <Icon className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">{stat.label}</span>
                  </div>
                  <span className="font-medium">{stat.value}</span>
                </div>
              )
            })}
          </div>
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center justify-between">
            <NavLink
              to="/settings"
              className="flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium text-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </NavLink>
            
            <button
              onClick={handleLogout}
              className="flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium text-muted-foreground hover:bg-destructive hover:text-destructive-foreground transition-colors"
            >
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>
    </motion.nav>
  )
}

export default Navigation
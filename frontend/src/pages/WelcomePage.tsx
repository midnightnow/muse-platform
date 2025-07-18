/**
 * Welcome Page Component
 * 
 * The landing page for new users, introducing the concept of
 * Computational Platonism and the MUSE Platform.
 */

import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Sparkles,
  Target,
  Users,
  BookOpen,
  ArrowRight,
  Play,
  ChevronDown,
} from 'lucide-react'
import { useCreateProfile } from '@/hooks/useMuseAPI'
import { useMuseActions } from '@/stores/useMuseStore'
import { ProfileCreateRequest } from '@/types'

const WelcomePage: React.FC = () => {
  const navigate = useNavigate()
  const { execute: createProfile, loading } = useCreateProfile()
  const { login, handleError, showSuccess } = useMuseActions()
  
  const [showSignup, setShowSignup] = useState(false)
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    display_name: '',
  })
  
  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault()
    
    try {
      const profileData: ProfileCreateRequest = {
        username: formData.username,
        email: formData.email,
        display_name: formData.display_name,
        preferred_forms: [],
        favorite_themes: [],
        discovery_style: 'balanced',
      }
      
      const profile = await createProfile(profileData)
      login(profile)
      showSuccess('Welcome to MUSE! Begin your archetypal assessment.')
      navigate('/assessment')
    } catch (error) {
      handleError(error, 'Profile creation failed')
    }
  }
  
  const features = [
    {
      icon: <Target className="w-8 h-8" />,
      title: 'Frequency Signatures',
      description: 'Discover your unique archetypal frequencies based on the 12 classical Muses',
    },
    {
      icon: <Sparkles className="w-8 h-8" />,
      title: 'Creative Discovery',
      description: 'Find creative works that already exist in the mathematical realm',
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: 'Resonant Community',
      description: 'Connect with kindred spirits through archetypal compatibility',
    },
    {
      icon: <BookOpen className="w-8 h-8" />,
      title: 'Sacred Geometry',
      description: 'Apply mathematical constants to structure your creative works',
    },
  ]
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/10">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Sacred geometry background */}
        <div className="absolute inset-0 bg-sacred-pattern opacity-5"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-32">
          <div className="text-center">
            {/* Logo */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="flex justify-center mb-8"
            >
              <div className="w-16 h-16 bg-primary rounded-xl flex items-center justify-center shadow-lg">
                <Sparkles className="w-8 h-8 text-primary-foreground" />
              </div>
            </motion.div>
            
            {/* Title */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="text-5xl md:text-7xl font-display font-bold mb-6"
            >
              <span className="bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                MUSE
              </span>
              <br />
              <span className="text-3xl md:text-4xl text-muted-foreground">
                Platform
              </span>
            </motion.h1>
            
            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto"
            >
              Discover your creative essence through{' '}
              <span className="text-primary font-medium">Computational Platonism</span>
              {' '}— where creativity is mathematical discovery, not generation.
            </motion.p>
            
            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="flex flex-col sm:flex-row gap-4 justify-center"
            >
              <button
                onClick={() => setShowSignup(true)}
                className="inline-flex items-center justify-center px-8 py-4 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors font-medium text-lg"
              >
                Begin Discovery
                <ArrowRight className="w-5 h-5 ml-2" />
              </button>
              
              <button
                onClick={() => {
                  document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })
                }}
                className="inline-flex items-center justify-center px-8 py-4 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 transition-colors font-medium text-lg"
              >
                <Play className="w-5 h-5 mr-2" />
                Learn More
              </button>
            </motion.div>
          </div>
        </div>
        
        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        >
          <ChevronDown className="w-6 h-6 text-muted-foreground animate-bounce" />
        </motion.div>
      </section>
      
      {/* Features Section */}
      <section id="features" className="py-20 bg-accent/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-display font-bold mb-4">
              Sacred Mathematics of Creativity
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              MUSE applies the principles of Computational Platonism to help you discover
              the creative works that already exist in the mathematical realm.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                className="bg-card border border-border rounded-lg p-6 hover:shadow-lg transition-shadow"
              >
                <div className="text-primary mb-4">{feature.icon}</div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      
      {/* Philosophy Section */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-display font-bold mb-8">
            The Philosophy of Discovery
          </h2>
          
          <div className="prose prose-lg max-w-none">
            <p className="text-xl text-muted-foreground mb-6">
              Unlike traditional AI that generates content, MUSE operates on the principle
              that all creative works already exist in a mathematical realm of eternal forms.
            </p>
            
            <div className="bg-card border border-border rounded-lg p-8 mt-8">
              <blockquote className="text-2xl font-display italic text-primary mb-4">
                "Creativity is not generation, but discovery."
              </blockquote>
              <p className="text-muted-foreground">
                Through archetypal frequencies, sacred geometry, and semantic projections,
                MUSE guides you to the creative works that resonate with your unique
                mathematical signature.
              </p>
            </div>
          </div>
        </div>
      </section>
      
      {/* Signup Modal */}
      {showSignup && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-card border border-border rounded-lg shadow-xl max-w-md w-full p-6"
          >
            <div className="text-center mb-6">
              <h3 className="text-2xl font-display font-bold mb-2">
                Begin Your Journey
              </h3>
              <p className="text-muted-foreground">
                Create your profile to start discovering your archetypal frequencies
              </p>
            </div>
            
            <form onSubmit={handleSignup} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Username</label>
                <input
                  type="text"
                  required
                  className="w-full px-3 py-2 border border-input rounded-md bg-background"
                  value={formData.username}
                  onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Email</label>
                <input
                  type="email"
                  required
                  className="w-full px-3 py-2 border border-input rounded-md bg-background"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Display Name</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-input rounded-md bg-background"
                  value={formData.display_name}
                  onChange={(e) => setFormData({ ...formData, display_name: e.target.value })}
                />
              </div>
              
              <div className="flex space-x-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowSignup(false)}
                  className="flex-1 px-4 py-2 border border-input rounded-md hover:bg-accent transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading.isLoading}
                  className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors disabled:opacity-50"
                >
                  {loading.isLoading ? 'Creating...' : 'Create Profile'}
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </div>
  )
}

export default WelcomePage
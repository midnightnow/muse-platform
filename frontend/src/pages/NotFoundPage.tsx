/**
 * 404 Not Found Page
 * 
 * A mathematically-themed 404 page that maintains the MUSE aesthetic
 * while helping users navigate back to the main application.
 */

import React from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Home, ArrowLeft, Compass, Search } from 'lucide-react'

const NotFoundPage: React.FC = () => {
  const navigate = useNavigate()
  
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="max-w-2xl w-full text-center">
        {/* Sacred geometry 404 */}
        <div className="mb-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            className="relative inline-block"
          >
            {/* Golden ratio spiral background */}
            <div className="absolute inset-0 -z-10">
              <svg
                width="300"
                height="200"
                viewBox="0 0 300 200"
                className="w-full h-auto opacity-10"
              >
                <path
                  d="M 150 100 Q 200 100 200 150 Q 200 200 150 200 Q 100 200 100 150 Q 100 100 150 100 Q 175 100 175 125 Q 175 150 150 150 Q 125 150 125 125"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-primary"
                />
              </svg>
            </div>
            
            {/* 404 Text */}
            <div className="text-8xl md:text-9xl font-display font-bold text-primary/80">
              4<span className="text-secondary">0</span>4
            </div>
          </motion.div>
        </div>
        
        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-3xl md:text-4xl font-display font-bold mb-4"
        >
          Mathematical Void Detected
        </motion.h1>
        
        {/* Description */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-xl text-muted-foreground mb-8"
        >
          The archetypal frequencies you seek do not exist in this dimensional space.
          This pattern has not been discovered in the mathematical realm.
        </motion.p>
        
        {/* Mathematical quote */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="bg-card border border-border rounded-lg p-6 mb-8"
        >
          <blockquote className="text-lg font-display italic text-primary mb-2">
            "In the mathematics of discovery, undefined paths lead to new dimensions."
          </blockquote>
          <p className="text-sm text-muted-foreground">
            — Computational Platonism Axiom π.404
          </p>
        </motion.div>
        
        {/* Navigation options */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <button
            onClick={() => navigate('/')}
            className="inline-flex items-center justify-center px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors font-medium"
          >
            <Home className="w-5 h-5 mr-2" />
            Return to Origin
          </button>
          
          <button
            onClick={() => navigate(-1)}
            className="inline-flex items-center justify-center px-6 py-3 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 transition-colors font-medium"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Previous Dimension
          </button>
          
          <button
            onClick={() => navigate('/discovery')}
            className="inline-flex items-center justify-center px-6 py-3 border border-input rounded-lg hover:bg-accent hover:text-accent-foreground transition-colors font-medium"
          >
            <Compass className="w-5 h-5 mr-2" />
            New Discovery
          </button>
        </motion.div>
        
        {/* Sacred geometry decoration */}
        <motion.div
          initial={{ opacity: 0, rotate: -180 }}
          animate={{ opacity: 1, rotate: 0 }}
          transition={{ duration: 1.2, delay: 1 }}
          className="mt-12 flex justify-center"
        >
          <div className="w-32 h-32 relative">
            <div className="absolute inset-0 border-2 border-primary/20 rounded-full"></div>
            <div className="absolute inset-4 border-2 border-primary/40 rounded-full"></div>
            <div className="absolute inset-8 border-2 border-primary/60 rounded-full"></div>
            <div className="absolute inset-12 w-8 h-8 bg-primary/80 rounded-full"></div>
          </div>
        </motion.div>
        
        {/* Help text */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.2 }}
          className="mt-8 text-sm text-muted-foreground"
        >
          <p>
            If you believe this is an error in the mathematical matrix,
            please check the URL or contact our dimensional support team.
          </p>
        </motion.div>
      </div>
    </div>
  )
}

export default NotFoundPage
/**
 * MUSE Platform Main Application Component
 * 
 * The root component that provides routing, theming, and global state
 * management for the entire MUSE Platform application.
 */

import React, { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { useMuseStore, useMuseActions } from '@/stores/useMuseStore'
import { useWebSocket } from '@/hooks/useMuseAPI'
import ArchetypalThemeProvider from './ui/ArchetypalThemeProvider'
import ErrorBoundary from './ui/ErrorBoundary'
import LoadingStates from './ui/LoadingStates'
import Navigation from './Navigation'
import MuseAssessment from './MuseAssessment'
import MuseDiscoveryInterface from './MuseDiscoveryInterface'
import MuseFrequencyDisplay from './MuseFrequencyDisplay'
import MuseCreativeSession from './MuseCreativeSession'
import MuseCommunityPage from '@/pages/MuseCommunityPage'
import MuseProfilePage from '@/pages/MuseProfilePage'
import MuseDashboardPage from '@/pages/MuseDashboardPage'
import WelcomePage from '@/pages/WelcomePage'
import NotFoundPage from '@/pages/NotFoundPage'

const MuseApp: React.FC = () => {
  const {
    user,
    isAuthenticated,
    currentSignature,
    theme,
    loading,
    error,
    notifications,
  } = useMuseStore()
  
  const { handleError } = useMuseActions()
  
  // Initialize WebSocket connection for authenticated users
  const { connected, messages } = useWebSocket(user?.id)
  
  // Handle WebSocket messages
  useEffect(() => {
    if (messages.length > 0) {
      const latestMessage = messages[messages.length - 1]
      
      // Handle different message types
      switch (latestMessage.type) {
        case 'discovery_update':
          // Handle discovery updates
          break
        case 'collaboration_update':
          // Handle collaboration updates
          break
        case 'notification':
          // Handle notifications
          break
        default:
          break
      }
    }
  }, [messages])
  
  // Protected route wrapper
  const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    if (!isAuthenticated) {
      return <Navigate to="/welcome" replace />
    }
    return <>{children}</>
  }
  
  // Assessment required route wrapper
  const AssessmentRequiredRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    if (!isAuthenticated) {
      return <Navigate to="/welcome" replace />
    }
    
    if (!currentSignature) {
      return <Navigate to="/assessment" replace />
    }
    
    return <>{children}</>
  }
  
  return (
    <ErrorBoundary>
      <ArchetypalThemeProvider theme={theme}>
        <div className="min-h-screen bg-background text-foreground">
          {/* Global Loading Overlay */}
          {loading.isLoading && (
            <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
              <div className="absolute inset-0 flex items-center justify-center">
                <LoadingStates
                  isLoading={loading.isLoading}
                  phase={loading.phase}
                  progress={loading.progress}
                  message={loading.message}
                />
              </div>
            </div>
          )}
          
          {/* Navigation */}
          {isAuthenticated && <Navigation />}
          
          {/* Main Content */}
          <main className={isAuthenticated ? 'pl-64' : ''}>
            <AnimatePresence mode="wait">
              <Routes>
                {/* Public Routes */}
                <Route
                  path="/welcome"
                  element={
                    isAuthenticated ? (
                      <Navigate to="/dashboard" replace />
                    ) : (
                      <WelcomePage />
                    )
                  }
                />
                
                {/* Authentication Required Routes */}
                <Route
                  path="/assessment"
                  element={
                    <ProtectedRoute>
                      <MuseAssessment />
                    </ProtectedRoute>
                  }
                />
                
                {/* Assessment Required Routes */}
                <Route
                  path="/dashboard"
                  element={
                    <AssessmentRequiredRoute>
                      <MuseDashboardPage />
                    </AssessmentRequiredRoute>
                  }
                />
                
                <Route
                  path="/discovery"
                  element={
                    <AssessmentRequiredRoute>
                      <MuseDiscoveryInterface />
                    </AssessmentRequiredRoute>
                  }
                />
                
                <Route
                  path="/discovery/session/:sessionId"
                  element={
                    <AssessmentRequiredRoute>
                      <MuseCreativeSession />
                    </AssessmentRequiredRoute>
                  }
                />
                
                <Route
                  path="/signature"
                  element={
                    <AssessmentRequiredRoute>
                      <MuseFrequencyDisplay />
                    </AssessmentRequiredRoute>
                  }
                />
                
                <Route
                  path="/signature/:signatureId"
                  element={
                    <AssessmentRequiredRoute>
                      <MuseFrequencyDisplay />
                    </AssessmentRequiredRoute>
                  }
                />
                
                <Route
                  path="/community"
                  element={
                    <AssessmentRequiredRoute>
                      <MuseCommunityPage />
                    </AssessmentRequiredRoute>
                  }
                />
                
                <Route
                  path="/community/profile/:userId"
                  element={
                    <AssessmentRequiredRoute>
                      <MuseProfilePage />
                    </AssessmentRequiredRoute>
                  }
                />
                
                <Route
                  path="/profile"
                  element={
                    <AssessmentRequiredRoute>
                      <MuseProfilePage />
                    </AssessmentRequiredRoute>
                  }
                />
                
                {/* Default Routes */}
                <Route
                  path="/"
                  element={
                    isAuthenticated ? (
                      currentSignature ? (
                        <Navigate to="/dashboard" replace />
                      ) : (
                        <Navigate to="/assessment" replace />
                      )
                    ) : (
                      <Navigate to="/welcome" replace />
                    )
                  }
                />
                
                {/* 404 Route */}
                <Route path="*" element={<NotFoundPage />} />
              </Routes>
            </AnimatePresence>
          </main>
          
          {/* Connection Status Indicator */}
          {isAuthenticated && (
            <div className="fixed bottom-4 right-4 z-40">
              <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                connected 
                  ? 'bg-green-100 text-green-800 border border-green-200' 
                  : 'bg-red-100 text-red-800 border border-red-200'
              }`}>
                {connected ? 'Connected' : 'Disconnected'}
              </div>
            </div>
          )}
          
          {/* Global Error Display */}
          {error.hasError && (
            <div className="fixed bottom-4 left-4 z-50 max-w-md">
              <div className="bg-destructive text-destructive-foreground rounded-lg p-4 shadow-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-semibold">Error</h4>
                    <p className="text-sm mt-1">{error.message}</p>
                  </div>
                  <button
                    onClick={() => useMuseStore.getState().clearError()}
                    className="ml-4 text-destructive-foreground/70 hover:text-destructive-foreground"
                  >
                    ×
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </ArchetypalThemeProvider>
    </ErrorBoundary>
  )
}

export default MuseApp
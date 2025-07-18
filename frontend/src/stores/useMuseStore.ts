/**
 * MUSE Platform State Management
 * 
 * Zustand-based state management for the MUSE Platform,
 * handling user authentication, frequency signatures,
 * discovery sessions, and community interactions.
 */

import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import {
  UserProfile,
  FrequencySignature,
  DiscoverySession,
  CommunityCreation,
  ResonantFeedItem,
  KindredSpirit,
  CollaborativeSession,
  ThemeConfig,
  LoadingState,
  ErrorState,
  MuseArchetype,
  WebSocketMessage,
} from '@/types'

// ============================================================================
// Theme Configuration
// ============================================================================

const getThemeForMuse = (muse: MuseArchetype): ThemeConfig => {
  const themes: Record<MuseArchetype, ThemeConfig> = {
    CALLIOPE: {
      primary_muse: 'CALLIOPE',
      color_palette: {
        primary: '#9333ea',
        secondary: '#c084fc',
        accent: '#f3e8ff',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'serif',
      sacred_geometry_pattern: 'golden-spiral',
      animation_style: 'dynamic',
    },
    CLIO: {
      primary_muse: 'CLIO',
      color_palette: {
        primary: '#dc2626',
        secondary: '#f87171',
        accent: '#fee2e2',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'serif',
      sacred_geometry_pattern: 'fibonacci-spiral',
      animation_style: 'moderate',
    },
    ERATO: {
      primary_muse: 'ERATO',
      color_palette: {
        primary: '#ec4899',
        secondary: '#f472b6',
        accent: '#fce7f3',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'serif',
      sacred_geometry_pattern: 'heart-spiral',
      animation_style: 'dynamic',
    },
    EUTERPE: {
      primary_muse: 'EUTERPE',
      color_palette: {
        primary: '#059669',
        secondary: '#34d399',
        accent: '#d1fae5',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'sans',
      sacred_geometry_pattern: 'wave-pattern',
      animation_style: 'dynamic',
    },
    MELPOMENE: {
      primary_muse: 'MELPOMENE',
      color_palette: {
        primary: '#1e40af',
        secondary: '#60a5fa',
        accent: '#dbeafe',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'serif',
      sacred_geometry_pattern: 'dramatic-spiral',
      animation_style: 'subtle',
    },
    POLYHYMNIA: {
      primary_muse: 'POLYHYMNIA',
      color_palette: {
        primary: '#7c3aed',
        secondary: '#a78bfa',
        accent: '#ede9fe',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'serif',
      sacred_geometry_pattern: 'sacred-geometry',
      animation_style: 'subtle',
    },
    TERPSICHORE: {
      primary_muse: 'TERPSICHORE',
      color_palette: {
        primary: '#ea580c',
        secondary: '#fb923c',
        accent: '#fed7aa',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'sans',
      sacred_geometry_pattern: 'dance-spiral',
      animation_style: 'dynamic',
    },
    THALIA: {
      primary_muse: 'THALIA',
      color_palette: {
        primary: '#eab308',
        secondary: '#facc15',
        accent: '#fef3c7',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'sans',
      sacred_geometry_pattern: 'joy-spiral',
      animation_style: 'dynamic',
    },
    URANIA: {
      primary_muse: 'URANIA',
      color_palette: {
        primary: '#0ea5e9',
        secondary: '#38bdf8',
        accent: '#e0f2fe',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'mono',
      sacred_geometry_pattern: 'cosmic-spiral',
      animation_style: 'moderate',
    },
    SOPHIA: {
      primary_muse: 'SOPHIA',
      color_palette: {
        primary: '#6366f1',
        secondary: '#818cf8',
        accent: '#e0e7ff',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'serif',
      sacred_geometry_pattern: 'wisdom-spiral',
      animation_style: 'subtle',
    },
    TECHNE: {
      primary_muse: 'TECHNE',
      color_palette: {
        primary: '#8b5cf6',
        secondary: '#a78bfa',
        accent: '#ede9fe',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'mono',
      sacred_geometry_pattern: 'craft-pattern',
      animation_style: 'moderate',
    },
    PSYCHE: {
      primary_muse: 'PSYCHE',
      color_palette: {
        primary: '#d946ef',
        secondary: '#e879f9',
        accent: '#fae8ff',
        background: '#fafafa',
        text: '#1f2937',
        border: '#e5e7eb',
      },
      font_family: 'serif',
      sacred_geometry_pattern: 'soul-spiral',
      animation_style: 'dynamic',
    },
  }
  
  return themes[muse]
}

// ============================================================================
// Main Application Store
// ============================================================================

interface MuseStore {
  // User Authentication & Profile
  user: UserProfile | null
  isAuthenticated: boolean
  
  // Frequency Signature
  currentSignature: FrequencySignature | null
  signatureHistory: FrequencySignature[]
  
  // Theme & UI
  theme: ThemeConfig
  isDarkMode: boolean
  
  // Loading & Error States
  loading: LoadingState
  error: ErrorState
  
  // Notifications
  notifications: Array<{
    id: string
    type: 'info' | 'success' | 'warning' | 'error'
    message: string
    timestamp: string
    read: boolean
  }>
  
  // Actions
  setUser: (user: UserProfile | null) => void
  setSignature: (signature: FrequencySignature | null) => void
  addSignatureToHistory: (signature: FrequencySignature) => void
  setTheme: (theme: ThemeConfig) => void
  updateThemeForMuse: (muse: MuseArchetype) => void
  toggleDarkMode: () => void
  setLoading: (loading: LoadingState) => void
  setError: (error: ErrorState) => void
  clearError: () => void
  addNotification: (notification: Omit<MuseStore['notifications'][0], 'id' | 'timestamp' | 'read'>) => void
  markNotificationRead: (id: string) => void
  clearNotifications: () => void
  reset: () => void
}

export const useMuseStore = create<MuseStore>()(
  persist(
    immer((set, get) => ({
      // Initial state
      user: null,
      isAuthenticated: false,
      currentSignature: null,
      signatureHistory: [],
      theme: getThemeForMuse('CALLIOPE'),
      isDarkMode: false,
      loading: { isLoading: false },
      error: { hasError: false },
      notifications: [],
      
      // Actions
      setUser: (user) => set((state) => {
        state.user = user
        state.isAuthenticated = !!user
        
        // Update theme based on user's primary muse
        if (user?.primary_muse) {
          state.theme = getThemeForMuse(user.primary_muse)
        }
      }),
      
      setSignature: (signature) => set((state) => {
        state.currentSignature = signature
        
        // Update theme based on signature's primary muse
        if (signature?.primary_muse) {
          state.theme = getThemeForMuse(signature.primary_muse)
        }
      }),
      
      addSignatureToHistory: (signature) => set((state) => {
        const existingIndex = state.signatureHistory.findIndex(
          s => s.id === signature.id
        )
        
        if (existingIndex >= 0) {
          state.signatureHistory[existingIndex] = signature
        } else {
          state.signatureHistory.push(signature)
        }
        
        // Keep only last 10 signatures
        if (state.signatureHistory.length > 10) {
          state.signatureHistory = state.signatureHistory.slice(-10)
        }
      }),
      
      setTheme: (theme) => set((state) => {
        state.theme = theme
      }),
      
      updateThemeForMuse: (muse) => set((state) => {
        state.theme = getThemeForMuse(muse)
      }),
      
      toggleDarkMode: () => set((state) => {
        state.isDarkMode = !state.isDarkMode
      }),
      
      setLoading: (loading) => set((state) => {
        state.loading = loading
      }),
      
      setError: (error) => set((state) => {
        state.error = error
      }),
      
      clearError: () => set((state) => {
        state.error = { hasError: false }
      }),
      
      addNotification: (notification) => set((state) => {
        const newNotification = {
          ...notification,
          id: crypto.randomUUID(),
          timestamp: new Date().toISOString(),
          read: false,
        }
        state.notifications.unshift(newNotification)
        
        // Keep only last 50 notifications
        if (state.notifications.length > 50) {
          state.notifications = state.notifications.slice(0, 50)
        }
      }),
      
      markNotificationRead: (id) => set((state) => {
        const notification = state.notifications.find(n => n.id === id)
        if (notification) {
          notification.read = true
        }
      }),
      
      clearNotifications: () => set((state) => {
        state.notifications = []
      }),
      
      reset: () => set((state) => {
        state.user = null
        state.isAuthenticated = false
        state.currentSignature = null
        state.signatureHistory = []
        state.theme = getThemeForMuse('CALLIOPE')
        state.isDarkMode = false
        state.loading = { isLoading: false }
        state.error = { hasError: false }
        state.notifications = []
      }),
    })),
    {
      name: 'muse-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
        currentSignature: state.currentSignature,
        signatureHistory: state.signatureHistory,
        theme: state.theme,
        isDarkMode: state.isDarkMode,
        notifications: state.notifications,
      }),
    }
  )
)

// ============================================================================
// Discovery Session Store
// ============================================================================

interface DiscoveryStore {
  currentSession: DiscoverySession | null
  sessionHistory: DiscoverySession[]
  activeCollaborations: CollaborativeSession[]
  
  // Real-time updates
  liveUpdates: WebSocketMessage[]
  
  // Loading & Error States
  loading: LoadingState
  error: ErrorState
  
  // Actions
  setCurrentSession: (session: DiscoverySession | null) => void
  addSessionToHistory: (session: DiscoverySession) => void
  setActiveCollaborations: (collaborations: CollaborativeSession[]) => void
  addLiveUpdate: (update: WebSocketMessage) => void
  clearLiveUpdates: () => void
  setLoading: (loading: LoadingState) => void
  setError: (error: ErrorState) => void
  clearError: () => void
  reset: () => void
}

export const useDiscoveryStore = create<DiscoveryStore>()(
  immer((set, get) => ({
    currentSession: null,
    sessionHistory: [],
    activeCollaborations: [],
    liveUpdates: [],
    loading: { isLoading: false },
    error: { hasError: false },
    
    setCurrentSession: (session) => set((state) => {
      state.currentSession = session
    }),
    
    addSessionToHistory: (session) => set((state) => {
      const existingIndex = state.sessionHistory.findIndex(
        s => s.session_id === session.session_id
      )
      
      if (existingIndex >= 0) {
        state.sessionHistory[existingIndex] = session
      } else {
        state.sessionHistory.unshift(session)
      }
      
      // Keep only last 20 sessions
      if (state.sessionHistory.length > 20) {
        state.sessionHistory = state.sessionHistory.slice(0, 20)
      }
    }),
    
    setActiveCollaborations: (collaborations) => set((state) => {
      state.activeCollaborations = collaborations
    }),
    
    addLiveUpdate: (update) => set((state) => {
      state.liveUpdates.push(update)
      
      // Keep only last 100 updates
      if (state.liveUpdates.length > 100) {
        state.liveUpdates = state.liveUpdates.slice(-100)
      }
    }),
    
    clearLiveUpdates: () => set((state) => {
      state.liveUpdates = []
    }),
    
    setLoading: (loading) => set((state) => {
      state.loading = loading
    }),
    
    setError: (error) => set((state) => {
      state.error = error
    }),
    
    clearError: () => set((state) => {
      state.error = { hasError: false }
    }),
    
    reset: () => set((state) => {
      state.currentSession = null
      state.sessionHistory = []
      state.activeCollaborations = []
      state.liveUpdates = []
      state.loading = { isLoading: false }
      state.error = { hasError: false }
    }),
  }))
)

// ============================================================================
// Community Store
// ============================================================================

interface CommunityStore {
  feed: ResonantFeedItem[]
  userCreations: CommunityCreation[]
  kindredSpirits: KindredSpirit[]
  followingActivity: CommunityCreation[]
  
  // Filters and preferences
  feedFilters: {
    themes: string[]
    forms: string[]
    minFitnessScore: number
    resonanceThreshold: number
  }
  
  // Loading & Error States
  loading: LoadingState
  error: ErrorState
  
  // Actions
  setFeed: (feed: ResonantFeedItem[]) => void
  addToFeed: (items: ResonantFeedItem[]) => void
  setUserCreations: (creations: CommunityCreation[]) => void
  addCreation: (creation: CommunityCreation) => void
  setKindredSpirits: (spirits: KindredSpirit[]) => void
  setFollowingActivity: (activity: CommunityCreation[]) => void
  updateFeedFilters: (filters: Partial<CommunityStore['feedFilters']>) => void
  setLoading: (loading: LoadingState) => void
  setError: (error: ErrorState) => void
  clearError: () => void
  reset: () => void
}

export const useCommunityStore = create<CommunityStore>()(
  persist(
    immer((set, get) => ({
      feed: [],
      userCreations: [],
      kindredSpirits: [],
      followingActivity: [],
      feedFilters: {
        themes: [],
        forms: [],
        minFitnessScore: 0,
        resonanceThreshold: 0.5,
      },
      loading: { isLoading: false },
      error: { hasError: false },
      
      setFeed: (feed) => set((state) => {
        state.feed = feed
      }),
      
      addToFeed: (items) => set((state) => {
        state.feed.push(...items)
      }),
      
      setUserCreations: (creations) => set((state) => {
        state.userCreations = creations
      }),
      
      addCreation: (creation) => set((state) => {
        state.userCreations.unshift(creation)
      }),
      
      setKindredSpirits: (spirits) => set((state) => {
        state.kindredSpirits = spirits
      }),
      
      setFollowingActivity: (activity) => set((state) => {
        state.followingActivity = activity
      }),
      
      updateFeedFilters: (filters) => set((state) => {
        state.feedFilters = { ...state.feedFilters, ...filters }
      }),
      
      setLoading: (loading) => set((state) => {
        state.loading = loading
      }),
      
      setError: (error) => set((state) => {
        state.error = error
      }),
      
      clearError: () => set((state) => {
        state.error = { hasError: false }
      }),
      
      reset: () => set((state) => {
        state.feed = []
        state.userCreations = []
        state.kindredSpirits = []
        state.followingActivity = []
        state.feedFilters = {
          themes: [],
          forms: [],
          minFitnessScore: 0,
          resonanceThreshold: 0.5,
        }
        state.loading = { isLoading: false }
        state.error = { hasError: false }
      }),
    })),
    {
      name: 'community-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        feedFilters: state.feedFilters,
      }),
    }
  )
)

// ============================================================================
// Store Selectors
// ============================================================================

// Muse Store Selectors
export const selectUser = (state: any) => state.user
export const selectIsAuthenticated = (state: any) => state.isAuthenticated
export const selectCurrentSignature = (state: any) => state.currentSignature
export const selectTheme = (state: any) => state.theme
export const selectLoading = (state: any) => state.loading
export const selectError = (state: any) => state.error
export const selectNotifications = (state: any) => state.notifications
export const selectUnreadNotifications = (state: any) => 
  state.notifications.filter((n: any) => !n.read)

// Discovery Store Selectors
export const selectCurrentSession = (state: any) => state.currentSession
export const selectSessionHistory = (state: any) => state.sessionHistory
export const selectActiveCollaborations = (state: any) => state.activeCollaborations
export const selectLiveUpdates = (state: any) => state.liveUpdates

// Community Store Selectors
export const selectFeed = (state: any) => state.feed
export const selectUserCreations = (state: any) => state.userCreations
export const selectKindredSpirits = (state: any) => state.kindredSpirits
export const selectFeedFilters = (state: any) => state.feedFilters

// ============================================================================
// Store Actions
// ============================================================================

// Combined actions for complex operations
export const useMuseActions = () => {
  const museStore = useMuseStore()
  const discoveryStore = useDiscoveryStore()
  const communityStore = useCommunityStore()
  
  return {
    // Authentication
    login: (user: UserProfile, signature?: FrequencySignature) => {
      museStore.setUser(user)
      if (signature) {
        museStore.setSignature(signature)
      }
    },
    
    logout: () => {
      museStore.reset()
      discoveryStore.reset()
      communityStore.reset()
    },
    
    // Theme management
    updateTheme: (muse: MuseArchetype) => {
      museStore.updateThemeForMuse(muse)
    },
    
    // Error handling
    handleError: (error: any, context?: string) => {
      museStore.setError({
        hasError: true,
        error,
        message: error.message || 'An error occurred',
        code: error.code || 'UNKNOWN',
      })
      
      museStore.addNotification({
        type: 'error',
        message: `${context ? `${context}: ` : ''}${error.message || 'An error occurred'}`,
      })
    },
    
    // Success notifications
    showSuccess: (message: string) => {
      museStore.addNotification({
        type: 'success',
        message,
      })
    },
    
    // Info notifications
    showInfo: (message: string) => {
      museStore.addNotification({
        type: 'info',
        message,
      })
    },
  }
}

export default {
  useMuseStore,
  useDiscoveryStore,
  useCommunityStore,
  useMuseActions,
}
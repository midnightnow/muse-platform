/**
 * MUSE API Custom Hooks
 * 
 * React hooks for interacting with the MUSE Platform API,
 * providing state management, caching, and error handling
 * for all API operations.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { Socket } from 'socket.io-client'
import MuseAPIService from '@/services/api'
import {
  UserProfile,
  FrequencySignature,
  DiscoverySession,
  DiscoveryResult,
  CommunityCreation,
  ResonantFeedItem,
  KindredSpirit,
  Comment,
  CollaborativeSession,
  PersonalityAssessmentRequest,
  SignatureTuningRequest,
  DiscoverySessionRequest,
  SessionFeedbackRequest,
  CreationShareRequest,
  CommentCreateRequest,
  PaginatedResponse,
  PaginationParams,
  LoadingState,
  ErrorState,
  WebSocketMessage,
  DiscoveryUpdate,
  CollaborationUpdate,
} from '@/types'

// ============================================================================
// Base Hook for API State Management
// ============================================================================

interface UseAPIStateOptions<T> {
  initialData?: T
  autoFetch?: boolean
  dependencies?: any[]
}

function useAPIState<T>(
  apiCall: () => Promise<T>,
  options: UseAPIStateOptions<T> = {}
) {
  const [data, setData] = useState<T | null>(options.initialData || null)
  const [loading, setLoading] = useState<LoadingState>({
    isLoading: false,
    phase: undefined,
    progress: undefined,
    message: undefined,
  })
  const [error, setError] = useState<ErrorState>({
    hasError: false,
    error: undefined,
    message: undefined,
    code: undefined,
  })
  
  const isMountedRef = useRef(true)
  const controllerRef = useRef<AbortController | null>(null)
  
  const execute = useCallback(async (...args: any[]) => {
    if (controllerRef.current) {
      controllerRef.current.abort()
    }
    
    controllerRef.current = new AbortController()
    
    setLoading({
      isLoading: true,
      phase: 'fetching',
      progress: 0,
      message: 'Loading...',
    })
    setError({ hasError: false })
    
    try {
      const result = await apiCall(...args)
      
      if (isMountedRef.current) {
        setData(result)
        setLoading({
          isLoading: false,
          phase: 'completed',
          progress: 100,
          message: 'Success',
        })
      }
      
      return result
    } catch (err: any) {
      if (isMountedRef.current && err.name !== 'AbortError') {
        setError({
          hasError: true,
          error: err,
          message: err.message || 'An error occurred',
          code: err.code || 'UNKNOWN',
        })
        setLoading({
          isLoading: false,
          phase: 'error',
          progress: 0,
          message: 'Error occurred',
        })
      }
      throw err
    }
  }, [apiCall])
  
  useEffect(() => {
    if (options.autoFetch) {
      execute()
    }
  }, options.dependencies || [])
  
  useEffect(() => {
    return () => {
      isMountedRef.current = false
      if (controllerRef.current) {
        controllerRef.current.abort()
      }
    }
  }, [])
  
  return {
    data,
    loading,
    error,
    execute,
    setData,
    clearError: () => setError({ hasError: false }),
  }
}

// ============================================================================
// User Profile Hooks
// ============================================================================

export function useUserProfile(userId?: string) {
  return useAPIState(
    () => MuseAPIService.getProfile(userId!),
    {
      autoFetch: !!userId,
      dependencies: [userId],
    }
  )
}

export function useCreateProfile() {
  return useAPIState(
    (data: any) => MuseAPIService.createProfile(data),
    { autoFetch: false }
  )
}

export function useUpdateProfile(userId: string) {
  return useAPIState(
    (data: any) => MuseAPIService.updateProfile(userId, data),
    { autoFetch: false }
  )
}

// ============================================================================
// Frequency Signature Hooks
// ============================================================================

export function useFrequencySignature(signatureId?: string) {
  return useAPIState(
    () => MuseAPIService.getFrequencySignature(signatureId!),
    {
      autoFetch: !!signatureId,
      dependencies: [signatureId],
    }
  )
}

export function useCompleteAssessment() {
  return useAPIState(
    (data: PersonalityAssessmentRequest) => MuseAPIService.completeAssessment(data),
    { autoFetch: false }
  )
}

export function useTuneSignature(signatureId: string) {
  return useAPIState(
    (data: SignatureTuningRequest) => MuseAPIService.tuneFrequencySignature(signatureId, data),
    { autoFetch: false }
  )
}

export function useUserSignatures(userId?: string) {
  return useAPIState(
    () => MuseAPIService.getUserSignatures(userId!),
    {
      autoFetch: !!userId,
      dependencies: [userId],
    }
  )
}

// ============================================================================
// Discovery Session Hooks
// ============================================================================

export function useDiscoverySession(sessionId?: string) {
  const [session, setSession] = useState<DiscoverySession | null>(null)
  const [loading, setLoading] = useState<LoadingState>({ isLoading: false })
  const [error, setError] = useState<ErrorState>({ hasError: false })
  
  const startSession = useCallback(async (data: DiscoverySessionRequest) => {
    setLoading({ isLoading: true, phase: 'initialization', message: 'Starting discovery session...' })
    setError({ hasError: false })
    
    try {
      const result = await MuseAPIService.startDiscoverySession(data)
      setSession(result)
      setLoading({ isLoading: false, phase: 'completed' })
      return result
    } catch (err: any) {
      setError({ hasError: true, error: err, message: err.message })
      setLoading({ isLoading: false, phase: 'error' })
      throw err
    }
  }, [])
  
  const continueSession = useCallback(async (feedback?: SessionFeedbackRequest) => {
    if (!session) return
    
    setLoading({ isLoading: true, phase: 'optimization', message: 'Optimizing discovery...' })
    
    try {
      const result = await MuseAPIService.continueDiscoverySession(session.session_id, feedback)
      setSession(result)
      setLoading({ isLoading: false, phase: 'completed' })
      return result
    } catch (err: any) {
      setError({ hasError: true, error: err, message: err.message })
      setLoading({ isLoading: false, phase: 'error' })
      throw err
    }
  }, [session])
  
  const completeSession = useCallback(async (feedback?: SessionFeedbackRequest) => {
    if (!session) return
    
    setLoading({ isLoading: true, phase: 'completion', message: 'Finalizing discovery...' })
    
    try {
      const result = await MuseAPIService.completeDiscoverySession(session.session_id, feedback)
      setLoading({ isLoading: false, phase: 'completed' })
      return result
    } catch (err: any) {
      setError({ hasError: true, error: err, message: err.message })
      setLoading({ isLoading: false, phase: 'error' })
      throw err
    }
  }, [session])
  
  const refreshSession = useCallback(async () => {
    if (!session) return
    
    try {
      const result = await MuseAPIService.getSessionStatus(session.session_id)
      setSession(result)
      return result
    } catch (err: any) {
      setError({ hasError: true, error: err, message: err.message })
      throw err
    }
  }, [session])
  
  return {
    session,
    loading,
    error,
    startSession,
    continueSession,
    completeSession,
    refreshSession,
    clearError: () => setError({ hasError: false }),
  }
}

export function useUserSessions(userId?: string) {
  return useAPIState(
    () => MuseAPIService.getUserSessions(userId!),
    {
      autoFetch: !!userId,
      dependencies: [userId],
    }
  )
}

// ============================================================================
// Community Hooks
// ============================================================================

export function useResonantFeed(userId?: string, params?: PaginationParams) {
  const [feed, setFeed] = useState<ResonantFeedItem[]>([])
  const [pagination, setPagination] = useState<any>(null)
  const [loading, setLoading] = useState<LoadingState>({ isLoading: false })
  const [error, setError] = useState<ErrorState>({ hasError: false })
  
  const loadFeed = useCallback(async (loadMore = false) => {
    if (!userId) return
    
    setLoading({ isLoading: true, phase: 'fetching', message: 'Loading feed...' })
    setError({ hasError: false })
    
    try {
      const result = await MuseAPIService.getResonantFeed(userId, params)
      
      if (loadMore) {
        setFeed(prev => [...prev, ...result.data])
      } else {
        setFeed(result.data)
      }
      
      setPagination(result.pagination)
      setLoading({ isLoading: false, phase: 'completed' })
      return result
    } catch (err: any) {
      setError({ hasError: true, error: err, message: err.message })
      setLoading({ isLoading: false, phase: 'error' })
      throw err
    }
  }, [userId, params])
  
  const loadMore = useCallback(() => {
    if (!pagination?.has_next) return
    
    const nextParams = {
      ...params,
      page: pagination.page + 1,
    }
    
    return loadFeed(true)
  }, [pagination, params, loadFeed])
  
  useEffect(() => {
    if (userId) {
      loadFeed()
    }
  }, [userId, loadFeed])
  
  return {
    feed,
    pagination,
    loading,
    error,
    loadFeed,
    loadMore,
    clearError: () => setError({ hasError: false }),
  }
}

export function useUserCreations(userId?: string, params?: PaginationParams) {
  return useAPIState(
    () => MuseAPIService.getUserCreations(userId!, params),
    {
      autoFetch: !!userId,
      dependencies: [userId, params],
    }
  )
}

export function useShareCreation() {
  return useAPIState(
    (data: CreationShareRequest) => MuseAPIService.shareCreation(data),
    { autoFetch: false }
  )
}

export function useKindredSpirits(userId?: string) {
  return useAPIState(
    () => MuseAPIService.getKindredSpirits(userId!),
    {
      autoFetch: !!userId,
      dependencies: [userId],
    }
  )
}

// ============================================================================
// WebSocket Hooks
// ============================================================================

export function useWebSocket(userId?: string) {
  const [socket, setSocket] = useState<Socket | null>(null)
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  
  const connect = useCallback(() => {
    if (!userId || socket) return
    
    const newSocket = MuseAPIService.connectWebSocket(userId)
    
    newSocket.on('connect', () => {
      setConnected(true)
    })
    
    newSocket.on('disconnect', () => {
      setConnected(false)
    })
    
    newSocket.on('message', (message: WebSocketMessage) => {
      setMessages(prev => [...prev, message])
    })
    
    setSocket(newSocket)
  }, [userId, socket])
  
  const disconnect = useCallback(() => {
    if (socket) {
      socket.disconnect()
      setSocket(null)
      setConnected(false)
    }
  }, [socket])
  
  const sendMessage = useCallback((message: any) => {
    if (socket && connected) {
      socket.emit('message', message)
    }
  }, [socket, connected])
  
  useEffect(() => {
    if (userId) {
      connect()
    }
    
    return () => {
      disconnect()
    }
  }, [userId])
  
  return {
    socket,
    connected,
    messages,
    connect,
    disconnect,
    sendMessage,
    clearMessages: () => setMessages([]),
  }
}

// ============================================================================
// Real-time Discovery Hook
// ============================================================================

export function useRealtimeDiscovery(sessionId?: string) {
  const [updates, setUpdates] = useState<DiscoveryUpdate[]>([])
  const [currentUpdate, setCurrentUpdate] = useState<DiscoveryUpdate | null>(null)
  const { socket, connected } = useWebSocket()
  
  useEffect(() => {
    if (!socket || !connected || !sessionId) return
    
    const handleDiscoveryUpdate = (update: DiscoveryUpdate) => {
      if (update.session_id === sessionId) {
        setUpdates(prev => [...prev, update])
        setCurrentUpdate(update)
      }
    }
    
    socket.on('discovery_update', handleDiscoveryUpdate)
    
    return () => {
      socket.off('discovery_update', handleDiscoveryUpdate)
    }
  }, [socket, connected, sessionId])
  
  return {
    updates,
    currentUpdate,
    clearUpdates: () => setUpdates([]),
  }
}

// ============================================================================
// Collaborative Session Hook
// ============================================================================

export function useCollaborativeSession(sessionId?: string) {
  const [session, setSession] = useState<CollaborativeSession | null>(null)
  const [updates, setUpdates] = useState<CollaborationUpdate[]>([])
  const [loading, setLoading] = useState<LoadingState>({ isLoading: false })
  const [error, setError] = useState<ErrorState>({ hasError: false })
  const { socket, connected } = useWebSocket()
  
  const loadSession = useCallback(async () => {
    if (!sessionId) return
    
    setLoading({ isLoading: true, phase: 'fetching', message: 'Loading collaborative session...' })
    setError({ hasError: false })
    
    try {
      const result = await MuseAPIService.getCollaborativeSession(sessionId)
      setSession(result)
      setLoading({ isLoading: false, phase: 'completed' })
      return result
    } catch (err: any) {
      setError({ hasError: true, error: err, message: err.message })
      setLoading({ isLoading: false, phase: 'error' })
      throw err
    }
  }, [sessionId])
  
  const joinSession = useCallback(async () => {
    if (!sessionId) return
    
    try {
      await MuseAPIService.joinCollaborativeSession(sessionId)
      await loadSession()
    } catch (err: any) {
      setError({ hasError: true, error: err, message: err.message })
      throw err
    }
  }, [sessionId, loadSession])
  
  const leaveSession = useCallback(async () => {
    if (!sessionId) return
    
    try {
      await MuseAPIService.leaveCollaborativeSession(sessionId)
      await loadSession()
    } catch (err: any) {
      setError({ hasError: true, error: err, message: err.message })
      throw err
    }
  }, [sessionId, loadSession])
  
  useEffect(() => {
    if (sessionId) {
      loadSession()
    }
  }, [sessionId, loadSession])
  
  useEffect(() => {
    if (!socket || !connected || !sessionId) return
    
    const handleCollaborationUpdate = (update: CollaborationUpdate) => {
      if (update.session_id === sessionId) {
        setUpdates(prev => [...prev, update])
        
        // Refresh session data on certain updates
        if (update.action === 'join' || update.action === 'leave') {
          loadSession()
        }
      }
    }
    
    socket.on('collaboration_update', handleCollaborationUpdate)
    
    return () => {
      socket.off('collaboration_update', handleCollaborationUpdate)
    }
  }, [socket, connected, sessionId, loadSession])
  
  return {
    session,
    updates,
    loading,
    error,
    joinSession,
    leaveSession,
    refreshSession: loadSession,
    clearError: () => setError({ hasError: false }),
  }
}

// ============================================================================
// Utility Hooks
// ============================================================================

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)
  
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value)
    }, delay)
    
    return () => {
      clearTimeout(handler)
    }
  }, [value, delay])
  
  return debouncedValue
}

export function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : initialValue
    } catch (error) {
      return initialValue
    }
  })
  
  const setValue = (value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value
      setStoredValue(valueToStore)
      window.localStorage.setItem(key, JSON.stringify(valueToStore))
    } catch (error) {
      console.error('Error saving to localStorage:', error)
    }
  }
  
  return [storedValue, setValue] as const
}

export default {
  useUserProfile,
  useCreateProfile,
  useUpdateProfile,
  useFrequencySignature,
  useCompleteAssessment,
  useTuneSignature,
  useUserSignatures,
  useDiscoverySession,
  useUserSessions,
  useResonantFeed,
  useUserCreations,
  useShareCreation,
  useKindredSpirits,
  useWebSocket,
  useRealtimeDiscovery,
  useCollaborativeSession,
  useDebounce,
  useLocalStorage,
}
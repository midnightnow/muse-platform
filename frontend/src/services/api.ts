/**
 * MUSE Platform API Service
 * 
 * Central API service for communicating with the MUSE backend,
 * including authentication, error handling, and request/response
 * transformation.
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios'
import { io, Socket } from 'socket.io-client'
import {
  APIResponse,
  APIError,
  PersonalityAssessmentRequest,
  FrequencySignature,
  SignatureTuningRequest,
  DiscoverySessionRequest,
  DiscoverySession,
  DiscoveryResult,
  SessionFeedbackRequest,
  UserProfile,
  ProfileCreateRequest,
  ProfileUpdateRequest,
  CommunityCreation,
  CreationShareRequest,
  Comment,
  CommentCreateRequest,
  ResonanceScore,
  KindredSpirit,
  ResonantFeedItem,
  CollaborativeSession,
  PaginatedResponse,
  PaginationParams,
  WebSocketMessage
} from '@/types'

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

// Create axios instance with default configuration
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
})

// Request interceptor for authentication
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('muse_auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => {
    const apiError: APIError = {
      error: error.response?.data?.error || 'Unknown error',
      message: error.response?.data?.message || error.message || 'Request failed',
      code: error.response?.status?.toString() || 'UNKNOWN',
      details: error.response?.data?.details || {},
      timestamp: new Date().toISOString(),
    }
    
    // Handle specific error cases
    if (error.response?.status === 401) {
      localStorage.removeItem('muse_auth_token')
      window.location.href = '/login'
    }
    
    return Promise.reject(apiError)
  }
)

// WebSocket connection management
let socketConnection: Socket | null = null

const getWebSocketConnection = (): Socket => {
  if (!socketConnection) {
    socketConnection = io(WS_URL, {
      transports: ['websocket'],
      autoConnect: false,
    })
  }
  return socketConnection
}

// ============================================================================
// Core API Functions
// ============================================================================

export class MuseAPIService {
  
  // Health and System
  static async getSystemHealth(): Promise<any> {
    const response = await apiClient.get('/health')
    return response.data
  }
  
  static async getSystemMetrics(): Promise<any> {
    const response = await apiClient.get('/metrics')
    return response.data
  }
  
  // ============================================================================
  // Authentication & User Management
  // ============================================================================
  
  static async createProfile(data: ProfileCreateRequest): Promise<UserProfile> {
    const response = await apiClient.post('/api/v1/community/profiles', data)
    return response.data
  }
  
  static async getProfile(userId: string): Promise<UserProfile> {
    const response = await apiClient.get(`/api/v1/community/profiles/${userId}`)
    return response.data
  }
  
  static async updateProfile(userId: string, data: ProfileUpdateRequest): Promise<UserProfile> {
    const response = await apiClient.put(`/api/v1/community/profiles/${userId}`, data)
    return response.data
  }
  
  static async deleteProfile(userId: string): Promise<void> {
    await apiClient.delete(`/api/v1/community/profiles/${userId}`)
  }
  
  // ============================================================================
  // Frequency Signatures & Assessment
  // ============================================================================
  
  static async completeAssessment(data: PersonalityAssessmentRequest): Promise<FrequencySignature> {
    const response = await apiClient.post('/api/v1/assessment/complete', data)
    return response.data
  }
  
  static async getFrequencySignature(signatureId: string): Promise<FrequencySignature> {
    const response = await apiClient.get(`/api/v1/signatures/${signatureId}`)
    return response.data
  }
  
  static async tuneFrequencySignature(
    signatureId: string,
    data: SignatureTuningRequest
  ): Promise<FrequencySignature> {
    const response = await apiClient.post(`/api/v1/signatures/${signatureId}/tune`, data)
    return response.data
  }
  
  static async getUserSignatures(userId: string): Promise<FrequencySignature[]> {
    const response = await apiClient.get(`/api/v1/community/profiles/${userId}/signatures`)
    return response.data
  }
  
  // ============================================================================
  // Discovery Sessions
  // ============================================================================
  
  static async startDiscoverySession(data: DiscoverySessionRequest): Promise<DiscoverySession> {
    const response = await apiClient.post('/api/v1/sessions/start', data)
    return response.data
  }
  
  static async getSessionStatus(sessionId: string): Promise<DiscoverySession> {
    const response = await apiClient.get(`/api/v1/sessions/${sessionId}/status`)
    return response.data
  }
  
  static async continueDiscoverySession(
    sessionId: string,
    feedback?: SessionFeedbackRequest
  ): Promise<DiscoverySession> {
    const response = await apiClient.post(`/api/v1/sessions/${sessionId}/continue`, feedback)
    return response.data
  }
  
  static async completeDiscoverySession(
    sessionId: string,
    feedback?: SessionFeedbackRequest
  ): Promise<DiscoveryResult> {
    const response = await apiClient.post(`/api/v1/sessions/${sessionId}/complete`, feedback)
    return response.data
  }
  
  static async getUserSessions(userId: string): Promise<DiscoverySession[]> {
    const response = await apiClient.get(`/api/v1/community/profiles/${userId}/sessions`)
    return response.data
  }
  
  // ============================================================================
  // Integration API (Advanced Features)
  // ============================================================================
  
  static async getLiveDiscoveryStream(sessionId: string): Promise<ReadableStream> {
    const response = await apiClient.get(`/api/v1/integration/live-discovery/${sessionId}`, {
      responseType: 'stream'
    })
    return response.data
  }
  
  static async getConstraintOptimization(sessionId: string): Promise<any> {
    const response = await apiClient.get(`/api/v1/integration/constraint-optimization/${sessionId}`)
    return response.data
  }
  
  static async updateConstraints(sessionId: string, constraints: any): Promise<any> {
    const response = await apiClient.post(`/api/v1/integration/constraint-optimization/${sessionId}`, {
      constraints
    })
    return response.data
  }
  
  // ============================================================================
  // Community Features
  // ============================================================================
  
  static async getFeed(params?: PaginationParams): Promise<PaginatedResponse<ResonantFeedItem>> {
    const response = await apiClient.get('/api/v1/community/feed', { params })
    return response.data
  }
  
  static async getResonantFeed(userId: string, params?: PaginationParams): Promise<PaginatedResponse<ResonantFeedItem>> {
    const response = await apiClient.get(`/api/v1/community/users/${userId}/resonant-feed`, { params })
    return response.data
  }
  
  static async shareCreation(data: CreationShareRequest): Promise<CommunityCreation> {
    const response = await apiClient.post('/api/v1/community/creations', data)
    return response.data
  }
  
  static async getCreation(creationId: string): Promise<CommunityCreation> {
    const response = await apiClient.get(`/api/v1/community/creations/${creationId}`)
    return response.data
  }
  
  static async getUserCreations(userId: string, params?: PaginationParams): Promise<PaginatedResponse<CommunityCreation>> {
    const response = await apiClient.get(`/api/v1/community/users/${userId}/creations`, { params })
    return response.data
  }
  
  static async likeCreation(creationId: string): Promise<void> {
    await apiClient.post(`/api/v1/community/creations/${creationId}/like`)
  }
  
  static async unlikeCreation(creationId: string): Promise<void> {
    await apiClient.delete(`/api/v1/community/creations/${creationId}/like`)
  }
  
  static async addComment(creationId: string, data: CommentCreateRequest): Promise<Comment> {
    const response = await apiClient.post(`/api/v1/community/creations/${creationId}/comments`, data)
    return response.data
  }
  
  static async getComments(creationId: string, params?: PaginationParams): Promise<PaginatedResponse<Comment>> {
    const response = await apiClient.get(`/api/v1/community/creations/${creationId}/comments`, { params })
    return response.data
  }
  
  static async deleteComment(commentId: string): Promise<void> {
    await apiClient.delete(`/api/v1/community/comments/${commentId}`)
  }
  
  // ============================================================================
  // Social Features
  // ============================================================================
  
  static async followUser(userId: string): Promise<void> {
    await apiClient.post(`/api/v1/community/users/${userId}/follow`)
  }
  
  static async unfollowUser(userId: string): Promise<void> {
    await apiClient.delete(`/api/v1/community/users/${userId}/follow`)
  }
  
  static async getFollowers(userId: string, params?: PaginationParams): Promise<PaginatedResponse<UserProfile>> {
    const response = await apiClient.get(`/api/v1/community/users/${userId}/followers`, { params })
    return response.data
  }
  
  static async getFollowing(userId: string, params?: PaginationParams): Promise<PaginatedResponse<UserProfile>> {
    const response = await apiClient.get(`/api/v1/community/users/${userId}/following`, { params })
    return response.data
  }
  
  static async getKindredSpirits(userId: string, params?: PaginationParams): Promise<PaginatedResponse<KindredSpirit>> {
    const response = await apiClient.get(`/api/v1/community/users/${userId}/kindred-spirits`, { params })
    return response.data
  }
  
  static async getResonanceScore(userId: string, targetUserId: string): Promise<ResonanceScore> {
    const response = await apiClient.get(`/api/v1/community/users/${userId}/resonance/${targetUserId}`)
    return response.data
  }
  
  // ============================================================================
  // Collaborative Sessions
  // ============================================================================
  
  static async createCollaborativeSession(data: any): Promise<CollaborativeSession> {
    const response = await apiClient.post('/api/v1/community/collaborative-sessions', data)
    return response.data
  }
  
  static async getCollaborativeSession(sessionId: string): Promise<CollaborativeSession> {
    const response = await apiClient.get(`/api/v1/community/collaborative-sessions/${sessionId}`)
    return response.data
  }
  
  static async joinCollaborativeSession(sessionId: string): Promise<void> {
    await apiClient.post(`/api/v1/community/collaborative-sessions/${sessionId}/join`)
  }
  
  static async leaveCollaborativeSession(sessionId: string): Promise<void> {
    await apiClient.post(`/api/v1/community/collaborative-sessions/${sessionId}/leave`)
  }
  
  static async getActiveCollaborativeSessions(params?: PaginationParams): Promise<PaginatedResponse<CollaborativeSession>> {
    const response = await apiClient.get('/api/v1/community/collaborative-sessions', { params })
    return response.data
  }
  
  // ============================================================================
  // Search and Discovery
  // ============================================================================
  
  static async searchUsers(query: string, params?: PaginationParams): Promise<PaginatedResponse<UserProfile>> {
    const response = await apiClient.get('/api/v1/community/search/users', {
      params: { query, ...params }
    })
    return response.data
  }
  
  static async searchCreations(query: string, params?: PaginationParams): Promise<PaginatedResponse<CommunityCreation>> {
    const response = await apiClient.get('/api/v1/community/search/creations', {
      params: { query, ...params }
    })
    return response.data
  }
  
  static async getPopularCreations(params?: PaginationParams): Promise<PaginatedResponse<CommunityCreation>> {
    const response = await apiClient.get('/api/v1/community/popular', { params })
    return response.data
  }
  
  static async getTrendingThemes(): Promise<string[]> {
    const response = await apiClient.get('/api/v1/community/trending-themes')
    return response.data
  }
  
  // ============================================================================
  // WebSocket Management
  // ============================================================================
  
  static connectWebSocket(userId: string): Socket {
    const socket = getWebSocketConnection()
    
    socket.auth = {
      userId,
      token: localStorage.getItem('muse_auth_token'),
    }
    
    socket.connect()
    
    return socket
  }
  
  static disconnectWebSocket(): void {
    if (socketConnection) {
      socketConnection.disconnect()
      socketConnection = null
    }
  }
  
  static onWebSocketMessage(callback: (message: WebSocketMessage) => void): void {
    const socket = getWebSocketConnection()
    socket.on('message', callback)
  }
  
  static removeWebSocketListener(event: string, callback: Function): void {
    const socket = getWebSocketConnection()
    socket.off(event, callback)
  }
  
  // ============================================================================
  // Utility Functions
  // ============================================================================
  
  static async uploadFile(file: File, type: 'avatar' | 'creation-asset'): Promise<string> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('type', type)
    
    const response = await apiClient.post('/api/v1/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    
    return response.data.url
  }
  
  static async getUploadSignedUrl(filename: string, contentType: string): Promise<string> {
    const response = await apiClient.post('/api/v1/upload/signed-url', {
      filename,
      contentType,
    })
    return response.data.signedUrl
  }
  
  static setAuthToken(token: string): void {
    localStorage.setItem('muse_auth_token', token)
  }
  
  static getAuthToken(): string | null {
    return localStorage.getItem('muse_auth_token')
  }
  
  static clearAuthToken(): void {
    localStorage.removeItem('muse_auth_token')
  }
  
  static isAuthenticated(): boolean {
    return !!this.getAuthToken()
  }
}

// Export the API service as the default export
export default MuseAPIService
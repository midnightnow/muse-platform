/**
 * MUSE Platform TypeScript Type Definitions
 * 
 * Comprehensive type definitions for the MUSE Computational Platonism
 * creative discovery platform, including API models, UI components,
 * and mathematical structures.
 */

// ============================================================================
// Core MUSE Types
// ============================================================================

export type MuseArchetype = 
  | 'CALLIOPE'    // Epic poetry
  | 'CLIO'        // History
  | 'ERATO'       // Love poetry
  | 'EUTERPE'     // Music
  | 'MELPOMENE'   // Tragedy
  | 'POLYHYMNIA'  // Sacred hymns
  | 'TERPSICHORE' // Dance
  | 'THALIA'      // Comedy
  | 'URANIA'      // Astronomy
  | 'SOPHIA'      // Wisdom
  | 'TECHNE'      // Craft
  | 'PSYCHE'      // Soul

export type DiscoveryStyle = 'mathematical' | 'intuitive' | 'experimental' | 'balanced'
export type DiscoveryMode = 'individual' | 'collaborative' | 'guided' | 'experimental'
export type DiscoveryPhase = 'initialization' | 'optimization' | 'refinement' | 'completion'
export type Visibility = 'public' | 'followers' | 'private'

// ============================================================================
// Mathematical Structures
// ============================================================================

export interface SacredRatios {
  phi: number        // Golden ratio (1.618...)
  pi: number         // Pi (3.14159...)
  euler: number      // Euler's number (2.718...)
  root2: number      // Square root of 2
  root3: number      // Square root of 3
  root5: number      // Square root of 5
  [key: string]: number
}

export interface SpiralCoordinates {
  radius: number
  angle: number
  height: number
  spiral_type: 'fibonacci' | 'golden' | 'archimedes' | 'logarithmic'
  turn_count: number
  [key: string]: number | string
}

export interface HarmonicBlend {
  [key in MuseArchetype]: number
}

export interface FrequencySignature {
  id: string
  user_id: string
  harmonic_blend: HarmonicBlend
  sacred_ratios: SacredRatios
  spiral_coordinates: SpiralCoordinates
  primary_muse: MuseArchetype
  secondary_muse: MuseArchetype
  entropy_seed: string
  characteristics: {
    specialization_index: number
    diversity_index: number
    coherence_score: number
    uniqueness_score: number
  }
  performance_metrics: {
    discovery_success_rate: number
    average_fitness_score: number
    user_satisfaction_score: number
  }
  created_at: string
  updated_at?: string
}

// ============================================================================
// User and Profile Types
// ============================================================================

export interface UserProfile {
  id: string
  username: string
  email: string
  display_name?: string
  bio?: string
  avatar_url?: string
  current_signature_id?: string
  primary_muse?: MuseArchetype
  secondary_muse?: MuseArchetype
  harmonic_blend?: HarmonicBlend
  sacred_ratios?: SacredRatios
  spiral_coordinates?: SpiralCoordinates
  preferred_forms: string[]
  favorite_themes: string[]
  discovery_style: DiscoveryStyle
  profile_visibility: Visibility
  signature_sharing: boolean
  discovery_sharing: boolean
  followers_count: number
  following_count: number
  creations_count: number
  total_discoveries: number
  average_fitness_score: number
  created_at: string
  updated_at?: string
}

export interface UserStats {
  total_discoveries: number
  total_creations: number
  average_fitness_score: number
  best_fitness_score: number
  discovery_success_rate: number
  favorite_themes: string[]
  most_used_forms: string[]
  signature_evolution: number
  community_engagement: number
  collaboration_count: number
}

// ============================================================================
// Discovery and Creative Session Types
// ============================================================================

export interface DiscoverySession {
  session_id: string
  user_id: string
  theme: string
  form_type: string
  mode: DiscoveryMode
  phase: DiscoveryPhase
  progress: number
  current_iteration: number
  max_iterations: number
  fitness_scores: {
    mathematical_fitness: number
    semantic_coherence: number
    archetypal_alignment: number
    sacred_geometry_score: number
    overall_fitness: number
  }
  current_discovery: {
    content: string
    metadata: Record<string, any>
    constraints: Record<string, any>
  }
  discovery_path: Array<{
    iteration: number
    fitness_score: number
    content_preview: string
    optimization_notes: string
    timestamp: string
  }>
  constraints?: Record<string, any>
  created_at: string
  updated_at: string
}

export interface DiscoveryResult {
  session_id: string
  user_id: string
  discovered_content: string
  form_type: string
  theme: string
  mathematical_fitness: number
  semantic_coherence: number
  archetypal_alignment: number
  sacred_geometry_score: number
  discovery_path: Array<{
    iteration: number
    fitness_score: number
    content_preview: string
    optimization_notes: string
    timestamp: string
  }>
  optimization_metrics: {
    total_iterations: number
    session_duration: number
    convergence_rate: number
    final_fitness: number
    improvement_rate: number
  }
  user_satisfaction?: number
  saved_to_community?: boolean
}

// ============================================================================
// Community and Social Types
// ============================================================================

export interface CommunityCreation {
  id: string
  creator_id: string
  creator_username: string
  creator_display_name?: string
  creator_avatar_url?: string
  creator_primary_muse?: MuseArchetype
  title: string
  content: string
  content_preview: string
  form_type: string
  primary_theme: string
  secondary_themes: string[]
  mathematical_fitness: number
  semantic_coherence: number
  archetypal_alignment: number
  sacred_geometry_score: number
  discovery_coordinates: Array<{
    iteration: number
    fitness_score: number
    content_preview: string
    optimization_notes: string
    timestamp: string
  }>
  discovery_time_seconds: number
  iteration_count: number
  constraint_satisfaction: number
  visibility: Visibility
  likes_count: number
  comments_count: number
  shares_count: number
  resonance_score: number
  is_featured: boolean
  featured_reason?: string
  tags: string[]
  created_at: string
  updated_at?: string
}

export interface Comment {
  id: string
  creation_id: string
  user_id: string
  username: string
  display_name?: string
  avatar_url?: string
  content: string
  parent_comment_id?: string
  replies_count: number
  likes_count: number
  is_liked: boolean
  created_at: string
  updated_at?: string
}

export interface Like {
  id: string
  user_id: string
  creation_id?: string
  comment_id?: string
  created_at: string
}

export interface Follow {
  id: string
  follower_id: string
  following_id: string
  created_at: string
}

export interface CollaborativeSession {
  id: string
  creator_id: string
  title: string
  description: string
  theme: string
  form_type: string
  mode: DiscoveryMode
  max_participants: number
  current_participants: number
  status: 'open' | 'active' | 'completed' | 'cancelled'
  visibility: Visibility
  participants: SessionParticipant[]
  current_discovery?: {
    content: string
    fitness_scores: Record<string, number>
    contributor_weights: Record<string, number>
  }
  created_at: string
  updated_at?: string
}

export interface SessionParticipant {
  id: string
  session_id: string
  user_id: string
  username: string
  display_name?: string
  avatar_url?: string
  primary_muse?: MuseArchetype
  role: 'creator' | 'participant' | 'observer'
  contribution_weight: number
  joined_at: string
  last_active: string
}

// ============================================================================
// API Request and Response Types
// ============================================================================

export interface PersonalityAssessmentRequest {
  user_id: string
  creative_preferences: Record<string, any>
  personality_traits: Record<string, any>
  mathematical_affinity: Record<string, any>
  preferred_forms: string[]
  favorite_themes: string[]
  discovery_style: DiscoveryStyle
}

export interface SignatureTuningRequest {
  target_muses: MuseArchetype[]
  blend_ratios: number[]
}

export interface DiscoverySessionRequest {
  user_id: string
  theme: string
  form_type: string
  mode: DiscoveryMode
  constraints?: Record<string, any>
}

export interface SessionFeedbackRequest {
  user_feedback: Record<string, any>
  satisfaction_score?: number
}

export interface ProfileCreateRequest {
  username: string
  email: string
  display_name?: string
  bio?: string
  avatar_url?: string
  preferred_forms: string[]
  favorite_themes: string[]
  discovery_style: DiscoveryStyle
}

export interface ProfileUpdateRequest {
  display_name?: string
  bio?: string
  avatar_url?: string
  preferred_forms?: string[]
  favorite_themes?: string[]
  discovery_style?: DiscoveryStyle
  profile_visibility?: Visibility
  signature_sharing?: boolean
  discovery_sharing?: boolean
}

export interface CreationShareRequest {
  title: string
  content: string
  form_type: string
  primary_theme: string
  secondary_themes: string[]
  visibility: Visibility
}

export interface CommentCreateRequest {
  content: string
  parent_comment_id?: string
}

// ============================================================================
// Resonance and Matching Types
// ============================================================================

export interface ResonanceScore {
  user_id: string
  target_user_id: string
  archetypal_similarity: number
  mathematical_compatibility: number
  creative_synergy: number
  overall_resonance: number
  shared_interests: string[]
  complementary_strengths: string[]
  potential_collaboration_themes: string[]
}

export interface KindredSpirit {
  user_id: string
  username: string
  display_name?: string
  avatar_url?: string
  primary_muse: MuseArchetype
  secondary_muse: MuseArchetype
  resonance_score: number
  compatibility_reasons: string[]
  shared_themes: string[]
  mutual_follows: boolean
  recent_creations: CommunityCreation[]
}

export interface ResonantFeedItem {
  creation: CommunityCreation
  resonance_score: number
  resonance_reasons: string[]
  creator_compatibility: number
  theme_alignment: number
  mathematical_harmony: number
}

// ============================================================================
// Visualization and UI Types
// ============================================================================

export interface VisualizationData {
  spiral_coordinates: SpiralCoordinates
  harmonic_blend: HarmonicBlend
  sacred_ratios: SacredRatios
  fitness_scores: Record<string, number>
  discovery_path: Array<{
    iteration: number
    fitness_score: number
    coordinates: { x: number; y: number; z: number }
    color: string
    timestamp: string
  }>
}

export interface ThemeConfig {
  primary_muse: MuseArchetype
  color_palette: {
    primary: string
    secondary: string
    accent: string
    background: string
    text: string
    border: string
  }
  font_family: string
  sacred_geometry_pattern: string
  animation_style: 'subtle' | 'moderate' | 'dynamic'
}

export interface LoadingState {
  isLoading: boolean
  phase?: string
  progress?: number
  message?: string
}

export interface ErrorState {
  hasError: boolean
  error?: Error
  message?: string
  code?: string
}

// ============================================================================
// WebSocket and Real-time Types
// ============================================================================

export interface WebSocketMessage {
  type: 'discovery_update' | 'session_update' | 'collaboration_update' | 'notification'
  data: any
  timestamp: string
  user_id?: string
  session_id?: string
}

export interface DiscoveryUpdate {
  session_id: string
  phase: DiscoveryPhase
  progress: number
  current_iteration: number
  fitness_scores: Record<string, number>
  current_discovery: {
    content: string
    metadata: Record<string, any>
  }
}

export interface CollaborationUpdate {
  session_id: string
  participant_id: string
  action: 'join' | 'leave' | 'contribute' | 'comment'
  data: any
  timestamp: string
}

// ============================================================================
// Form and Validation Types
// ============================================================================

export interface FormField {
  name: string
  type: 'text' | 'email' | 'select' | 'multiselect' | 'slider' | 'textarea' | 'checkbox'
  label: string
  placeholder?: string
  required?: boolean
  validation?: {
    min?: number
    max?: number
    pattern?: string
    message?: string
  }
  options?: Array<{ value: string; label: string }>
  defaultValue?: any
}

export interface AssessmentQuestion {
  id: string
  category: 'personality' | 'creative' | 'mathematical' | 'preferences'
  question: string
  type: 'scale' | 'choice' | 'multiple' | 'ranking'
  options?: string[]
  scale_range?: [number, number]
  weight: number
  archetypal_mapping?: Partial<HarmonicBlend>
}

// ============================================================================
// Utility Types
// ============================================================================

export type ID = string
export type Timestamp = string
export type UUID = string

export interface PaginationParams {
  page?: number
  limit?: number
  sort_by?: string
  sort_order?: 'asc' | 'desc'
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: {
    page: number
    limit: number
    total: number
    pages: number
    has_next: boolean
    has_prev: boolean
  }
}

export interface APIResponse<T> {
  success: boolean
  data: T
  message?: string
  error?: string
  timestamp: string
}

export interface APIError {
  error: string
  message: string
  code?: string
  details?: Record<string, any>
  timestamp: string
}

// ============================================================================
// Component Props Types
// ============================================================================

export interface BaseComponentProps {
  className?: string
  children?: React.ReactNode
  id?: string
  'data-testid'?: string
}

export interface MuseComponentProps extends BaseComponentProps {
  theme?: ThemeConfig
  loading?: LoadingState
  error?: ErrorState
}

export interface VisualizationProps extends BaseComponentProps {
  data: VisualizationData
  width?: number
  height?: number
  interactive?: boolean
  animation?: boolean
  onInteraction?: (data: any) => void
}

// ============================================================================
// Store and State Types
// ============================================================================

export interface AppState {
  user: UserProfile | null
  signature: FrequencySignature | null
  theme: ThemeConfig
  loading: LoadingState
  error: ErrorState
  socket: WebSocket | null
  notifications: Array<{
    id: string
    type: 'info' | 'success' | 'warning' | 'error'
    message: string
    timestamp: string
  }>
}

export interface DiscoveryState {
  current_session: DiscoverySession | null
  session_history: DiscoverySession[]
  active_collaborations: CollaborativeSession[]
  loading: LoadingState
  error: ErrorState
}

export interface CommunityState {
  feed: ResonantFeedItem[]
  kindred_spirits: KindredSpirit[]
  user_creations: CommunityCreation[]
  following_activity: CommunityCreation[]
  loading: LoadingState
  error: ErrorState
}

export default {}
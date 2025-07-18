/**
 * MUSE Mathematical Utilities
 * 
 * Frontend mathematical utilities for sacred geometry calculations,
 * frequency signature processing, and archetypal mathematics.
 */

import { SacredRatios, SpiralCoordinates, HarmonicBlend, MuseArchetype } from '@/types'

// ============================================================================
// Sacred Geometry Constants
// ============================================================================

export const SACRED_CONSTANTS = {
  PHI: 1.618033988749895, // Golden ratio
  PI: Math.PI,
  E: Math.E,
  SQRT2: Math.SQRT2,
  SQRT3: Math.sqrt(3),
  SQRT5: Math.sqrt(5),
  SILVER_RATIO: 1 + Math.SQRT2,
  BRONZE_RATIO: (3 + Math.sqrt(13)) / 2,
  PLASTIC_NUMBER: 1.324717957244746,
} as const

// ============================================================================
// Fibonacci Sequence
// ============================================================================

export function fibonacci(n: number): number {
  if (n <= 1) return n
  let a = 0, b = 1
  for (let i = 2; i <= n; i++) {
    [a, b] = [b, a + b]
  }
  return b
}

export function fibonacciSequence(length: number): number[] {
  return Array.from({ length }, (_, i) => fibonacci(i))
}

// ============================================================================
// Sacred Geometry Calculations
// ============================================================================

export function calculateGoldenRatio(a: number, b: number): number {
  return (a + b) / a
}

export function isGoldenRatio(ratio: number, tolerance = 0.01): boolean {
  return Math.abs(ratio - SACRED_CONSTANTS.PHI) < tolerance
}

export function goldenRectangleProportions(width: number): { height: number; ratio: number } {
  const height = width / SACRED_CONSTANTS.PHI
  return { height, ratio: width / height }
}

export function goldenSpiralPoint(angle: number, growth: number = SACRED_CONSTANTS.PHI): {
  x: number
  y: number
  radius: number
} {
  const radius = Math.pow(growth, angle / (Math.PI / 2))
  const x = radius * Math.cos(angle)
  const y = radius * Math.sin(angle)
  return { x, y, radius }
}

// ============================================================================
// Frequency Signature Mathematics
// ============================================================================

export function calculateHarmonicMean(values: number[]): number {
  if (values.length === 0) return 0
  const reciprocalSum = values.reduce((sum, value) => sum + (1 / (value || 1)), 0)
  return values.length / reciprocalSum
}

export function calculateSpecializationIndex(harmonicBlend: HarmonicBlend): number {
  const values = Object.values(harmonicBlend)
  if (values.length === 0) return 0
  
  const maxValue = Math.max(...values)
  const meanValue = values.reduce((sum, value) => sum + value, 0) / values.length
  
  return maxValue / (meanValue || 1)
}

export function calculateDiversityIndex(harmonicBlend: HarmonicBlend): number {
  const values = Object.values(harmonicBlend)
  if (values.length === 0) return 0
  
  // Shannon diversity index
  const total = values.reduce((sum, value) => sum + value, 0)
  if (total === 0) return 0
  
  const diversity = -values.reduce((sum, value) => {
    const proportion = value / total
    return sum + (proportion > 0 ? proportion * Math.log(proportion) : 0)
  }, 0)
  
  const maxDiversity = Math.log(values.length)
  return maxDiversity > 0 ? diversity / maxDiversity : 0
}

export function calculateArchetypalDistance(
  blend1: HarmonicBlend,
  blend2: HarmonicBlend
): number {
  const muses = Object.keys(blend1) as MuseArchetype[]
  const squaredDifferences = muses.reduce((sum, muse) => {
    const diff = (blend1[muse] || 0) - (blend2[muse] || 0)
    return sum + diff * diff
  }, 0)
  
  return Math.sqrt(squaredDifferences)
}

export function calculateResonanceScore(
  blend1: HarmonicBlend,
  blend2: HarmonicBlend
): number {
  const muses = Object.keys(blend1) as MuseArchetype[]
  const dotProduct = muses.reduce((sum, muse) => {
    return sum + (blend1[muse] || 0) * (blend2[muse] || 0)
  }, 0)
  
  const magnitude1 = Math.sqrt(muses.reduce((sum, muse) => {
    return sum + Math.pow(blend1[muse] || 0, 2)
  }, 0))
  
  const magnitude2 = Math.sqrt(muses.reduce((sum, muse) => {
    return sum + Math.pow(blend2[muse] || 0, 2)
  }, 0))
  
  if (magnitude1 === 0 || magnitude2 === 0) return 0
  
  return dotProduct / (magnitude1 * magnitude2)
}

// ============================================================================
// Spiral Coordinate Mathematics
// ============================================================================

export function calculateSpiralCoordinates(
  harmonicBlend: HarmonicBlend,
  sacredRatios: SacredRatios
): SpiralCoordinates {
  const primaryStrength = Math.max(...Object.values(harmonicBlend))
  const diversityIndex = calculateDiversityIndex(harmonicBlend)
  
  // Calculate spiral parameters based on harmonic blend
  const radius = primaryStrength * 100 // Scale to meaningful range
  const angle = diversityIndex * 2 * Math.PI // Full rotation based on diversity
  const height = sacredRatios.phi * 50 // Height based on golden ratio
  
  // Determine spiral type based on dominant characteristics
  let spiralType: SpiralCoordinates['spiral_type'] = 'fibonacci'
  if (sacredRatios.phi > 1.6) spiralType = 'golden'
  else if (sacredRatios.pi > 3.2) spiralType = 'archimedes'
  else if (primaryStrength > 0.8) spiralType = 'logarithmic'
  
  const turnCount = diversityIndex * 3 // Number of turns based on diversity
  
  return {
    radius,
    angle,
    height,
    spiral_type: spiralType,
    turn_count: turnCount,
  }
}

export function getSpiralPointAt(
  coordinates: SpiralCoordinates,
  t: number // Parameter from 0 to 1
): { x: number; y: number; z: number } {
  const { radius, angle, height, spiral_type, turn_count } = coordinates
  
  const fullAngle = angle + (t * turn_count * 2 * Math.PI)
  
  let r: number
  switch (spiral_type) {
    case 'fibonacci':
      r = radius * Math.pow(SACRED_CONSTANTS.PHI, t)
      break
    case 'golden':
      r = radius * Math.pow(SACRED_CONSTANTS.PHI, fullAngle / (Math.PI / 2))
      break
    case 'archimedes':
      r = radius * (1 + t)
      break
    case 'logarithmic':
      r = radius * Math.exp(t * Math.log(2))
      break
    default:
      r = radius
  }
  
  const x = r * Math.cos(fullAngle)
  const y = r * Math.sin(fullAngle)
  const z = height * t
  
  return { x, y, z }
}

// ============================================================================
// Fitness Score Calculations
// ============================================================================

export function calculateMathematicalFitness(
  content: string,
  harmonicBlend: HarmonicBlend,
  sacredRatios: SacredRatios
): number {
  // Simplified mathematical fitness based on content structure
  const lines = content.split('\n').filter(line => line.trim())
  const wordCount = content.split(/\s+/).length
  
  // Golden ratio analysis
  const goldenRatioScore = lines.length > 0 ? 
    Math.abs(wordCount / lines.length - SACRED_CONSTANTS.PHI) / SACRED_CONSTANTS.PHI : 0
  
  // Harmonic structure analysis
  const harmonicScore = Object.values(harmonicBlend).reduce((sum, value) => sum + value, 0)
  
  // Sacred ratio alignment
  const sacredScore = Object.values(sacredRatios).reduce((sum, value) => {
    return sum + Math.abs(value - Math.round(value))
  }, 0) / Object.keys(sacredRatios).length
  
  // Combine scores (0 to 1 range)
  const fitness = (
    (1 - goldenRatioScore) * 0.4 +
    harmonicScore * 0.4 +
    (1 - sacredScore) * 0.2
  )
  
  return Math.max(0, Math.min(1, fitness))
}

export function calculateSemanticCoherence(
  content: string,
  theme: string,
  harmonicBlend: HarmonicBlend
): number {
  // Simplified semantic coherence calculation
  const words = content.toLowerCase().split(/\s+/)
  const themeWords = theme.toLowerCase().split(/\s+/)
  
  // Calculate theme relevance
  const themeScore = themeWords.reduce((score, themeWord) => {
    const occurrences = words.filter(word => word.includes(themeWord)).length
    return score + (occurrences / words.length)
  }, 0)
  
  // Calculate archetypal alignment
  const primaryMuse = Object.entries(harmonicBlend).reduce((max, [muse, value]) => {
    return value > max.value ? { muse, value } : max
  }, { muse: 'CALLIOPE', value: 0 })
  
  // Simple archetypal keyword matching (would be more sophisticated in practice)
  const archetypalKeywords = {
    CALLIOPE: ['epic', 'hero', 'journey', 'quest', 'legend'],
    CLIO: ['history', 'time', 'past', 'memory', 'chronicle'],
    ERATO: ['love', 'heart', 'passion', 'beauty', 'desire'],
    EUTERPE: ['music', 'rhythm', 'melody', 'harmony', 'song'],
    MELPOMENE: ['sorrow', 'tragedy', 'loss', 'grief', 'pain'],
    POLYHYMNIA: ['sacred', 'divine', 'holy', 'prayer', 'worship'],
    TERPSICHORE: ['dance', 'movement', 'grace', 'flow', 'rhythm'],
    THALIA: ['joy', 'laughter', 'comedy', 'mirth', 'celebration'],
    URANIA: ['stars', 'cosmos', 'universe', 'celestial', 'astronomy'],
    SOPHIA: ['wisdom', 'knowledge', 'truth', 'understanding', 'insight'],
    TECHNE: ['craft', 'skill', 'art', 'creation', 'mastery'],
    PSYCHE: ['soul', 'spirit', 'mind', 'consciousness', 'essence'],
  }
  
  const keywords = archetypalKeywords[primaryMuse.muse as MuseArchetype] || []
  const archetypalScore = keywords.reduce((score, keyword) => {
    const occurrences = words.filter(word => word.includes(keyword)).length
    return score + (occurrences / words.length)
  }, 0)
  
  // Combine scores
  const coherence = (themeScore * 0.6 + archetypalScore * 0.4)
  
  return Math.max(0, Math.min(1, coherence))
}

// ============================================================================
// Utility Functions
// ============================================================================

export function normalizeHarmonicBlend(blend: Partial<HarmonicBlend>): HarmonicBlend {
  const muses: MuseArchetype[] = [
    'CALLIOPE', 'CLIO', 'ERATO', 'EUTERPE', 'MELPOMENE', 'POLYHYMNIA',
    'TERPSICHORE', 'THALIA', 'URANIA', 'SOPHIA', 'TECHNE', 'PSYCHE'
  ]
  
  // Initialize with zeros
  const normalized = muses.reduce((acc, muse) => {
    acc[muse] = blend[muse] || 0
    return acc
  }, {} as HarmonicBlend)
  
  // Normalize to sum to 1
  const total = Object.values(normalized).reduce((sum, value) => sum + value, 0)
  if (total > 0) {
    Object.keys(normalized).forEach(muse => {
      normalized[muse as MuseArchetype] /= total
    })
  }
  
  return normalized
}

export function interpolateHarmonicBlend(
  blend1: HarmonicBlend,
  blend2: HarmonicBlend,
  t: number
): HarmonicBlend {
  const muses = Object.keys(blend1) as MuseArchetype[]
  const interpolated = {} as HarmonicBlend
  
  muses.forEach(muse => {
    interpolated[muse] = blend1[muse] * (1 - t) + blend2[muse] * t
  })
  
  return interpolated
}

export function generateSacredRatios(baseValues: Partial<SacredRatios> = {}): SacredRatios {
  return {
    phi: baseValues.phi || SACRED_CONSTANTS.PHI,
    pi: baseValues.pi || SACRED_CONSTANTS.PI,
    euler: baseValues.euler || SACRED_CONSTANTS.E,
    root2: baseValues.root2 || SACRED_CONSTANTS.SQRT2,
    root3: baseValues.root3 || SACRED_CONSTANTS.SQRT3,
    root5: baseValues.root5 || SACRED_CONSTANTS.SQRT5,
    silver: baseValues.silver || SACRED_CONSTANTS.SILVER_RATIO,
    bronze: baseValues.bronze || SACRED_CONSTANTS.BRONZE_RATIO,
    plastic: baseValues.plastic || SACRED_CONSTANTS.PLASTIC_NUMBER,
  }
}

export function formatFrequencyValue(value: number, precision: number = 3): string {
  return value.toFixed(precision)
}

export function formatPercentage(value: number, precision: number = 1): string {
  return `${(value * 100).toFixed(precision)}%`
}

export function formatSacredRatio(value: number, name: string): string {
  if (name === 'phi' && Math.abs(value - SACRED_CONSTANTS.PHI) < 0.01) {
    return 'φ (Golden Ratio)'
  }
  if (name === 'pi' && Math.abs(value - SACRED_CONSTANTS.PI) < 0.01) {
    return 'π (Pi)'
  }
  if (name === 'euler' && Math.abs(value - SACRED_CONSTANTS.E) < 0.01) {
    return 'e (Euler)'
  }
  
  return `${value.toFixed(3)} (${name})`
}

export default {
  SACRED_CONSTANTS,
  fibonacci,
  fibonacciSequence,
  calculateGoldenRatio,
  isGoldenRatio,
  goldenRectangleProportions,
  goldenSpiralPoint,
  calculateHarmonicMean,
  calculateSpecializationIndex,
  calculateDiversityIndex,
  calculateArchetypalDistance,
  calculateResonanceScore,
  calculateSpiralCoordinates,
  getSpiralPointAt,
  calculateMathematicalFitness,
  calculateSemanticCoherence,
  normalizeHarmonicBlend,
  interpolateHarmonicBlend,
  generateSacredRatios,
  formatFrequencyValue,
  formatPercentage,
  formatSacredRatio,
}
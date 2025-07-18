import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock window.ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock window.IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock WebGL Context for Three.js
const mockWebGLContext = {
  canvas: {},
  drawingBufferWidth: 1024,
  drawingBufferHeight: 768,
  getParameter: vi.fn(),
  getExtension: vi.fn(),
  // Add other WebGL methods as needed
};

HTMLCanvasElement.prototype.getContext = vi.fn().mockImplementation((contextId) => {
  if (contextId === 'webgl' || contextId === 'webgl2') {
    return mockWebGLContext;
  }
  return null;
});

// Mock requestAnimationFrame
global.requestAnimationFrame = vi.fn().mockImplementation((cb) => {
  setTimeout(cb, 16);
});

// Mock cancelAnimationFrame
global.cancelAnimationFrame = vi.fn();

// Mock fetch
global.fetch = vi.fn();

// Mock console methods to reduce noise during tests
global.console = {
  ...console,
  log: vi.fn(),
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
};

// Mock AudioContext for frequency analysis
global.AudioContext = vi.fn().mockImplementation(() => ({
  createAnalyser: vi.fn().mockReturnValue({
    connect: vi.fn(),
    disconnect: vi.fn(),
    fftSize: 2048,
    frequencyBinCount: 1024,
    getByteFrequencyData: vi.fn(),
    getByteTimeDomainData: vi.fn(),
  }),
  createOscillator: vi.fn().mockReturnValue({
    connect: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    frequency: { value: 440 },
  }),
  createGain: vi.fn().mockReturnValue({
    connect: vi.fn(),
    gain: { value: 1 },
  }),
  destination: {},
}));

// Mock performance API
global.performance = {
  ...performance,
  now: vi.fn().mockReturnValue(Date.now()),
  mark: vi.fn(),
  measure: vi.fn(),
  getEntriesByType: vi.fn().mockReturnValue([]),
  getEntriesByName: vi.fn().mockReturnValue([]),
  clearMarks: vi.fn(),
  clearMeasures: vi.fn(),
};

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn(),
};

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
  writable: true,
});

// Mock sessionStorage
Object.defineProperty(window, 'sessionStorage', {
  value: localStorageMock,
  writable: true,
});

// Mock URL.createObjectURL
global.URL.createObjectURL = vi.fn().mockReturnValue('mock-object-url');
global.URL.revokeObjectURL = vi.fn();

// Mock Worker
global.Worker = vi.fn().mockImplementation(() => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null,
  onerror: null,
}));

// Mock crypto for testing
Object.defineProperty(global, 'crypto', {
  value: {
    getRandomValues: vi.fn().mockReturnValue(new Uint8Array(16)),
    randomUUID: vi.fn().mockReturnValue('mock-uuid'),
  },
  writable: true,
});

// Set up environment variables for testing
process.env.NODE_ENV = 'test';
process.env.VITE_API_BASE_URL = 'http://localhost:8000';
process.env.VITE_WS_BASE_URL = 'ws://localhost:8000';

// Custom test utilities
export const createMockUser = (overrides = {}) => ({
  id: 'mock-user-id',
  username: 'testuser',
  email: 'test@example.com',
  profile_data: {
    interests: ['mathematics', 'art'],
    creativity_level: 0.8,
    preferred_themes: ['golden_ratio', 'nature'],
  },
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides,
});

export const createMockCreativeSession = (overrides = {}) => ({
  session_id: 'mock-session-id',
  user_id: 'mock-user-id',
  session_type: 'exploration',
  session_data: {
    prompt: 'Test creative prompt',
    generated_content: {
      text: 'Mock generated content',
      frequency_signature: [440, 554.37, 659.25],
      geometry_data: { spiral_points: [[0, 0], [1, 1.618]] },
    },
  },
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides,
});

export const createMockCommunity = (overrides = {}) => ({
  community_id: 'mock-community-id',
  name: 'Test Community',
  description: 'A test community',
  creator_id: 'mock-creator-id',
  community_type: 'open',
  settings: {
    privacy_level: 'public',
    moderation_level: 'moderate',
  },
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides,
});

export const createMockCreativeWork = (overrides = {}) => ({
  work_id: 'mock-work-id',
  title: 'Test Creative Work',
  description: 'A test creative work',
  creator_id: 'mock-creator-id',
  work_data: {
    frequency_signature: [440, 554.37, 659.25],
    geometry_data: { golden_ratio_presence: 0.95 },
    semantic_vector: [0.1, 0.3, 0.7, 0.2, 0.9],
  },
  tags: ['test', 'creative', 'work'],
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides,
});

export const mockApiResponse = (data: any, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: vi.fn().mockResolvedValue(data),
  text: vi.fn().mockResolvedValue(JSON.stringify(data)),
  headers: new Headers(),
});

export const mockApiError = (message: string, status = 500) => ({
  ok: false,
  status,
  json: vi.fn().mockResolvedValue({ error: message }),
  text: vi.fn().mockResolvedValue(JSON.stringify({ error: message })),
  headers: new Headers(),
});

// Clean up after each test
beforeEach(() => {
  vi.clearAllMocks();
  localStorageMock.clear();
});

// Additional setup for specific test environments
afterEach(() => {
  // Clean up any global state changes
  vi.restoreAllMocks();
});
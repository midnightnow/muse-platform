# MUSE Platform Frontend

A React-based frontend for the MUSE Platform - Computational Platonism Creative Discovery System.

## Overview

The MUSE Platform frontend provides an intuitive interface for users to:
- Complete personality assessments to generate archetypal frequency signatures
- Discover creative works through mathematical optimization
- Visualize frequency signatures with sacred geometry
- Connect with kindred spirits through resonance matching
- Collaborate on creative discoveries in real-time

## Features

### Core Features
- **Personality Assessment**: Interactive questionnaire generating frequency signatures
- **Creative Discovery**: Real-time optimization interface with live fitness tracking
- **Frequency Visualization**: 3D spiral coordinates and radar chart displays
- **Community Platform**: Resonant feed and kindred spirit discovery
- **Archetypal Theming**: Dynamic UI themes based on user's primary Muse

### Technical Features
- **TypeScript**: Full type safety throughout the application
- **React 18**: Modern React with hooks and concurrent features
- **Tailwind CSS**: Utility-first styling with custom MUSE design system
- **Framer Motion**: Smooth animations and transitions
- **Recharts**: Interactive data visualizations
- **Zustand**: Lightweight state management
- **React Router**: Client-side routing with protected routes
- **Axios**: HTTP client with interceptors and error handling
- **Socket.io**: Real-time WebSocket communication
- **Radix UI**: Accessible component primitives

## Project Structure

```
src/
├── components/          # React components
│   ├── ui/             # Reusable UI components
│   ├── community/      # Community-specific components
│   └── visualizations/ # Data visualization components
├── pages/              # Page components
├── hooks/              # Custom React hooks
├── stores/             # Zustand state stores
├── services/           # API service layer
├── utils/              # Utility functions
├── types/              # TypeScript type definitions
└── assets/             # Static assets
```

## Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

### Environment Variables

Create a `.env.local` file with:
```
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Key Components

### MuseApp
Main application component handling routing, authentication, and global state.

### MuseAssessment
Multi-step personality assessment form that generates frequency signatures.

### MuseDiscoveryInterface
Real-time creative discovery interface with constraint optimization.

### MuseFrequencyDisplay
Visualization of user's archetypal frequency signature using radar charts and sacred geometry.

### MuseCommunityPage
Community hub with resonant feed and kindred spirit discovery.

## State Management

### Stores
- **MuseStore**: User authentication, profile, and theme management
- **DiscoveryStore**: Discovery sessions and real-time updates
- **CommunityStore**: Community feed and social interactions

### API Integration
- **MuseAPIService**: Centralized API client with authentication
- **Custom Hooks**: React hooks for API operations and state management

## Theming System

### Archetypal Themes
Each of the 12 Muses has a unique theme:
- **CALLIOPE**: Epic poetry - purple tones
- **CLIO**: History - red tones
- **ERATO**: Love poetry - pink tones
- **EUTERPE**: Music - green tones
- **MELPOMENE**: Tragedy - blue tones
- **POLYHYMNIA**: Sacred hymns - violet tones
- **TERPSICHORE**: Dance - orange tones
- **THALIA**: Comedy - yellow tones
- **URANIA**: Astronomy - sky blue tones
- **SOPHIA**: Wisdom - indigo tones
- **TECHNE**: Craft - purple tones
- **PSYCHE**: Soul - magenta tones

### Sacred Geometry Patterns
- Golden ratio grids and spirals
- Fibonacci sequences in layouts
- Mathematical constants in proportions
- Sacred geometry background patterns

## Mathematical Utilities

### Sacred Geometry
- Golden ratio calculations
- Fibonacci sequence generation
- Sacred spiral coordinates
- Geometric pattern generation

### Frequency Calculations
- Harmonic blend normalization
- Archetypal distance calculations
- Resonance score computation
- Fitness score algorithms

## Development

### Code Style
- ESLint configuration for TypeScript
- Prettier for code formatting
- Tailwind CSS for consistent styling
- TypeScript strict mode enabled

### Testing
- React Testing Library setup
- Jest for unit testing
- MSW for API mocking
- Component testing utilities

### Build Process
- Vite for fast development and building
- Code splitting for optimal loading
- Bundle analysis and optimization
- Production-ready deployment

## API Integration

### Authentication
- JWT token management
- Automatic token refresh
- Protected route guards
- User session persistence

### Real-time Features
- WebSocket connection management
- Live discovery updates
- Collaborative session support
- Real-time notifications

### Error Handling
- Global error boundaries
- API error interceptors
- User-friendly error messages
- Retry mechanisms

## Accessibility

### Features
- ARIA labels and roles
- Keyboard navigation
- Screen reader support
- High contrast mode
- Focus management
- Semantic HTML structure

### Design Considerations
- Color contrast compliance
- Typography accessibility
- Interactive element sizing
- Loading state indicators
- Error state handling

## Performance

### Optimizations
- Code splitting by route
- Lazy loading of components
- Image optimization
- Bundle size optimization
- Caching strategies

### Monitoring
- Performance metrics
- Error tracking
- User analytics
- Load time monitoring

## Deployment

### Production Build
```bash
npm run build
npm run preview
```

### Environment Configuration
- Production API endpoints
- CDN configuration
- Service worker setup
- Analytics integration

## Contributing

### Development Workflow
1. Create feature branch
2. Implement with tests
3. Update documentation
4. Submit pull request

### Code Quality
- TypeScript strict mode
- ESLint and Prettier
- Component testing
- Performance audits

## Architecture Decisions

### Why React?
- Component-based architecture
- Rich ecosystem
- TypeScript support
- Performance optimization

### Why Zustand?
- Lightweight state management
- Simple API
- TypeScript support
- Minimal boilerplate

### Why Tailwind CSS?
- Utility-first approach
- Consistent design system
- Responsive design
- Performance benefits

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues, please contact the MUSE Platform team or create an issue in the repository.
# MUSE Platform Backend

## 🎭 Computational Platonism Creative Discovery System

The MUSE Platform backend implements a revolutionary approach to creative discovery based on **Computational Platonism** - the philosophical idea that mathematical structures exist in a realm of eternal forms, and creativity is the process of discovering (not generating) these pre-existing patterns.

### Core Philosophy

Unlike traditional AI systems that generate content, MUSE **discovers** creative works that already exist in the mathematical realm. Our three-engine system coordinates to map users' archetypal frequencies and guide them toward their optimal creative expressions.

## 🏗️ Architecture

### Three Core Engines

1. **🎭 Frequency Engine** (`muse/core/frequency_engine.py`)
   - Maps personality traits to archetypal frequencies using the 12 Muses
   - Generates unique frequency signatures for each user
   - Handles hardware entropy integration for true randomness

2. **📐 Sacred Geometry Calculator** (`muse/core/sacred_geometry_calculator.py`)
   - Applies sacred mathematical constants (φ, π, fibonacci sequence)
   - Structures creative works according to universal proportions
   - Implements geometric optimization algorithms

3. **🧠 Semantic Projection Engine** (`muse/core/semantic_projection_engine.py`)
   - Bridges meaning and mathematics through semantic-mathematical mappings
   - Ensures discovered works are both mathematically elegant and emotionally resonant
   - Provides word embeddings and semantic analysis

### Service Layer

1. **🎼 Discovery Orchestrator** (`muse/services/discovery_orchestrator.py`)
   - Coordinates all three engines for unified creative discovery
   - Manages discovery sessions and optimization iterations
   - Handles both individual and collaborative discovery modes

2. **🤝 Resonance Matcher** (`muse/services/resonance_matcher.py`)
   - Calculates archetypal similarity between users
   - Provides community matching and compatibility analysis
   - Implements frequency-based social connections

3. **👥 Community Curator** (`muse/services/community_curator.py`)
   - Manages content curation based on archetypal resonance
   - Provides personalized feeds and recommendations
   - Handles social interactions and community engagement

### API Layer

1. **Core API** (`muse/api/main.py`)
   - Personality assessment and frequency signature generation
   - Discovery session management
   - Signature tuning and optimization

2. **Integration API** (`muse/api/integration.py`)
   - Real-time discovery endpoints
   - Constraint optimization
   - WebSocket streaming for live discovery
   - Collaborative session management

3. **Community API** (`muse/api/community.py`)
   - User profile management
   - Content sharing and social interactions
   - Community feeds and galleries
   - Resonance-based matching

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL (optional, SQLite used by default)
- Hardware entropy source (optional, `/dev/urandom` fallback)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/muse-platform.git
   cd muse-platform/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Quick Start

1. **Run startup script**
   ```bash
   python startup.py
   ```

2. **Or start manually**
   ```bash
   python main.py
   ```

3. **Visit the API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Development Mode

```bash
# Start with auto-reload
python startup.py --reload

# Reset database on startup
python startup.py --reset-db

# Run health checks only
python startup.py --check-only

# Custom port and host
python startup.py --host 127.0.0.1 --port 8080
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./muse_platform.db` | Database connection string |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `RELOAD` | `false` | Enable auto-reload |
| `WORKERS` | `1` | Number of worker processes |
| `DB_ECHO` | `false` | Enable SQLAlchemy query logging |

### Database Configuration

#### SQLite (Default)
```bash
DATABASE_URL=sqlite:///./muse_platform.db
```

#### PostgreSQL
```bash
DATABASE_URL=postgresql://user:password@localhost/muse_platform
```

## 📊 Database Schema

The platform uses SQLAlchemy ORM with the following key models:

- **UserProfile**: User information and frequency signature cache
- **FrequencySignature**: Complete archetypal frequency data
- **CommunityCreation**: Shared creative works with mathematical analysis
- **CollaborativeSession**: Multi-user discovery sessions
- **Comment/Like/Follow**: Social interaction models
- **ResonanceCache**: Cached resonance calculations for performance

## 🎯 API Endpoints

### Core API (`/api/v1/`)

- `POST /assessment/complete` - Complete personality assessment
- `GET /signatures/{id}` - Get frequency signature
- `POST /signatures/{id}/tune` - Tune frequency signature
- `POST /sessions/start` - Start discovery session
- `GET /sessions/{id}/status` - Get session status
- `POST /sessions/{id}/continue` - Continue discovery session
- `POST /sessions/{id}/complete` - Complete discovery session

### Integration API (`/api/v1/integration/`)

- `POST /live/discover-poem` - Real-time poetry discovery
- `POST /live/optimize-constraints` - Constraint optimization
- `WS /live/stream-discovery/{id}` - WebSocket discovery streaming
- `POST /live/collaborative-session` - Create collaborative session
- `GET /live/collaborative-session/{id}` - Get session details
- `POST /live/collaborative-session/{id}/join` - Join session

### Community API (`/api/v1/community/`)

- `POST /profiles/create` - Create user profile
- `GET /profiles/{id}` - Get user profile
- `PUT /profiles/{id}` - Update user profile
- `POST /creations/share` - Share creation
- `GET /gallery` - Community gallery
- `GET /feed/{user_id}` - Personalized feed
- `GET /kindred/{user_id}` - Find kindred spirits
- `POST /follow/{user_id}` - Follow user
- `POST /creations/{id}/like` - Like creation
- `POST /creations/{id}/comment` - Comment on creation

## 🔄 Discovery Process

The discovery process follows these phases:

1. **Initialization** - User provides theme and preferences
2. **Frequency Alignment** - Map user's archetypal signature to constraints
3. **Geometric Optimization** - Apply sacred geometry for structural harmony
4. **Semantic Projection** - Ensure meaning-mathematics alignment
5. **Creative Synthesis** - Coordinate all engines for unified discovery
6. **Refinement** - Iterative optimization based on fitness scores
7. **Completion** - Final optimization and result generation

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=muse

# Run specific test file
pytest tests/test_frequency_engine.py

# Run dependency tests
python -c "from muse.dependencies import test_all_dependencies; print(test_all_dependencies())"
```

### Health Checks
```bash
# System health check
curl http://localhost:8000/health

# Dependency check
python startup.py --check-only

# Database health
python -c "from database import health_check; print(health_check())"
```

## 📈 Performance

### Optimization Features

- **Singleton Pattern**: Core engines are cached as singletons
- **Dependency Injection**: Centralized dependency management
- **Connection Pooling**: Database connection pooling
- **Resonance Caching**: Cached resonance calculations
- **Async Processing**: Asynchronous endpoint processing

### Monitoring

- Request/response logging
- Performance metrics in headers
- Health check endpoints
- Dependency status monitoring

## 🔐 Security

### Best Practices

- Input validation with Pydantic models
- SQL injection prevention with SQLAlchemy ORM
- CORS configuration
- Error handling without information leakage
- Hardware entropy for cryptographic randomness

## 🤝 Contributing

### Development Setup

1. Install development dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Set up pre-commit hooks
   ```bash
   pre-commit install
   ```

3. Run code formatting
   ```bash
   black muse/
   flake8 muse/
   mypy muse/
   ```

### Code Standards

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing
- **Docstrings** for all public functions

## 📚 Mathematical Foundation

The platform is built on these mathematical principles:

### Archetypal Frequencies

Each of the 12 Muses corresponds to a specific frequency:
- **Calliope**: 432 Hz (Epic resonance)
- **Clio**: 528 Hz (Historical truth)
- **Erato**: 639 Hz (Love frequency)
- **Euterpe**: 741 Hz (Musical harmony)
- And more...

### Sacred Geometry Constants

- **φ (Phi)**: Golden ratio (1.618...)
- **π (Pi)**: Circle constant (3.14159...)
- **e**: Euler's number (2.718...)
- **Fibonacci**: Natural sequence (1, 1, 2, 3, 5, 8, 13...)

### Frequency Signature Calculation

```python
signature = MuseFrequencyEngine.generate_frequency_signature({
    'personality_traits': user_traits,
    'creative_preferences': user_preferences,
    'mathematical_affinity': user_math_prefs
})
```

## 🔗 Integration

### WebSocket Integration

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/integration/live/stream-discovery/session_id');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Discovery update:', data);
};
```

### REST API Integration

```python
import requests

# Complete assessment
response = requests.post('http://localhost:8000/api/v1/assessment/complete', json={
    'user_id': 'user123',
    'creative_preferences': {...},
    'personality_traits': {...},
    'mathematical_affinity': {...}
})

signature = response.json()
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ancient Greek Muses for archetypal inspiration
- Mathematical constants discovered by great mathematicians
- Open source community for foundational tools
- Computational Platonism philosophical framework

---

*The MUSE Platform: Where mathematics meets creativity, and discovery becomes art.*
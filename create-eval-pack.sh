#!/bin/bash

# MUSE Platform - Create Evaluation Package
# Creates a minimal but complete package for code review

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Creating MUSE evaluation package...${NC}"

# Create dist directory
mkdir -p .dist

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="muse_eval_pack_${TIMESTAMP}.tar.gz"

# Create the archive with essential files
tar -czf ".dist/${FILENAME}" \
    README.md \
    IMPLEMENTATION_COMPLETE.md \
    EVALUATION.md \
    MUSE_SHOWCASE_FOR_SEIYA.md \
    LICENSE 2>/dev/null || echo "(No LICENSE file)" \
    backend/main.py \
    backend/requirements.txt \
    backend/Dockerfile 2>/dev/null || echo "(No Dockerfile)" \
    backend/muse/api/music.py \
    backend/muse/core/sacred_geometry_calculator.py \
    backend/muse/core/frequency_engine.py \
    backend/muse/core/predictive_music_engine.py \
    backend/muse/core/empire_resonance_sync.py \
    frontend/package.json \
    frontend/src/components/MuseSoundEngine.tsx \
    launch-muse.sh \
    stop-muse.sh \
    2>/dev/null || true

echo -e "${GREEN}✅ Created evaluation package: .dist/${FILENAME}${NC}"
echo -e "${YELLOW}Size: $(du -h .dist/${FILENAME} | cut -f1)${NC}"

# Create a simple extraction guide
cat > .dist/EXTRACT_AND_RUN.txt << 'EOF'
MUSE Platform - Quick Start Guide
==================================

1. Extract the archive:
   tar -xzf muse_eval_pack_*.tar.gz

2. Backend setup:
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload

3. Frontend setup:
   cd frontend
   npm install
   npm run dev

4. Test the API:
   curl http://localhost:8000/api/muse/music/golden-melody?base_freq=440&length=8

5. Open browser:
   http://localhost:3000 (Frontend)
   http://localhost:8000/docs (API Documentation)

For detailed evaluation guide, see EVALUATION.md
For philosophy and architecture, see MUSE_SHOWCASE_FOR_SEIYA.md
EOF

echo -e "${BLUE}Created extraction guide: .dist/EXTRACT_AND_RUN.txt${NC}"

# Create email template
cat > .dist/email_template.txt << 'EOF'
Subject: MUSE Platform - Mathematical Music Discovery Engine

Hi Seiya,

Following up on our discussion about music and code, I'm sharing MUSE - a platform that treats creativity as mathematical discovery rather than generation.

Core concept: We use sacred geometry (golden ratio φ, Fibonacci sequences) to discover pre-existing musical forms in the Platonic realm, similar to how your "OS in 1000 Lines" reveals the essential nature of operating systems.

Key files to review:
1. sacred_geometry_calculator.py - Mathematical foundations
2. frequency_engine.py - 12 Muse archetypes with frequencies
3. predictive_music_engine.py - Music generation from math
4. MuseSoundEngine.tsx - Web Audio synthesis

The system maps personality traits to archetypal frequencies, then uses Pythagorean intervals and golden ratio progressions to generate actual music.

I'd particularly value your thoughts on:
- The mathematical approach to creativity
- System architecture and API design
- Balance between determinism and variation

The attached package includes everything needed to run locally. See EVALUATION.md for a quick review guide.

Best regards,
[Your name]

GitHub: https://github.com/midnightnow/muse-platform
EOF

echo -e "${BLUE}Created email template: .dist/email_template.txt${NC}"
echo ""
echo -e "${GREEN}📦 Evaluation package ready!${NC}"
echo -e "${GREEN}📧 Email template ready!${NC}"
echo -e "${GREEN}📖 Extraction guide ready!${NC}"
echo ""
echo -e "Send these files:"
echo -e "  1. ${YELLOW}.dist/${FILENAME}${NC}"
echo -e "  2. ${YELLOW}.dist/EXTRACT_AND_RUN.txt${NC}"
echo ""
echo -e "Use this email template:"
echo -e "  ${YELLOW}.dist/email_template.txt${NC}"
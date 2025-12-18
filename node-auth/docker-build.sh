#!/bin/bash

# ==========================================
# SETRAF Auth Backend - Docker Build Script
# ==========================================

set -e  # Exit on error

echo "🔨 Building SETRAF Authentication Backend Docker Image..."
echo "=================================================="

# Variables
IMAGE_NAME="belikanm/setraf-auth"
VERSION="1.0.0"
DOCKERFILE="Dockerfile"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is accessible
echo -e "${YELLOW}🔍 Checking Docker...${NC}"
if ! /mnt/c/Program\ Files/Docker/Docker/resources/bin/docker --version > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker not accessible!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker OK${NC}"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}❌ Dockerfile not found!${NC}"
    exit 1
fi

# Display build information
echo ""
echo -e "${YELLOW}📦 Build Configuration:${NC}"
echo "   Image: $IMAGE_NAME"
echo "   Tags: $VERSION, latest"
echo "   Dockerfile: $DOCKERFILE"
echo ""

# Start build
echo -e "${YELLOW}🚀 Starting build...${NC}"
START_TIME=$(date +%s)

/mnt/c/Program\ Files/Docker/Docker/resources/bin/docker build \
    -t $IMAGE_NAME:$VERSION \
    -t $IMAGE_NAME:latest \
    -f $DOCKERFILE \
    . 2>&1 | tee docker-build.log

BUILD_EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=================================================="

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Build completed successfully!${NC}"
    echo ""
    echo "⏱️  Duration: ${DURATION}s"
    echo ""
    echo "📦 Images created:"
    /mnt/c/Program\ Files/Docker/Docker/resources/bin/docker images | grep $IMAGE_NAME
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Test locally: docker run -p 5000:5000 --env-file ../.env $IMAGE_NAME:latest"
    echo "   2. Push to Docker Hub: ./docker-push.sh"
    echo ""
else
    echo -e "${RED}❌ Build failed!${NC}"
    echo "Check docker-build.log for details"
    exit 1
fi

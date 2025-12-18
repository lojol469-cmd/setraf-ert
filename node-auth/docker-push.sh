#!/bin/bash

# ==========================================
# SETRAF Auth Backend - Docker Push Script
# ==========================================

set -e  # Exit on error

echo "🚀 Pushing SETRAF Authentication Backend to Docker Hub..."
echo "=================================================="

# Variables
IMAGE_NAME="belikanm/setraf-auth"
VERSION="1.0.0"

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

# Check if image exists
echo -e "${YELLOW}🔍 Checking if image exists...${NC}"
if ! /mnt/c/Program\ Files/Docker/Docker/resources/bin/docker images | grep -q "$IMAGE_NAME"; then
    echo -e "${RED}❌ Image $IMAGE_NAME not found!${NC}"
    echo "Run ./docker-build.sh first"
    exit 1
fi
echo -e "${GREEN}✅ Image found${NC}"

# Check if logged in to Docker Hub
echo ""
echo -e "${YELLOW}🔐 Checking Docker Hub authentication...${NC}"
if ! /mnt/c/Program\ Files/Docker/Docker/resources/bin/docker info | grep -q "Username"; then
    echo -e "${YELLOW}⚠️  Not logged in to Docker Hub${NC}"
    echo "Please login:"
    /mnt/c/Program\ Files/Docker/Docker/resources/bin/docker login
fi
echo -e "${GREEN}✅ Authenticated${NC}"

# Display push information
echo ""
echo -e "${YELLOW}📤 Push Configuration:${NC}"
echo "   Image: $IMAGE_NAME"
echo "   Tags: $VERSION, latest"
echo ""

# Get image size
IMAGE_SIZE=$(/mnt/c/Program\ Files/Docker/Docker/resources/bin/docker images $IMAGE_NAME:latest --format "{{.Size}}")
echo "   Size: $IMAGE_SIZE"
echo ""

# Confirm push
read -p "🤔 Ready to push to Docker Hub? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Push cancelled"
    exit 0
fi

# Push version tag
echo ""
echo -e "${YELLOW}📤 Pushing $IMAGE_NAME:$VERSION...${NC}"
START_TIME=$(date +%s)

/mnt/c/Program\ Files/Docker/Docker/resources/bin/docker push $IMAGE_NAME:$VERSION

PUSH_VERSION_EXIT=$?
END_TIME=$(date +%s)
DURATION_VERSION=$((END_TIME - START_TIME))

if [ $PUSH_VERSION_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Failed to push version $VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pushed $IMAGE_NAME:$VERSION (${DURATION_VERSION}s)${NC}"

# Push latest tag
echo ""
echo -e "${YELLOW}📤 Pushing $IMAGE_NAME:latest...${NC}"
START_TIME=$(date +%s)

/mnt/c/Program\ Files/Docker/Docker/resources/bin/docker push $IMAGE_NAME:latest

PUSH_LATEST_EXIT=$?
END_TIME=$(date +%s)
DURATION_LATEST=$((END_TIME - START_TIME))

if [ $PUSH_LATEST_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Failed to push latest tag${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pushed $IMAGE_NAME:latest (${DURATION_LATEST}s)${NC}"

# Get image digest
DIGEST=$(/mnt/c/Program\ Files/Docker/Docker/resources/bin/docker inspect --format='{{index .RepoDigests 0}}' $IMAGE_NAME:latest 2>/dev/null | cut -d'@' -f2 || echo "N/A")

# Summary
echo ""
echo "=================================================="
echo -e "${GREEN}🎉 Push completed successfully!${NC}"
echo ""
echo "📦 Pushed images:"
echo "   - $IMAGE_NAME:$VERSION"
echo "   - $IMAGE_NAME:latest"
echo ""
echo "🔗 Docker Hub: https://hub.docker.com/r/belikanm/setraf-auth"
echo "📊 Image size: $IMAGE_SIZE"
echo "🆔 Digest: $DIGEST"
echo ""
echo "⏱️  Total time:"
echo "   - Version push: ${DURATION_VERSION}s"
echo "   - Latest push: ${DURATION_LATEST}s"
echo ""
echo "✅ Ready to deploy with:"
echo "   docker pull $IMAGE_NAME:latest"
echo "   docker run -p 5000:5000 --env-file .env $IMAGE_NAME:latest"
echo ""

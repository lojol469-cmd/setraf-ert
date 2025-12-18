# Frontend React 3D
FROM node:20-alpine AS builder

WORKDIR /app

# Copie package files
COPY frontend/package*.json ./

# Installation dépendances
RUN npm install

# Copie code source
COPY frontend/ ./

# Build production
RUN npm run build

# Image production
FROM nginx:alpine

# Copie build
COPY --from=builder /app/dist /usr/share/nginx/html

# Configuration nginx custom
COPY docker/nginx-frontend.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]

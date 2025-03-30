# Stage 1: Build
FROM node:18-alpine AS build
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy application code
COPY . .


# Expose port and start the development server with autoreload
EXPOSE 3000
CMD ["npm", "run", "serve"]

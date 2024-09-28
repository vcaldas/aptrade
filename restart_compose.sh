COMPOSE_FILE_PATH="/home/vcaldas/aptrade/compose.yml"

# Stop the Docker Compose services
docker-compose -f $COMPOSE_FILE_PATH down

# Start the Docker Compose services
docker-compose -f $COMPOSE_FILE_PATH up -d

echo "Docker Compose services have been restarted."
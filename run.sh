#!/bin/bash

IMAGE_NAME=zanon-pm

docker run --rm \
  --user $(id -u):$(id -g) \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/data:/app/data" \
  -e PYTHONPATH=/app/src \
  $IMAGE_NAME
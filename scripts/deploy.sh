#!/bin/bash
# Deployment script for Plant Disease Detector

set -e

echo "========================================="
echo "Plant Disease Detector - Deployment"
echo "========================================="
echo

# Configuration
REGISTRY="${DOCKER_REGISTRY:-docker.io/yourname}"
IMAGE_NAME="plant-disease-detector"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        log_warn "kubectl is not installed (required for Kubernetes deployment)"
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build CPU version
    log_info "Building CPU image..."
    docker build --target cpu -t "${IMAGE_NAME}:${VERSION}" .
    docker tag "${IMAGE_NAME}:${VERSION}" "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    
    # Build API version
    log_info "Building API image..."
    docker build --target api -t "${IMAGE_NAME}:api-${VERSION}" .
    docker tag "${IMAGE_NAME}:api-${VERSION}" "${REGISTRY}/${IMAGE_NAME}:api-${VERSION}"
    
    # Build GPU version (optional)
    if [ "${BUILD_GPU}" == "true" ]; then
        log_info "Building GPU image..."
        docker build --target gpu -t "${IMAGE_NAME}:gpu-${VERSION}" .
        docker tag "${IMAGE_NAME}:gpu-${VERSION}" "${REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}"
    fi
    
    log_info "Docker images built successfully"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."
    
    docker push "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker push "${REGISTRY}/${IMAGE_NAME}:api-${VERSION}"
    
    if [ "${BUILD_GPU}" == "true" ]; then
        docker push "${REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}"
    fi
    
    log_info "Images pushed successfully"
}

# Deploy with Docker Compose
deploy_compose() {
    log_info "Deploying with Docker Compose..."
    
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml not found"
        exit 1
    fi
    
    docker-compose down
    docker-compose up -d
    
    log_info "Waiting for services to start..."
    sleep 10
    
    # Health check
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_info "API is healthy"
    else
        log_warn "API health check failed"
    fi
    
    if curl -f http://localhost:8501 &> /dev/null; then
        log_info "Web UI is healthy"
    else
        log_warn "Web UI health check failed"
    fi
    
    log_info "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Update image tags in deployment
    sed -i.bak "s|image:.*plant-disease-detector:.*|image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}|g" kubernetes/deployment.yaml
    sed -i.bak "s|image:.*plant-disease-detector:api-.*|image: ${REGISTRY}/${IMAGE_NAME}:api-${VERSION}|g" kubernetes/deployment.yaml
    
    # Apply manifests
    kubectl apply -f kubernetes/deployment.yaml
    
    # Wait for rollout
    log_info "Waiting for deployment to complete..."
    kubectl rollout status deployment/plant-disease-api -n plant-disease-detector
    kubectl rollout status deployment/plant-disease-web -n plant-disease-detector
    
    # Get service URLs
    log_info "Deployment completed. Services:"
    kubectl get services -n plant-disease-detector
    
    # Restore original deployment file
    mv kubernetes/deployment.yaml.bak kubernetes/deployment.yaml
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    docker run --rm \
        -v "$(pwd)/tests:/app/tests" \
        "${REGISTRY}/${IMAGE_NAME}:${VERSION}" \
        pytest tests/ -v
    
    log_info "Tests passed"
}

# Main deployment flow
main() {
    case "${DEPLOYMENT_TYPE}" in
        "compose")
            check_prerequisites
            build_images
            deploy_compose
            ;;
        "kubernetes")
            check_prerequisites
            build_images
            push_images
            deploy_kubernetes
            ;;
        "all")
            check_prerequisites
            run_tests
            build_images
            push_images
            log_info "Images built and pushed. Choose deployment method:"
            log_info "  - docker-compose up -d  (Docker Compose)"
            log_info "  - kubectl apply -f kubernetes/  (Kubernetes)"
            ;;
        *)
            log_error "Invalid DEPLOYMENT_TYPE: ${DEPLOYMENT_TYPE}"
            echo "Usage: DEPLOYMENT_TYPE=<compose|kubernetes|all> ./scripts/deploy.sh"
            exit 1
            ;;
    esac
    
    echo
    log_info "========================================="
    log_info "Deployment completed successfully!"
    log_info "========================================="
}

# Run main function
main "$@"

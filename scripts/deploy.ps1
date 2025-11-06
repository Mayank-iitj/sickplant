# PowerShell Deployment script for Plant Disease Detector

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("compose", "kubernetes", "all")]
    [string]$DeploymentType = "compose",
    
    [Parameter(Mandatory=$false)]
    [string]$Registry = "docker.io/yourname",
    
    [Parameter(Mandatory=$false)]
    [string]$Version = "latest",
    
    [Parameter(Mandatory=$false)]
    [switch]$BuildGPU
)

$ErrorActionPreference = "Stop"

$ImageName = "plant-disease-detector"

# Colors for output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Plant Disease Detector - Deployment" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed"
        exit 1
    }
    
    if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        Write-Warn "kubectl is not installed (required for Kubernetes deployment)"
    }
    
    Write-Info "Prerequisites check passed"
}

# Build Docker images
function Build-Images {
    Write-Info "Building Docker images..."
    
    # Build CPU version
    Write-Info "Building CPU image..."
    docker build --target cpu -t "${ImageName}:${Version}" .
    docker tag "${ImageName}:${Version}" "${Registry}/${ImageName}:${Version}"
    
    # Build API version
    Write-Info "Building API image..."
    docker build --target api -t "${ImageName}:api-${Version}" .
    docker tag "${ImageName}:api-${Version}" "${Registry}/${ImageName}:api-${Version}"
    
    # Build GPU version (optional)
    if ($BuildGPU) {
        Write-Info "Building GPU image..."
        docker build --target gpu -t "${ImageName}:gpu-${Version}" .
        docker tag "${ImageName}:gpu-${Version}" "${Registry}/${ImageName}:gpu-${Version}"
    }
    
    Write-Info "Docker images built successfully"
}

# Push images to registry
function Push-Images {
    Write-Info "Pushing images to registry..."
    
    docker push "${Registry}/${ImageName}:${Version}"
    docker push "${Registry}/${ImageName}:api-${Version}"
    
    if ($BuildGPU) {
        docker push "${Registry}/${ImageName}:gpu-${Version}"
    }
    
    Write-Info "Images pushed successfully"
}

# Deploy with Docker Compose
function Deploy-Compose {
    Write-Info "Deploying with Docker Compose..."
    
    if (-not (Test-Path "docker-compose.yml")) {
        Write-Error "docker-compose.yml not found"
        exit 1
    }
    
    docker-compose down
    docker-compose up -d
    
    Write-Info "Waiting for services to start..."
    Start-Sleep -Seconds 10
    
    # Health check
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Info "API is healthy"
        }
    } catch {
        Write-Warn "API health check failed"
    }
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Info "Web UI is healthy"
        }
    } catch {
        Write-Warn "Web UI health check failed"
    }
    
    Write-Info "Docker Compose deployment completed"
}

# Deploy to Kubernetes
function Deploy-Kubernetes {
    Write-Info "Deploying to Kubernetes..."
    
    if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        Write-Error "kubectl is not installed"
        exit 1
    }
    
    # Update image tags in deployment
    $deploymentFile = "kubernetes/deployment.yaml"
    $content = Get-Content $deploymentFile -Raw
    
    # Backup original file
    Copy-Item $deploymentFile "${deploymentFile}.bak"
    
    $content = $content -replace "image:.*plant-disease-detector:(?!api).*", "image: ${Registry}/${ImageName}:${Version}"
    $content = $content -replace "image:.*plant-disease-detector:api-.*", "image: ${Registry}/${ImageName}:api-${Version}"
    Set-Content $deploymentFile $content
    
    # Apply manifests
    kubectl apply -f $deploymentFile
    
    # Wait for rollout
    Write-Info "Waiting for deployment to complete..."
    kubectl rollout status deployment/plant-disease-api -n plant-disease-detector
    kubectl rollout status deployment/plant-disease-web -n plant-disease-detector
    
    # Get service URLs
    Write-Info "Deployment completed. Services:"
    kubectl get services -n plant-disease-detector
    
    # Restore original deployment file
    Move-Item "${deploymentFile}.bak" $deploymentFile -Force
}

# Run tests
function Test-Application {
    Write-Info "Running tests..."
    
    docker run --rm `
        -v "${PWD}/tests:/app/tests" `
        "${Registry}/${ImageName}:${Version}" `
        pytest tests/ -v
    
    Write-Info "Tests passed"
}

# Main deployment flow
try {
    switch ($DeploymentType) {
        "compose" {
            Test-Prerequisites
            Build-Images
            Deploy-Compose
        }
        "kubernetes" {
            Test-Prerequisites
            Build-Images
            Push-Images
            Deploy-Kubernetes
        }
        "all" {
            Test-Prerequisites
            Test-Application
            Build-Images
            Push-Images
            Write-Info "Images built and pushed. Choose deployment method:"
            Write-Info "  - docker-compose up -d  (Docker Compose)"
            Write-Info "  - kubectl apply -f kubernetes/  (Kubernetes)"
        }
    }
    
    Write-Host ""
    Write-Info "========================================="
    Write-Info "Deployment completed successfully!"
    Write-Info "========================================="
    
} catch {
    Write-Error "Deployment failed: $_"
    exit 1
}

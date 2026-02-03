#!/usr/bin/env bash
###############################################################################
# Install Kubeflow Pipelines Standalone
#
# This script installs Kubeflow Pipelines (KFP) on your existing Kubernetes
# cluster using the official manifests.
#
# Usage:
#   ./scripts/install_kubeflow_pipelines.sh
#
# Prerequisites:
#   - kubectl configured with access to your cluster
#   - Cluster with at least 4GB RAM available
###############################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
KFP_VERSION="${KFP_VERSION:-2.0.5}"
NAMESPACE="kubeflow"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Kubeflow Pipelines Installation${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Version: $KFP_VERSION"
echo "Namespace: $NAMESPACE"
echo ""

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${YELLOW}Error: kubectl not found${NC}"
    exit 1
fi

# Check cluster access
echo -e "${BLUE}[1/5]${NC} Checking cluster access..."
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${YELLOW}Error: Cannot access Kubernetes cluster${NC}"
    echo "Please configure kubectl first"
    exit 1
fi
echo -e "${GREEN}✓ Cluster accessible${NC}"

# Install cluster-scoped resources
echo ""
echo -e "${BLUE}[2/5]${NC} Installing cluster-scoped resources..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$KFP_VERSION"

# Wait for CRDs to be established
echo -e "${BLUE}[3/5]${NC} Waiting for CRDs to be established..."
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# Install KFP components
echo ""
echo -e "${BLUE}[4/5]${NC} Installing Kubeflow Pipelines components..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$KFP_VERSION"

# Wait for deployments
echo ""
echo -e "${BLUE}[5/5]${NC} Waiting for deployments to be ready (this may take a few minutes)..."
echo "This will wait up to 10 minutes for all components to be ready..."

# Wait for each deployment separately with timeout
deployments=(
    "ml-pipeline"
    "ml-pipeline-ui"
    "mysql"
    "minio"
    "workflow-controller"
)

for deployment in "${deployments[@]}"; do
    echo -n "  Waiting for $deployment... "
    if kubectl wait --for=condition=available --timeout=600s \
        -n $NAMESPACE deployment/$deployment &> /dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠ (may still be starting)${NC}"
    fi
done

# Create pipeline runner service account
echo ""
echo -e "${BLUE}Creating pipeline runner service account...${NC}"
kubectl create serviceaccount pipeline-runner -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
kubectl create clusterrolebinding pipeline-runner-binding \
    --clusterrole=cluster-admin \
    --serviceaccount=$NAMESPACE:pipeline-runner \
    --dry-run=client -o yaml | kubectl apply -f -

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ Kubeflow Pipelines Installed Successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "Access the UI:"
echo -e "  ${BLUE}kubectl port-forward -n $NAMESPACE svc/ml-pipeline-ui 8080:80${NC}"
echo -e "  Then open: ${BLUE}http://localhost:8080${NC}"
echo ""
echo -e "Install KFP SDK:"
echo -e "  ${BLUE}pip install kfp==2.5.0${NC}"
echo ""
echo -e "Check status:"
echo -e "  ${BLUE}kubectl get pods -n $NAMESPACE${NC}"
echo ""
echo -e "View logs:"
echo -e "  ${BLUE}kubectl logs -n $NAMESPACE -l app=ml-pipeline${NC}"
echo ""
echo -e "${GREEN}============================================${NC}"

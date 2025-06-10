#!/bin/bash

# Quick test script for basic functionality
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"
TEST_COUNT=0
PASSED_COUNT=0

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED_COUNT++))
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

test_endpoint() {
    local test_name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_status="${5:-200}"

    ((TEST_COUNT++))
    echo -n "Test $TEST_COUNT: $test_name... "

    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "%{http_code}" -o /tmp/test_response "$API_URL$endpoint" 2>/dev/null || echo "000")
    else
        response=$(curl -s -w "%{http_code}" -o /tmp/test_response -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_URL$endpoint" 2>/dev/null || echo "000")
    fi

    if [ "$response" = "$expected_status" ]; then
        success "$test_name"
        return 0
    else
        error "$test_name (Expected: $expected_status, Got: $response)"
        if [ -f /tmp/test_response ]; then
            echo "Response: $(cat /tmp/test_response | head -c 200)"
        fi
        return 1
    fi
}

echo "🧪 FintelligenceAI Quick System Tests"
echo "====================================="

# Check if system is running
log "Checking if system is accessible..."
if ! curl -f -s "$API_URL/health" > /dev/null 2>&1; then
    error "System is not accessible at $API_URL"
    echo "Please make sure the system is running with: ./scripts/start_system.sh"
    exit 1
fi

echo "System is accessible, running tests..."
echo ""

# Test 1: Health Check
test_endpoint "Health Check" "GET" "/health"

# Test 2: Agent Status
test_endpoint "Agent Status" "GET" "/agents/status"

# Test 3: Agent Health
test_endpoint "Agent Health Check" "GET" "/agents/health"

# Test 4: Simple Research Query
test_endpoint "Research Agent" "POST" "/agents/research" '{
    "query": "What is ErgoScript?",
    "max_results": 2
}'

# Test 5: System Reset (should be allowed)
test_endpoint "System Reset" "POST" "/agents/reset" '{}'

# Test 6: Optimization Metrics
test_endpoint "Optimization Metrics" "GET" "/optimization/metrics/summary"

# Test 7: Simple Code Generation
test_endpoint "Simple Code Generation" "POST" "/agents/generate-code/simple" '{
    "requirements": "Create a simple box",
    "complexity": "beginner"
}'

# Test 8: Code Validation
test_endpoint "Code Validation" "POST" "/agents/validate-code" '{
    "code": "val myBox = SELF",
    "context": "Simple validation test"
}'

# Test 9: API Documentation
test_endpoint "API Documentation" "GET" "/docs"

# Test 10: OpenAPI Schema
test_endpoint "OpenAPI Schema" "GET" "/openapi.json"

echo ""
echo "🧪 Quick Test Results"
echo "===================="
echo "Tests Run: $TEST_COUNT"
echo "Tests Passed: $PASSED_COUNT"
echo "Tests Failed: $((TEST_COUNT - PASSED_COUNT))"

if [ $PASSED_COUNT -eq $TEST_COUNT ]; then
    echo ""
    success "🎉 All quick tests passed! Your system is working correctly."
    echo ""
    echo "Next steps:"
    echo "• Run full validation: python3 scripts/validate_system.py"
    echo "• Open API docs: http://localhost:8000/docs"
    echo "• Try the Jupyter notebook: http://localhost:8888"
    exit 0
else
    echo ""
    error "⚠️  Some tests failed. Your system may need attention."
    echo ""
    echo "Troubleshooting:"
    echo "• Check system logs: docker-compose logs -f"
    echo "• Restart system: docker-compose restart"
    echo "• Check .env configuration"
    exit 1
fi

# Cleanup
rm -f /tmp/test_response

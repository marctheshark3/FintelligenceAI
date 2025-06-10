#!/bin/bash

# FintelligenceAI API Test Script
# Quick tests to verify your FintelligenceAI instance is working

set -e

BASE_URL="http://localhost:8000"
echo "🧪 Testing FintelligenceAI API at $BASE_URL"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test API endpoint
test_endpoint() {
    local name="$1"
    local method="$2" 
    local endpoint="$3"
    local data="$4"
    
    echo -n "Testing $name... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "%{http_code}" -X GET "$BASE_URL$endpoint" -o /tmp/response.json)
    else
        response=$(curl -s -w "%{http_code}" -H "Content-Type: application/json" -X POST "$BASE_URL$endpoint" -d "$data" -o /tmp/response.json)
    fi
    
    http_code="${response: -3}"
    
    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✅ PASSED${NC} ($http_code)"
        return 0
    else
        echo -e "${RED}❌ FAILED${NC} ($http_code)"
        if [ -f /tmp/response.json ]; then
            echo "Response: $(cat /tmp/response.json | head -c 200)..."
        fi
        return 1
    fi
}

# Test counter
PASSED=0
TOTAL=0

# Test 1: Basic Health Check
echo -e "\n${YELLOW}🏥 Health Checks${NC}"
if test_endpoint "API Health" "GET" "/health"; then ((PASSED++)); fi
((TOTAL++))

if test_endpoint "Agent Health" "GET" "/agents/health"; then ((PASSED++)); fi
((TOTAL++))

# Test 2: Agent Status
echo -e "\n${YELLOW}🤖 Agent Status${NC}"
if test_endpoint "Agent Status" "GET" "/agents/status"; then ((PASSED++)); fi
((TOTAL++))

# Test 3: Simple Operations (these might fail due to configuration)
echo -e "\n${YELLOW}🔬 Simple Operations${NC}"
echo "Note: These tests may fail if OpenAI API key is not configured"

# Simple research query  
if test_endpoint "Research Query" "POST" "/agents/research" '{"query": "ErgoScript basics", "scope": "quick"}'; then ((PASSED++)); fi
((TOTAL++))

# Simple code generation
if test_endpoint "Code Generation" "POST" "/agents/generate-code/simple" '{"description": "Hello world contract", "complexity_level": "beginner"}'; then ((PASSED++)); fi
((TOTAL++))

# Code validation
if test_endpoint "Code Validation" "POST" "/agents/validate-code" '{"code": "{ 1 + 1 }", "validation_criteria": {"syntax_check": true}}'; then ((PASSED++)); fi
((TOTAL++))

# Test 4: Optimization endpoints
echo -e "\n${YELLOW}⚡ Optimization Features${NC}"
if test_endpoint "Optimization Metrics" "GET" "/optimization/metrics/summary"; then ((PASSED++)); fi
((TOTAL++))

# Summary
echo -e "\n${YELLOW}📊 Test Summary${NC}"
echo "=================================="
echo "Passed: $PASSED/$TOTAL tests"

if [ $PASSED -eq $TOTAL ]; then
    echo -e "${GREEN}🎉 All tests passed! Your FintelligenceAI is working perfectly.${NC}"
    exit 0
elif [ $PASSED -ge 3 ]; then
    echo -e "${YELLOW}⚠️  Most tests passed. Some advanced features may need configuration.${NC}"
    echo "Check that your OpenAI API key is set in .env file for AI features."
    exit 0
else
    echo -e "${RED}❌ Many tests failed. Check that FintelligenceAI is running properly.${NC}"
    echo "Try: PYTHONPATH=src poetry run python src/fintelligence_ai/api/main.py"
    exit 1
fi 
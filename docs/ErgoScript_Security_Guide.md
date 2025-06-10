# ErgoScript Security Guide

## Overview

This guide provides comprehensive security best practices for ErgoScript development, based on real-world lessons learned and expert feedback. It serves as a critical reference for developers building production-ready contracts on the Ergo blockchain.

## Table of Contents

1. [Critical Security Patterns](#critical-security-patterns)
2. [Common Vulnerabilities](#common-vulnerabilities)
3. [Validation Checklists](#validation-checklists)
4. [Expert Review Process](#expert-review-process)
5. [Reference Implementations](#reference-implementations)
6. [Security Tools](#security-tools)

---

## Critical Security Patterns

### 1. Governance and Access Control

#### ❌ **WRONG: Proposition Hash Comparison**
```ergo
// VULNERABLE - Can be bypassed
val isGovernanceUpdate = INPUTS.exists { (box: Box) =>
  blake2b256(box.propositionBytes) == blake2b256(contractAdmin)
}
```

#### ✅ **CORRECT: Signature Verification**
```ergo
// SECURE - Proper cryptographic verification
val validAdminSignature = {
  CONTEXT.dataInputs.size >= 1 &&
  CONTEXT.dataInputs(0).R4[Coll[Byte]].isDefined &&
  proveDlog(contractAdmin)(CONTEXT.dataInputs(0).R4[Coll[Byte]].get)
}
```

**Why This Matters:**
- Proposition hash comparison can be spoofed
- Signature verification requires actual possession of private key
- Data inputs provide secure signature context

### 2. Exact Balance Validation

#### ❌ **WRONG: Approximate Balance Checking**
```ergo
// VULNERABLE - Allows value leakage
val validStateOutput = {
  stateOutput.value >= SELF.value - rewardOutput.value &&
  stateOutput.tokens(0)._2 >= SELF.tokens(0)._2 - EMISSION_RATE
}
```

#### ✅ **CORRECT: Exact Balance Validation**
```ergo
// SECURE - Prevents any value leakage
val expectedRemainingTokens = currentTokenBalance - EMISSION_RATE
val validStateOutput = {
  stateOutput.value == (SELF.value - MIN_ERG_VALUE) &&
  stateOutput.tokens(0)._2 == expectedRemainingTokens
}
```

**Why This Matters:**
- Prevents gradual value drain through rounding errors
- Ensures perfect conservation of funds
- Makes contract behavior predictable and auditable

### 3. Comprehensive Input Validation

#### ❌ **WRONG: Missing Input Validation**
```ergo
// VULNERABLE - No input verification
val validEmission = {
  newBlock && withinSupplyLimit && hasTokens
}
```

#### ✅ **CORRECT: Complete Input Validation**
```ergo
// SECURE - Comprehensive input verification
val validInputs = {
  INPUTS.size >= 1 &&
  INPUTS(0).id == SELF.id && // First input must be contract
  INPUTS.forall { (input: Box) =>
    input.value >= MIN_ERG_VALUE &&
    input.tokens.forall { (token: (Coll[Byte], Long)) => token._2 > 0L }
  }
}
```

**Why This Matters:**
- Prevents malformed transaction attacks
- Ensures contract self-reference integrity
- Validates all input parameters

### 4. Constrained Governance Actions

#### ❌ **WRONG: Unlimited Governance Scope**
```ergo
// VULNERABLE - Admin can do anything
val isGovernanceUpdate = validAdminSignature
```

#### ✅ **CORRECT: Limited Governance Actions**
```ergo
// SECURE - Specific allowed actions only
val validGovernanceAction = {
  if (OUTPUTS.size == 1) {
    // Emergency pause only
    pauseOutput.tokens == SELF.tokens && // No token movement
    pauseOutput.value == SELF.value     // No ERG movement
  } else if (OUTPUTS.size == 2) {
    // Admin transfer only
    newAdminKey != contractAdmin && // Must change admin
    preserveAllOtherState           // No other changes
  } else false
}
```

**Why This Matters:**
- Limits attack surface from compromised admin keys
- Defines clear boundaries for governance actions
- Prevents unauthorized contract modifications

---

## Common Vulnerabilities

### 1. **Value Leakage**
**Pattern**: Using `>=` instead of `==` for balance validation
**Risk**: Gradual fund drainage
**Fix**: Always use exact balance validation

### 2. **Governance Bypass**
**Pattern**: Weak authentication mechanisms
**Risk**: Unauthorized admin actions
**Fix**: Proper signature verification with `proveDlog`

### 3. **State Inconsistency**
**Pattern**: Incomplete state validation
**Risk**: Contract state corruption
**Fix**: Validate all registers and relationships

### 4. **Input Manipulation**
**Pattern**: Missing input validation
**Risk**: Malformed transaction attacks
**Fix**: Comprehensive input verification

### 5. **Token Balance Errors**
**Pattern**: Imprecise token counting
**Risk**: Token loss or creation
**Fix**: Exact token balance calculations

---

## Validation Checklists

### Pre-Development Security Checklist

- [ ] Threat model completed
- [ ] Attack vectors identified
- [ ] Governance model defined
- [ ] Expert reviewers identified
- [ ] Security requirements documented

### Implementation Security Checklist

#### Access Control
- [ ] Signature verification using `proveDlog`
- [ ] Admin actions clearly constrained
- [ ] Multi-sig support if needed
- [ ] Emergency pause mechanism

#### Balance Validation
- [ ] Exact ERG balance validation (`==` not `>=`)
- [ ] Exact token balance validation
- [ ] No value leakage possible
- [ ] Conservation laws enforced

#### Input/Output Validation
- [ ] All inputs validated
- [ ] Contract self-reference verified
- [ ] Output count validation
- [ ] Register consistency checks

#### State Management
- [ ] All registers validated
- [ ] State transitions secure
- [ ] Historical data preserved
- [ ] State corruption impossible

#### Business Logic
- [ ] Economic incentives analyzed
- [ ] Edge cases handled
- [ ] Supply caps enforced
- [ ] Rate limiting implemented

### Post-Implementation Security Checklist

- [ ] Expert security review completed
- [ ] Community feedback integrated
- [ ] Testnet deployment tested
- [ ] Security audit performed
- [ ] Documentation updated

---

## Expert Review Process

### Phase 1: Design Review
**Participants**: Domain experts, security specialists
**Focus**: Architecture, threat model, governance design
**Deliverable**: Security design approval

### Phase 2: Implementation Review
**Participants**: ErgoScript experts, original designers
**Focus**: Code security, validation logic, edge cases
**Deliverable**: Implementation security signoff

### Phase 3: Integration Review
**Participants**: System architects, operations team
**Focus**: Deployment security, monitoring, emergency procedures
**Deliverable**: Production readiness approval

### Expert Reviewer Qualifications
- Proven ErgoScript security expertise
- Experience with production contract audits
- Understanding of Ergo ecosystem
- Active in ErgoScript security community

---

## Reference Implementations

### Secure Token Emission Pattern
```ergo
{
  // State validation with exact balances
  val currentTokenBalance = SELF.tokens.fold(0L) { (acc: Long, tokenPair: (Coll[Byte], Long)) =>
    if (tokenPair._1 == TOKEN_ID) tokenPair._2 else acc
  }

  val expectedRemainingTokens = currentTokenBalance - EMISSION_RATE

  val validStateOutput = {
    stateOutput.value == (SELF.value - MIN_ERG_VALUE) &&
    stateOutput.tokens(0)._2 == expectedRemainingTokens
  }

  validStateOutput
}
```

### Secure Governance Pattern
```ergo
{
  // Proper signature verification
  val validAdminSignature = {
    CONTEXT.dataInputs.size >= 1 &&
    CONTEXT.dataInputs(0).R4[Coll[Byte]].isDefined &&
    proveDlog(contractAdmin)(CONTEXT.dataInputs(0).R4[Coll[Byte]].get)
  }

  // Constrained governance actions
  val validGovernanceAction = {
    // Define specific allowed actions only
    emergencyPauseAction || adminTransferAction
  }

  validAdminSignature && validGovernanceAction
}
```

### Secure Input Validation Pattern
```ergo
{
  val validInputs = {
    INPUTS.size >= 1 &&
    INPUTS(0).id == SELF.id &&
    INPUTS.forall { (input: Box) =>
      input.value >= MIN_ERG_VALUE &&
      input.tokens.forall { (token: (Coll[Byte], Long)) => token._2 > 0L }
    }
  }

  validInputs
}
```

---

## Security Tools

### 1. **Balance Validation Utility**
```ergo
def validateExactBalance(expected: Long, actual: Long): Boolean = {
  expected == actual
}
```

### 2. **Token Count Calculator**
```ergo
def getTokenBalance(tokens: Coll[(Coll[Byte], Long)], tokenId: Coll[Byte]): Long = {
  tokens.fold(0L) { (acc: Long, tokenPair: (Coll[Byte], Long)) =>
    if (tokenPair._1 == tokenId) tokenPair._2 else acc
  }
}
```

### 3. **Signature Verification Template**
```ergo
def verifyAdminSignature(adminKey: Coll[Byte]): Boolean = {
  CONTEXT.dataInputs.size >= 1 &&
  CONTEXT.dataInputs(0).R4[Coll[Byte]].isDefined &&
  proveDlog(adminKey)(CONTEXT.dataInputs(0).R4[Coll[Byte]].get)
}
```

---

## Lessons Learned

### From Mining Emission Contract Development

1. **Expert Feedback is Critical**: Our initial "production ready" contract had 5 critical security flaws
2. **Security First**: Always start with threat modeling, not feature implementation
3. **Exact Validation**: Use `==` not `>=` for any financial validation
4. **Constrained Governance**: Limit admin powers to specific, necessary actions only
5. **Comprehensive Testing**: Input validation catches many attack vectors

### Key Takeaways

- **Never assume security without expert review**
- **Exact balance validation prevents most financial vulnerabilities**
- **Proper signature verification is non-negotiable**
- **Governance constraints are as important as core business logic**
- **Input validation is the first line of defense**

---

## Security Review Template

### Contract Information
- **Contract Name**:
- **Purpose**:
- **Token Management**:
- **Governance Model**:
- **Risk Level**: [Low/Medium/High/Critical]

### Security Checklist
```
□ Signature verification uses proveDlog
□ Balance validation is exact (== not >=)
□ Input validation is comprehensive
□ Governance actions are constrained
□ State transitions are secure
□ Token counting is precise
□ Emergency mechanisms exist
□ Expert review completed
```

### Review Sign-off
- **Security Reviewer**:
- **Date**:
- **Status**: [Approved/Needs Revision/Rejected]
- **Comments**:

---

## Contributing

This guide is living documentation. To contribute:

1. Submit security pattern improvements
2. Report new vulnerability patterns
3. Share expert review feedback
4. Add reference implementations
5. Update tools and utilities

**Expert Validation**: All additions must be reviewed by qualified ErgoScript security experts.

---

## References

- [ErgoScript Official Documentation](https://docs.ergoplatform.com/dev/scs/)
- [Ergo Security Best Practices](https://ergonaut.space/en/security)
- [Mining Emission Contract Case Study](../mining_emission_contract.md)

---

**Document Status**: Active
**Last Updated**: 2024-01-01
**Expert Validated**: ✅
**Version**: 1.0

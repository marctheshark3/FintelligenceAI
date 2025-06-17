# Sample ErgoScript Tutorial

This is a sample tutorial for testing the knowledge base ingestion functionality.

## Introduction to ErgoScript

ErgoScript is a powerful smart contract language built for the Ergo blockchain. It provides:

- **Security**: Non-Turing complete design prevents infinite loops
- **Flexibility**: Rich type system and cryptographic operations
- **Efficiency**: Optimized for UTXO model

## Basic Example

Here's a simple ErgoScript contract that locks funds until a specific blockchain height:

```ergoscript
{
  val unlockHeight = 100000
  sigmaProp(HEIGHT > unlockHeight)
}
```

This contract:
1. Defines a threshold height of 100,000 blocks
2. Returns true only when the current blockchain height exceeds the threshold
3. Allows spending only after the condition is met

## Advanced Concepts

### Box Model
Every UTXO in Ergo is represented as a "box" containing:
- Value (nanoERGs)
- Tokens
- Registers (data storage)
- Guard script (ErgoScript)

### Sigma Propositions
ErgoScript uses sigma propositions for cryptographic proofs:
- `sigmaProp()` - Basic signature requirement
- `atLeast()` - Multi-signature thresholds
- `anyOf()` - Logical OR operations
- `allOf()` - Logical AND operations

## Best Practices

1. **Always validate inputs**: Check box values and token amounts
2. **Use proper error handling**: Ensure graceful failure modes
3. **Optimize gas usage**: Minimize computation complexity
4. **Test thoroughly**: Use testnet for validation

## Example: Token Sale Contract

```ergoscript
{
  val tokenPrice = 1000000L // Price in nanoERGs
  val tokenId = fromBase64("tokenIdHere")

  val validPayment = OUTPUTS(0).value >= tokenPrice
  val tokenTransfer = OUTPUTS(1).tokens(0)._1 == tokenId

  sigmaProp(validPayment && tokenTransfer)
}
```

This contract demonstrates:
- Price validation
- Token transfer verification
- Multi-condition logic

## Resources

- Official Documentation: https://docs.ergoplatform.com
- ErgoScript Tutorial: https://ergoscript.org
- Community Examples: https://github.com/ergoplatform/eips

---

*This tutorial is designed to help developers understand ErgoScript fundamentals and build secure smart contracts on the Ergo blockchain.*

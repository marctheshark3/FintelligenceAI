# Getting Started with ErgoScript

This tutorial will help you get started with ErgoScript development.

## Prerequisites

- Basic understanding of blockchain concepts
- Knowledge of functional programming principles
- Ergo node setup

## Your First Contract

Let's create a simple contract that locks funds until a specific height:

```ergoscript
{
  val unlockHeight = 100000
  sigmaProp(HEIGHT > unlockHeight)
}
```

This contract demonstrates:
- Variable declaration with `val`
- Blockchain height checking
- Basic proposition logic

## Next Steps

1. Set up your development environment
2. Learn about data types
3. Explore box operations
4. Practice with more complex contracts

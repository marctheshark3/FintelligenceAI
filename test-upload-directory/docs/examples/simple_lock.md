# Simple Time Lock Example

This example demonstrates a time-based lock contract.

## Contract Code

```ergoscript
{
  val lockTime = 1609459200000L // Unix timestamp
  sigmaProp(HEIGHT > lockTime)
}
```

## Explanation

- `lockTime`: Target timestamp for unlocking
- `HEIGHT`: Current blockchain height
- Returns true when current time exceeds lock time

## Usage

1. Deploy with specific timestamp
2. Funds remain locked until time passes
3. Can be used for vesting schedules

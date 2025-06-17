# Ergo Python AppKit - Correct Usage Examples

This document provides the correct usage patterns for the ergo-python-appkit library based on the official repository at https://github.com/ergo-pad/ergo-python-appkit.

## Installation

```bash
pip install git+https://github.com/ergo-pad/ergo-python-appkit.git
```

## Basic Setup

The ergo-python-appkit uses JPype to bridge to the JVM and requires Java to be installed.

### Correct Imports

```python
# CORRECT imports
import ergo_python_appkit.appkit
from ergo_python_appkit.appkit import ErgoAppKit
from org.ergoplatform.appkit import Address
```

**Important**: You must import `ergo_python_appkit.appkit` BEFORE importing Java classes.

## Simple Transaction Example

Here's the correct way to send ERG from one address to another:

```python
import ergo_python_appkit.appkit
from ergo_python_appkit.appkit import ErgoAppKit
from org.ergoplatform.appkit import Address

# Initialize ErgoAppKit
appKit = ErgoAppKit(
    "http://myergonode:9053",      # Node URL
    "mainnet",                      # Network type (mainnet/testnet)
    "http://api.ergoplatform.org",  # Explorer URL
    "nodeapikey"                    # Node API key
)

# Sender address
myAddress = "9gF2wmTWcEX2EZu5QY3dJwURep1HWsXjUZdVavVWESun4Sp8BYj"

# Collect boxes to spend 1 ERG (1e9 nanoERG)
inputs = appKit.boxesToSpend(address=myAddress, nergToSpend=int(1e9))

# Define output box
outputBox = appKit.buildOutBox(
    value=int(1e9),                              # Amount in nanoERG
    tokens=None,                                 # No tokens
    registers=None,                              # No registers
    contract=appKit.contractFromAddress(myAddress)  # Contract for the address
)

# Build the unsigned transaction
unsignedTx = appKit.buildUnsignedTransaction(
    inputs=inputs,
    outputs=[outputBox],
    fee=int(1e6),                                # Fee in nanoERG
    sendChangeTo=Address.create(myAddress).getErgoAddress()
)

# Sign the transaction with the node
signedTx = appKit.signTransactionWithNode(unsignedTx)

# Send the transaction
appKit.sendTransaction(signedTx)
```

## Key Methods

### ErgoAppKit Constructor
```python
ErgoAppKit(node_url, network_type, explorer_url, api_key)
```

### Box Operations
```python
# Get boxes to spend
inputs = appKit.boxesToSpend(address=address, nergToSpend=amount)

# Build output box
outputBox = appKit.buildOutBox(
    value=amount,
    tokens=token_list,  # Optional
    registers=registers,  # Optional
    contract=contract
)
```

### Transaction Operations
```python
# Build unsigned transaction
unsignedTx = appKit.buildUnsignedTransaction(
    inputs=input_boxes,
    outputs=output_boxes,
    fee=fee_amount,
    sendChangeTo=change_address
)

# Sign with node
signedTx = appKit.signTransactionWithNode(unsignedTx)

# Send transaction
txId = appKit.sendTransaction(signedTx)
```

## Address Operations

```python
# Create address from string
myErgoAddress = Address.create("9gF2wmTWcEX2EZu5QY3dJwURep1HWsXjUZdVavVWESun4Sp8BYj")

# Get ErgoTree
print(myErgoAddress.toErgoContract().getErgoTree())

# Get contract from address
contract = appKit.contractFromAddress(address_string)
```

## Common Patterns

### Sending 2 ERG with Error Handling

```python
import ergo_python_appkit.appkit
from ergo_python_appkit.appkit import ErgoAppKit
from org.ergoplatform.appkit import Address
import logging

def send_erg(node_url, api_key, sender_address, recipient_address, amount_erg):
    """
    Send ERG from sender to recipient

    Args:
        node_url: Ergo node URL
        api_key: Node API key
        sender_address: Sender's address
        recipient_address: Recipient's address
        amount_erg: Amount in ERG (will be converted to nanoERG)
    """
    try:
        # Initialize ErgoAppKit
        appKit = ErgoAppKit(
            node_url,
            "mainnet",  # or "testnet"
            "http://api.ergoplatform.org",
            api_key
        )

        # Convert ERG to nanoERG
        amount_nerg = int(amount_erg * 1e9)
        fee_nerg = int(1e6)  # 0.001 ERG fee

        # Collect boxes to spend
        inputs = appKit.boxesToSpend(
            address=sender_address,
            nergToSpend=amount_nerg + fee_nerg
        )

        # Build output box for recipient
        outputBox = appKit.buildOutBox(
            value=amount_nerg,
            tokens=None,
            registers=None,
            contract=appKit.contractFromAddress(recipient_address)
        )

        # Build unsigned transaction
        unsignedTx = appKit.buildUnsignedTransaction(
            inputs=inputs,
            outputs=[outputBox],
            fee=fee_nerg,
            sendChangeTo=Address.create(sender_address).getErgoAddress()
        )

        # Sign and send
        signedTx = appKit.signTransactionWithNode(unsignedTx)
        txId = appKit.sendTransaction(signedTx)

        logging.info(f"Transaction sent successfully: {txId}")
        return txId

    except Exception as e:
        logging.error(f"Transaction failed: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    tx_id = send_erg(
        node_url="http://localhost:9053",
        api_key="your_api_key",
        sender_address="9gF2wmTWcEX2EZu5QY3dJwURep1HWsXjUZdVavVWESun4Sp8BYj",
        recipient_address="9f4QF8AD1nQ3nJahQVkMj8hFSVVzVom77b52JU7EW71Zexg6N8v",
        amount_erg=2.0
    )
    print(f"Transaction ID: {tx_id}")
```

## Important Notes

1. **Java Requirement**: Java 8 or higher must be installed
2. **Import Order**: Always import `ergo_python_appkit.appkit` before Java classes
3. **Network Types**: Use "mainnet" or "testnet" as strings
4. **Amount Units**: All amounts are in nanoERG (1 ERG = 1e9 nanoERG)
5. **Node Connection**: Requires a running Ergo node with API access

## Common Errors to Avoid

### ❌ Incorrect Imports
```python
# WRONG - these don't exist
from ergo_python_appkit import ErgoClient, TransactionBuilder
from ergo_appkit import ErgoAppKit
```

### ✅ Correct Imports
```python
# CORRECT
import ergo_python_appkit.appkit
from ergo_python_appkit.appkit import ErgoAppKit
from org.ergoplatform.appkit import Address
```

### ❌ Wrong Method Names
```python
# WRONG - these methods don't exist
ErgoClient.create()
TransactionBuilder()
```

### ✅ Correct Method Names
```python
# CORRECT
ErgoAppKit(node_url, network, explorer_url, api_key)
appKit.boxesToSpend()
appKit.buildOutBox()
appKit.buildUnsignedTransaction()
```

## Repository Links

- Official Repository: https://github.com/ergo-pad/ergo-python-appkit
- Documentation: https://docs.ergoplatform.com/dev/stack/appkit/appkit_py/
- Examples: See the repository README for more examples

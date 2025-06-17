# JavaScript Email Validation Examples

## Basic Email Validation Function

```javascript
function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}
```

## Advanced Email Validation

```javascript
function validateEmailAdvanced(email) {
    // More comprehensive regex pattern
    const emailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;

    if (!email || typeof email !== 'string') {
        return false;
    }

    return emailRegex.test(email);
}
```

## Usage Examples

```javascript
// Basic usage
console.log(validateEmail("test@example.com")); // true
console.log(validateEmail("invalid-email")); // false

// Advanced usage with error handling
try {
    const isValid = validateEmailAdvanced("user@domain.com");
    if (isValid) {
        console.log("Email is valid");
    } else {
        console.log("Email is invalid");
    }
} catch (error) {
    console.error("Validation error:", error);
}
```

## Form Validation Integration

```javascript
function validateForm() {
    const emailInput = document.getElementById('email');
    const email = emailInput.value;

    if (!validateEmail(email)) {
        alert('Please enter a valid email address');
        return false;
    }

    return true;
}

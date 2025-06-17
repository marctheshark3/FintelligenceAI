# FintelligenceAI Web Interface

This is a simple web interface for interacting with the FintelligenceAI agent system. It provides an easy-to-use frontend for the various AI agents and their capabilities.

## Features

- **Code Generation**: Generate ErgoScript code using natural language descriptions
- **Research**: Query the knowledge base for information about ErgoScript and blockchain development
- **Code Validation**: Validate ErgoScript code for syntax, security, and other criteria
- **Agent Status**: Monitor the status of all AI agents in the system

## Usage

1. Make sure your FintelligenceAI server is running on `http://localhost:8000`
2. Open `index.html` in a web browser
3. The interface will automatically check if the API is online (shown in the top-right corner)

### Code Generation Tab

- Enter a description of what you want the ErgoScript to do
- Select a use case (token, auction, oracle, etc.)
- Choose complexity level (beginner, intermediate, advanced)
- Add specific requirements and constraints
- Click "Generate ErgoScript" to get your code

### Research Tab

- Enter your research query about ErgoScript or blockchain development
- Choose research scope (comprehensive, focused, quick overview)
- Option to include examples in the results
- Click "Start Research" to get information

### Validate Code Tab

- Paste your ErgoScript code
- Provide context about the code's intended use
- Select validation criteria (syntax, semantic, security checks)
- Click "Validate Code" to check your code

### Agent Status Tab

- View the current status of all agents
- See active and completed task counts
- Monitor system health
- Click "Refresh Status" to update

## File Structure

```
ui/
├── index.html      # Main HTML file
├── styles.css      # CSS styling
├── app.js          # JavaScript functionality
└── README.md       # This file
```

## API Endpoints Used

- `GET /health` - Check API health
- `POST /agents/generate-code` - Generate ErgoScript code
- `POST /agents/research` - Research queries
- `POST /agents/validate-code` - Validate code
- `GET /agents/status` - Get agent status

## Features

- **Real-time API Status**: Shows whether the backend is online
- **Syntax Highlighting**: Code responses are highlighted for better readability
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Clear error messages when something goes wrong
- **Form Helpers**: Easy input management for requirements and constraints

## Troubleshooting

1. **API Offline**: Make sure the FintelligenceAI server is running on port 8000
2. **CORS Issues**: The server should allow connections from your browser
3. **Empty Responses**: Check the server logs for errors
4. **Form Not Working**: Make sure JavaScript is enabled in your browser

## Development

To modify the interface:

1. Edit `index.html` for structure changes
2. Edit `styles.css` for visual styling
3. Edit `app.js` for functionality changes

The interface uses:
- Vanilla JavaScript (no frameworks)
- Prism.js for syntax highlighting
- Font Awesome for icons
- Modern CSS Grid and Flexbox for layout

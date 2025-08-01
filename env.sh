# LLM-Story-Board Environment Configuration

# Required: Groq API Key
# Get your API key from: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Dataset Configuration
DATASET_PATH=dataset/description-in-isolation.json
MAX_IMAGES_FOR_SELECTION=30
MAX_STORY_LENGTH=2000

# Server Configuration
PORT=5000
DEBUG=False
RATE_LIMIT=10 per minute
SECRET_KEY=your-secret-key-change-this-in-production

# Optional: Logging
LOG_LEVEL=INFO
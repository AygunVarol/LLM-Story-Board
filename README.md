# LLM-Story-Board

AI-powered storyboard generator that creates visual narratives by intelligently selecting images from your dataset based on story content. Uses Groq's LLM API to analyze stories and match them with relevant images.

## Screenshot of UI

<img width="1919" height="1871" alt="Screenshot 2025-08-01 at 14-05-07 LLM Storyboard Generator" src="https://github.com/user-attachments/assets/00986c67-09b2-40ad-a2fc-2143c6d586c2" />

## Features

- 🎬 Generate storyboards from text stories
- 🖼️ Intelligent image selection from dataset
- 🎨 Clean web interface with real-time feedback
- ⚡ Fast processing with image filtering
- 🔄 Rate limiting and error handling
- 📱 Responsive design

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/AygunVarol/LLM-Story-Board.git
   cd LLM-Story-Board
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` file and add your API keys (see Configuration section)

4. **Prepare your dataset**
   - Place your image dataset JSON file in the `dataset/` directory
   - Name it `description-in-isolation.json` or update the path in `.env`

5. **Run the application**
   ```bash
   python llm-story-board.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Required: Groq API Key
GROQ_API_KEY=your_groq_api_key_here

# Optional: Dataset configuration
DATASET_PATH=dataset/description-in-isolation.json
MAX_IMAGES_FOR_SELECTION=30
MAX_STORY_LENGTH=2000

# Optional: Server configuration
PORT=5000
DEBUG=False
RATE_LIMIT=10 per minute
SECRET_KEY=your-secret-key-here
```

## Dataset Format

Your image dataset should be a JSON file with this structure:

```json
{
  "images": [
    {
      "id": "unique_image_id",
      "title": "Image title",
      "tags": "comma, separated, tags",
      "url_o": "https://example.com/image.jpg"
    }
  ]
}
```

## API Endpoints

- `GET /` - Main web interface
- `POST /generate_storyboard` - Generate storyboard from story
- `GET /health` - Health check endpoint
- `GET /debug_dataset` - Dataset information (for debugging)

## How It Works

1. **Story Analysis**: The LLM analyzes your story to extract keywords and themes
2. **Image Filtering**: Relevant images are selected based on story content
3. **Storyboard Generation**: AI creates 3-5 scenes using the best matching images
4. **Visual Presentation**: Results displayed in an intuitive web interface

## Requirements

- Python 3.7+
- Groq API key
- Image dataset in JSON format
- Internet connection

## Dependencies

- Flask - Web framework
- Groq - AI API client
- Flask-CORS - Cross-origin requests
- Flask-Limiter - Rate limiting

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

If you encounter any issues:
1. Check the logs in `storyboard_app.log`
2. Verify your API key and dataset format
3. Use the `/health` endpoint to check system status

## Screenshots

The application provides:
- Clean story input interface
- Real-time character counting
- Loading animations during processing
- Visual storyboard with images and descriptions
- Error handling with retry options

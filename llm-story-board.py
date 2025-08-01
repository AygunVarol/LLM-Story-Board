import json
import os
import random
import logging
import time
from functools import wraps
from typing import Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
import re
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('storyboard_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'YOUR_API_KEY')
    DATASET_PATH = os.getenv('DATASET_PATH', 'dataset/description-in-isolation.json')
    MAX_IMAGES_FOR_SELECTION = int(os.getenv('MAX_IMAGES_FOR_SELECTION', '30'))
    MAX_STORY_LENGTH = int(os.getenv('MAX_STORY_LENGTH', '2000'))
    RATE_LIMIT = os.getenv('RATE_LIMIT', '10 per minute')
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

app.config.from_object(Config)

# Alternative: Initialize limiter without app, then init later
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT]
)

# Add this line after app configuration
limiter.init_app(app)

# Initialize Groq client
try:
    client = Groq(api_key=Config.GROQ_API_KEY)
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    client = None

# Global variables
IMAGE_DATASET = None
DATASET_LOADED = False

def load_dataset() -> Optional[Dict]:
    """Load the image dataset from JSON file with error handling"""
    global IMAGE_DATASET, DATASET_LOADED
    
    try:
        if not os.path.exists(Config.DATASET_PATH):
            logger.error(f"Dataset file not found: {Config.DATASET_PATH}")
            return None
            
        with open(Config.DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        # Validate dataset structure
        if 'images' not in dataset:
            logger.error("Dataset missing 'images' key")
            return None
            
        if not isinstance(dataset['images'], list):
            logger.error("Dataset 'images' is not a list")
            return None
            
        logger.info(f"Dataset loaded successfully with {len(dataset['images'])} images")
        DATASET_LOADED = True
        return dataset
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        return None

def validate_image(img: Dict) -> bool:
    """Validate that an image has required fields"""
    if not isinstance(img, dict):
        return False
        
    # Check for URL in various possible fields
    url_fields = ['url_o', 'url', 'image_url', 'src', 'photo_url']
    has_url = any(field in img and img[field] for field in url_fields)
    
    return has_url and 'id' in img

def create_image_summary(images: List[Dict]) -> List[Dict]:
    """Create a summary of available images for the LLM"""
    summary = []
    
    for img in images:
        if not validate_image(img):
            continue
            
        # Try different possible URL field names
        url = None
        for url_field in ['url_o', 'url', 'image_url', 'src', 'photo_url']:
            if url_field in img and img[url_field]:
                url = img[url_field]
                break
        
        if not url:
            continue
            
        summary.append({
            "id": str(img.get("id", "unknown")),
            "title": str(img.get("title", "Untitled"))[:100],  # Limit title length
            "tags": str(img.get("tags", ""))[:200],  # Limit tags length
            "url": url
        })
    
    return summary

def extract_keywords(text: str) -> set:
    """Extract keywords from text with improved processing"""
    if not text:
        return set()
        
    # Convert to lowercase and extract words
    text_lower = text.lower()
    
    # Remove common stop words
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
        'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so'
    }
    
    words = set(re.findall(r'\b\w{3,}\b', text_lower))  # Only words 3+ chars
    return words - stop_words

def filter_relevant_images(story: str, images: List[Dict], max_images: int = 30) -> List[Dict]:
    """Filter images based on story keywords and randomly sample to reduce token usage"""
    
    if not story or not images:
        return []
        
    story_keywords = extract_keywords(story)
    
    # Score images based on relevance
    scored_images = []
    
    for img in images:
        if not validate_image(img):
            continue
            
        score = 0
        
        # Check title
        title = str(img.get("title", ""))
        title_keywords = extract_keywords(title)
        score += len(story_keywords.intersection(title_keywords)) * 3
        
        # Check tags
        tags = str(img.get("tags", ""))
        tag_keywords = extract_keywords(tags)
        score += len(story_keywords.intersection(tag_keywords)) * 2
        
        # Boost score for common themes
        common_themes = ['family', 'child', 'parent', 'home', 'christmas', 'holiday', 
                        'people', 'group', 'celebration', 'gathering', 'love', 'joy']
        
        for theme in common_themes:
            if theme in story.lower():
                if theme in title.lower() or theme in tags.lower():
                    score += 5
        
        scored_images.append((score, img))
    
    # Sort by score (highest first)
    scored_images.sort(key=lambda x: x[0], reverse=True)
    
    # Take top scored images
    top_images = [img for score, img in scored_images[:max_images//2] if score > 0]
    
    # Add some random images for variety if we need more
    if len(top_images) < max_images:
        remaining_images = [img for score, img in scored_images[max_images//2:]]
        random.shuffle(remaining_images)
        needed = max_images - len(top_images)
        top_images.extend(remaining_images[:needed])
    
    return top_images[:max_images]

def sanitize_story(story: str) -> str:
    """Sanitize and validate story input"""
    if not story:
        return ""
        
    # Remove excessive whitespace
    story = re.sub(r'\s+', ' ', story.strip())
    
    # Limit length
    if len(story) > Config.MAX_STORY_LENGTH:
        story = story[:Config.MAX_STORY_LENGTH]
        
    # Remove potentially harmful content (basic sanitization)
    story = re.sub(r'[<>]', '', story)
    
    return story

def generate_storyboard(story: str, image_dataset: Dict) -> Dict:
    """Generate storyboard using Groq API with filtered images"""
    
    if not client:
        return {"error": "Groq client not initialized. Please check your API key."}
    
    try:
        # Sanitize input
        story = sanitize_story(story)
        if not story:
            return {"error": "Invalid story provided"}
        
        # Get all images
        all_images = image_dataset.get("images", [])
        if not all_images:
            return {"error": "No images found in dataset"}
        
        # Filter to most relevant images
        relevant_images = filter_relevant_images(story, all_images, Config.MAX_IMAGES_FOR_SELECTION)
        
        if not relevant_images:
            return {"error": "No relevant images found for your story"}
        
        # Create summary of filtered images
        image_summary = create_image_summary(relevant_images)
        
        if not image_summary:
            return {"error": "No valid images with URLs found"}
        
        # Create the prompt
        prompt = f"""
You are a professional storyboard director. Create a compelling visual storyboard for the given story using the provided images.

Story: "{story}"

Available Images:
{json.dumps(image_summary, indent=2)}

Create a storyboard with 3-5 scenes that best tell this story. Respond ONLY with valid JSON in this exact format:

{{
  "storyboard": [
    {{
      "scene_number": 1,
      "scene_description": "Brief description of this scene",
      "selected_image_id": "exact_image_id_from_list",
      "image_title": "exact_image_title",
      "image_url": "exact_image_url",
      "narrative_explanation": "Why this image fits and advances the story"
    }}
  ],
  "overall_narrative": "How these images work together to tell the story"
}}

Requirements:
- Use only image IDs, titles, and URLs from the provided list
- Create 3-5 scenes maximum
- Each scene should advance the narrative
- Ensure image selections make logical sense for the story
"""

        # Make API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a professional storyboard director. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    storyboard_data = json.loads(json_str)
                    
                    # Validate response structure
                    if "storyboard" in storyboard_data:
                        logger.info(f"Successfully generated storyboard with {len(storyboard_data['storyboard'])} scenes")
                        return storyboard_data
                    else:
                        logger.warning("Response missing storyboard key")
                        
                logger.warning(f"Could not extract valid JSON from response (attempt {attempt + 1})")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {"error": "Failed to parse AI response after multiple attempts"}
                    
            except Exception as e:
                logger.error(f"API call error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {"error": f"API error: {str(e)}"}
                    
            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {"error": "Failed to generate storyboard after multiple attempts"}
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_storyboard: {e}")
        return {"error": "An unexpected error occurred"}

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

# Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "dataset_loaded": DATASET_LOADED,
        "groq_client_ready": client is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/debug_dataset')
@limiter.limit("5 per minute")
def debug_dataset():
    """Debug endpoint to see dataset structure"""
    if not DATASET_LOADED or IMAGE_DATASET is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    images = IMAGE_DATASET.get("images", [])
    debug_info = {
        "total_images": len(images),
        "valid_images": len([img for img in images if validate_image(img)]),
        "sample_images": images[:3] if images else [],
        "available_fields": list(images[0].keys()) if images else []
    }
    
    return jsonify(debug_info)

@app.route('/generate_storyboard', methods=['POST'])
@limiter.limit("5 per minute")
def generate_storyboard_endpoint():
    """Main endpoint for generating storyboards"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        story = data.get('story', '').strip()
        
        if not story:
            return jsonify({"error": "Story is required"}), 400
            
        if len(story) > Config.MAX_STORY_LENGTH:
            return jsonify({"error": f"Story too long. Maximum {Config.MAX_STORY_LENGTH} characters."}), 400
        
        # Check if dataset and client are ready
        if not DATASET_LOADED or IMAGE_DATASET is None:
            return jsonify({"error": "Dataset not loaded. Please contact administrator."}), 500
            
        if not client:
            return jsonify({"error": "AI service not available. Please contact administrator."}), 500
        
        # Generate storyboard
        start_time = time.time()
        storyboard = generate_storyboard(story, IMAGE_DATASET)
        end_time = time.time()
        
        logger.info(f"Storyboard generation took {end_time - start_time:.2f} seconds")
        
        return jsonify(storyboard)
        
    except Exception as e:
        logger.error(f"Error in generate_storyboard_endpoint: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="AI-powered storyboard generator using curated image datasets">
    <meta name="robots" content="noindex, nofollow">
    <title>LLM Storyboard Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            max-height: 300px;
            transition: border-color 0.3s ease;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .char-counter {
            text-align: right;
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        .char-counter.warning {
            color: #ff6b6b;
        }
        
        .generate-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            display: block;
            margin: 0 auto;
        }
        
        .generate-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        
        .generate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            text-align: center;
            margin: 20px 0;
            color: #667eea;
            font-size: 18px;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .storyboard-results {
            margin-top: 30px;
        }
        
        .scene-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #667eea;
        }
        
        .scene-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .scene-number {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
        }
        
        .scene-title {
            font-size: 1.4em;
            font-weight: 600;
            color: #333;
        }
        
        .scene-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            align-items: start;
        }
        
        .scene-image {
            width: 100%;
            max-width: 300px;
            height: 200px;
            object-fit: cover;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .scene-description {
            color: #555;
            line-height: 1.6;
        }
        
        .overall-narrative {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
        }
        
        .overall-narrative h3 {
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        
        .error {
            background: #ff6b6b;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
        }
        
        .info-banner {
            background: #4ecdc4;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .retry-btn {
            background: #ff6b6b;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
        }
        
        .retry-btn:hover {
            background: #ff5252;
        }
        
        @media (max-width: 768px) {
            .scene-content {
                grid-template-columns: 1fr;
            }
            
            .container {
                padding: 20px;
            }
            
            h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 LLM Storyboard Generator</h1>
        
        <div class="info-banner">
            📷 AI-powered storyboard creation using intelligently selected images from your dataset
        </div>
        
        <div class="input-section">
            <div class="input-group">
                <label for="story">Enter your story (max 2000 characters):</label>
                <textarea 
                    id="story" 
                    maxlength="2000"
                    placeholder="Write your story here... For example: 'A family gathering during Christmas where children help their parents at a craft fair, followed by visiting relatives and exploring childhood memories.'"
                ></textarea>
                <div id="charCounter" class="char-counter">0 / 2000 characters</div>
            </div>
            
            <button id="generateBtn" class="generate-btn">Generate Storyboard</button>
        </div>
        
        <div id="loading" class="loading" style="display: none;">
            <div class="spinner"></div>
            <span id="loadingText">Analyzing your story and selecting the best images...</span>
        </div>
        
        <div id="results" class="storyboard-results"></div>
    </div>

    <script>
        const generateBtn = document.getElementById('generateBtn');
        const storyInput = document.getElementById('story');
        const loadingDiv = document.getElementById('loading');
        const resultsDiv = document.getElementById('results');
        const charCounter = document.getElementById('charCounter');
        const loadingText = document.getElementById('loadingText');

        // Character counter
        storyInput.addEventListener('input', () => {
            const length = storyInput.value.length;
            charCounter.textContent = `${length} / 2000 characters`;
            
            if (length > 1800) {
                charCounter.classList.add('warning');
            } else {
                charCounter.classList.remove('warning');
            }
        });

        // Generate storyboard
        generateBtn.addEventListener('click', generateStoryboard);

        async function generateStoryboard() {
            const story = storyInput.value.trim();
            
            if (!story) {
                alert('Please enter a story first!');
                return;
            }
            
            if (story.length > 2000) {
                alert('Story is too long. Please keep it under 2000 characters.');
                return;
            }
            
            // Show loading
            loadingDiv.style.display = 'block';
            generateBtn.disabled = true;
            resultsDiv.innerHTML = '';
            
            let dots = 0;
            const loadingInterval = setInterval(() => {
                dots = (dots + 1) % 4;
                loadingText.textContent = 'Analyzing your story and selecting the best images' + '.'.repeat(dots);
            }, 500);
            
            try {
                const response = await fetch('/generate_storyboard', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ story: story })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayStoryboard(data);
                
            } catch (error) {
                console.error('Error:', error);
                displayError(error.message);
            } finally {
                clearInterval(loadingInterval);
                loadingDiv.style.display = 'none';
                generateBtn.disabled = false;
            }
        }

        function displayStoryboard(data) {
            let html = '';
            
            if (data.storyboard && data.storyboard.length > 0) {
                data.storyboard.forEach(scene => {
                    html += `
                        <div class="scene-card">
                            <div class="scene-header">
                                <div class="scene-number">${scene.scene_number}</div>
                                <div class="scene-title">${escapeHtml(scene.scene_description)}</div>
                            </div>
                            <div class="scene-content">
                                <div>
                                    <img src="${escapeHtml(scene.image_url)}" 
                                         alt="${escapeHtml(scene.image_title)}" 
                                         class="scene-image" 
                                         onerror="this.onerror=null; this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDMwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIzMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0xMjUgNzVMMTUwIDEwMEwxMjUgMTI1SDE3NUwxNTAgMTAwTDE3NSA3NUgxMjVaIiBmaWxsPSIjOUI5QjlCIi8+CjwvZz4KPC9zdmc+'; this.alt='Image not available';" 
                                         loading="lazy" />
                                    <p style="margin-top: 10px; font-weight: 600; color: #667eea;">${escapeHtml(scene.image_title)}</p>
                                </div>
                                <div class="scene-description">
                                    <p>${escapeHtml(scene.narrative_explanation)}</p>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                if (data.overall_narrative) {
                    html += `
                        <div class="overall-narrative">
                            <h3>Overall Narrative</h3>
                            <p>${escapeHtml(data.overall_narrative)}</p>
                        </div>
                    `;
                }
            } else {
                html = '<div class="error">No storyboard generated. Please try again.</div>';
            }
            
            resultsDiv.innerHTML = html;
        }

        function displayError(message) {
            resultsDiv.innerHTML = `
                <div class="error">
                    <strong>Error:</strong> ${escapeHtml(message)}
                    <br>
                    <button class="retry-btn" onclick="generateStoryboard()">Try Again</button>
                </div>
            `;
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""

# Initialize dataset on startup
IMAGE_DATASET = load_dataset()

if __name__ == '__main__':
    if not DATASET_LOADED:
        logger.error("Dataset failed to load. Please check your dataset file.")
        print("ERROR: Dataset not loaded. Please ensure 'description-in-isolation.json' exists.")
        exit(1)
    
    if not client:
        logger.error("Groq client not initialized. Please check your API key.")
        print("ERROR: Groq API key not configured. Please set GROQ_API_KEY environment variable.")
        exit(1)
    
    logger.info("Starting Flask application...")
    print(f"Dataset loaded: {len(IMAGE_DATASET.get('images', []))} images")
    print("Starting server on http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=Config.DEBUG,
        threaded=True
    )
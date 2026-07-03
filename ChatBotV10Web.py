import os
import sys
import re
import json
import threading
import urllib.parse
from datetime import datetime
import requests

# Try to import Flask
try:
    from flask import Flask, Response, request, render_template_string, jsonify
except ImportError:
    print("=" * 70)
    print("Error: Flask is required for the Web version of the chatbot.")
    print("Please install Flask using the following command:")
    print("  pip install flask")
    print("=" * 70)
    sys.exit(1)

# Patch click.echo to prevent "OSError: Windows error 6" in IPython/Jupyter on Windows
try:
    import click
    original_echo = click.echo
    def safe_echo(message=None, file=None, nl=True, err=False, color=None):
        if message is not None:
            if file is None:
                print(str(message), end='\n' if nl else '')
            else:
                try:
                    original_echo(message, file=file, nl=nl, err=err, color=color)
                except Exception:
                    pass
    click.echo = safe_echo
except ImportError:
    pass

from llama_cpp import Llama

# FIX: Allow model path to be overridden via environment variable for portability.
# Set CHATBOT_MODEL_PATH in your environment to avoid editing this file.
MODEL_PATH = os.environ.get(
    "CHATBOT_MODEL_PATH",
    r"C:/Users/kwnga/OneDrive/Desktop/ChatBot/llama.cpp/models/llama/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
)
N_CTX = 12288

# Initialize model
try:
    print("Loading model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False  # Keep terminal clean from cpp logs
    )
    print("Model loaded successfully.\n")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# FIX: Thread lock for llama-cpp-python.
# llama-cpp is NOT thread-safe — concurrent Flask requests will crash without this.
# All inference calls are serialized through this lock.
llm_lock = threading.Lock()

SYSTEM_PROMPT = (
     "You are a helpful, friendly assistant. "
     "Use the provided real-time context if available then answer clearly and honestly."
)

# FIX: Single source of truth for real-time context trigger words.
# Previously duplicated between needs_internet() and the JS frontend.
# Now defined once here and injected into the HTML template via Jinja2.
CONTEXT_TRIGGER_WORDS = ["current", "today", "latest", "price", "now", "weather"]

app = Flask(__name__)

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin", "vs_currencies": "usd"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return f"Bitcoin price (USD): ${r.json()['bitcoin']['usd']}"
    except requests.exceptions.RequestException as e:
        return f"Could not fetch Bitcoin price: {e}"

def needs_internet(text: str) -> bool:
    # FIX: Uses shared CONTEXT_TRIGGER_WORDS constant instead of a separate list
    return any(k in text.lower() for k in CONTEXT_TRIGGER_WORDS)

def get_weather(location: str = "") -> str:
    encoded_location = urllib.parse.quote(location) if location else ""
    url = f"https://wttr.in/{encoded_location}?format=3"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        loc_str = location if location else 'your location'
        return f"Weather for {loc_str}: {r.text.strip()}"
    except requests.exceptions.RequestException as e:
        return f"Could not fetch weather: {e}"

def fetch_realtime_context(user_input: str) -> str:
    context = []
    context.append(f"Current time: {get_current_time()}")

    lower_input = user_input.lower()
    if "bitcoin" in lower_input:
        context.append(get_bitcoin_price())

    if "weather" in lower_input:
        location = ""
        match = re.search(r'\b(?:in|for)\s+([a-zA-Z\s]+)', user_input, re.IGNORECASE)
        if match:
            words = match.group(1).strip().split()
            location = " ".join(words[:2])
        context.append(get_weather(location))

    return "\n".join(context)

def count_tokens(messages: list) -> int:
    text = "".join(m.get('content', '') for m in messages)
    return len(llm.tokenize(text.encode("utf-8")))

def manage_context_window(messages: list, max_ctx: int, buffer: int = 1024) -> list:
    while len(messages) > 3:
        current_tokens = count_tokens(messages)
        if current_tokens + buffer < max_ctx:
            break
        messages.pop(1)
        messages.pop(1)
    return messages

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="A local voice-enabled AI assistant powered by Llama 3.1, with real-time context fetching for weather and Bitcoin prices.">
    <title>Voice AI Assistant</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🎙️</text></svg>">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;700&display=swap');

        :root {
            --bg-color: #080914;
            --container-bg: rgba(15, 17, 34, 0.45);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.35);
            --accent: #d946ef;
            --accent-glow: rgba(217, 70, 239, 0.3);
            --user-bubble: linear-gradient(135deg, #4f46e5, #6366f1);
            --assistant-bubble: rgba(30, 35, 60, 0.65);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.1) transparent;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            position: relative;
        }

        /* Abstract glowing backgrounds */
        body::before {
            content: '';
            position: absolute;
            width: 450px;
            height: 450px;
            background: radial-gradient(circle, var(--primary-glow) 0%, transparent 70%);
            top: 10%;
            left: 15%;
            z-index: 0;
            filter: blur(40px);
            animation: pulse-glow 10s infinite alternate;
        }

        body::after {
            content: '';
            position: absolute;
            width: 450px;
            height: 450px;
            background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
            bottom: 10%;
            right: 15%;
            z-index: 0;
            filter: blur(40px);
            animation: pulse-glow 12s infinite alternate-reverse;
        }

        @keyframes pulse-glow {
            0% { transform: scale(1) translate(0, 0); opacity: 0.7; }
            100% { transform: scale(1.15) translate(30px, 30px); opacity: 1; }
        }

        /* Glassmorphic Container */
        .glass-container {
            width: 90vw;
            max-width: 950px;
            height: 85vh;
            background: var(--container-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            z-index: 1;
            overflow: hidden;
            animation: slide-up 0.6s ease-out;
        }

        @keyframes slide-up {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Header Styling */
        header {
            padding: 20px 30px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(10, 11, 23, 0.3);
        }

        .header-title-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-title-container h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #a5b4fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            background: rgba(34, 197, 94, 0.1);
            color: #4ade80;
            padding: 4px 10px;
            border-radius: 99px;
            border: 1px solid rgba(34, 197, 94, 0.2);
            font-weight: 500;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            background-color: #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 8px #22c55e;
            animation: blink 2s infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
        }

        /* Settings Bar */
        .controls-bar {
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }

        .select-wrapper, .slider-wrapper {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        select {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 6px 12px;
            border-radius: 8px;
            outline: none;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.2s;
            max-width: 150px;
        }

        select:focus {
            border-color: var(--primary);
            background: rgba(255, 255, 255, 0.08);
        }

        /* Toggle Speech button */
        .toggle-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 8px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            position: relative;
        }

        .toggle-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .toggle-btn.active {
            background: rgba(99, 102, 241, 0.15);
            border-color: var(--primary);
            color: #a5b4fc;
        }

        /* Range input styling */
        input[type="range"] {
            -webkit-appearance: none;
            width: 80px;
            height: 4px;
            border-radius: 2px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            transition: all 0.1s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            background: #818cf8;
        }

        /* Chat Messages Area */
        .messages-container {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
            background: rgba(0, 0, 0, 0.1);
        }

        .message-row {
            display: flex;
            width: 100%;
            opacity: 0;
            transform: translateY(12px);
            animation: message-appear 0.3s forwards cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes message-appear {
            to { opacity: 1; transform: translateY(0); }
        }

        .message-row.user {
            justify-content: flex-end;
        }

        .message-row.assistant {
            justify-content: flex-start;
        }

        .bubble {
            max-width: 75%;
            padding: 14px 20px;
            font-size: 0.95rem;
            line-height: 1.5;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            position: relative;
        }

        .user .bubble {
            background: var(--user-bubble);
            color: #ffffff;
            border-radius: 20px 20px 4px 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .assistant .bubble {
            background: var(--assistant-bubble);
            color: var(--text-main);
            border-radius: 20px 20px 20px 4px;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(5px);
        }

        .bubble p {
            margin-bottom: 8px;
        }
        .bubble p:last-child {
            margin-bottom: 0;
        }

        .context-fetch-indicator {
            font-size: 0.78rem;
            color: var(--accent);
            margin-top: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 500;
        }

        /* Typing & Streaming indicators */
        .typing-dots {
            display: flex;
            gap: 4px;
            align-items: center;
            height: 18px;
            padding: 0 4px;
        }

        .dot {
            width: 6px;
            height: 6px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: wave-dots 1.4s infinite ease-in-out;
        }

        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes wave-dots {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-4px); }
        }

        /* Speak Wave Animation */
        .speak-wave {
            display: flex;
            align-items: flex-end;
            gap: 2px;
            height: 14px;
            margin-left: auto;
            margin-top: 4px;
            opacity: 0.65;
            width: fit-content;
        }

        .wave-bar {
            width: 2px;
            background: var(--accent);
            border-radius: 1px;
            animation: bounce-wave 0.5s infinite ease-in-out alternate;
        }
        .wave-bar:nth-child(1) { height: 6px; animation-duration: 0.4s; }
        .wave-bar:nth-child(2) { height: 12px; animation-duration: 0.65s; }
        .wave-bar:nth-child(3) { height: 8px; animation-duration: 0.5s; }
        .wave-bar:nth-child(4) { height: 4px; animation-duration: 0.35s; }

        @keyframes bounce-wave {
            from { height: 2px; }
            to { }
        }

        /* Chat Input Area */
        .input-container {
            padding: 20px 30px;
            border-top: 1px solid var(--border-color);
            background: rgba(10, 11, 23, 0.4);
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .chat-input-wrapper {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
        }

        .chat-input {
            width: 100%;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 14px 20px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s;
            font-family: inherit;
        }

        .chat-input:focus {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(99, 102, 241, 0.6);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.15);
        }

        /* Audio feedback layout inside the input */
        .listening-visualizer {
            position: absolute;
            right: 15px;
            display: none;
            align-items: center;
            gap: 8px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 500;
            animation: pulse-red 1.5s infinite;
        }

        @keyframes pulse-red {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            50% { box-shadow: 0 0 8px 2px rgba(239, 68, 68, 0.4); }
        }

        .mic-active-dot {
            width: 6px;
            height: 6px;
            background: #ef4444;
            border-radius: 50%;
            animation: blink 1s infinite;
        }

        /* Voice/Send Buttons */
        .circle-btn {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            border: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            outline: none;
        }

        .circle-btn:hover {
            transform: scale(1.05);
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .mic-btn.listening {
            background: linear-gradient(135deg, #ef4444, #ec4899);
            border-color: transparent;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
            animation: mic-bounce 0.8s infinite alternate;
        }

        @keyframes mic-bounce {
            from { transform: scale(1.03); }
            to { transform: scale(1.12); }
        }

        .send-btn {
            background: var(--primary);
            border-color: transparent;
            color: white;
        }

        .send-btn:hover {
            background: #4f46e5;
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: scale(1) !important;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
        }

        /* SVG Icon styling */
        .circle-btn svg, .toggle-btn svg {
            width: 20px;
            height: 20px;
            fill: currentColor;
        }

        /* Layout adjustment for small screens */
        @media (max-width: 640px) {
            .glass-container {
                height: 98vh;
                width: 98vw;
                border-radius: 12px;
            }
            header {
                flex-direction: column;
                gap: 12px;
                padding: 15px;
                align-items: stretch;
            }
            .controls-bar {
                justify-content: space-between;
            }
            .input-container {
                padding: 15px;
            }
        }

        /* Toast Notification Banner */
        .toast-notification {
            position: absolute;
            top: 85px;
            left: 50%;
            transform: translateX(-50%) translateY(-20px);
            background: rgba(15, 17, 34, 0.9);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
            padding: 10px 20px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 500;
            z-index: 999;
            display: flex;
            align-items: center;
            gap: 8px;
            opacity: 0;
            pointer-events: none;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }

        .toast-notification.show {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
            pointer-events: auto;
        }

        .toast-notification.success {
            border-color: rgba(34, 197, 94, 0.3);
            color: #86efac;
        }
    </style>
</head>
<body>

    <!-- Toast Notification Banner -->
    <div id="toastNotification" class="toast-notification">
        <span id="toastIcon">⚠️</span>
        <span id="toastMessage">Notification message</span>
    </div>

    <div class="glass-container">
        <!-- Header Section -->
        <header>
            <div class="header-title-container">
                <h1>Voice AI Assistant</h1>
                <div class="status-badge">
                    <span class="status-dot"></span>
                    <span>Llama 3.1 Ready</span>
                </div>
            </div>
            
            <div class="controls-bar">
                <!-- Select Voice Dropdown -->
                <div class="select-wrapper">
                    <span>Voice:</span>
                    <select id="voiceSelect" style="max-width: 170px;">
                        <option value="">Default Browser Voice</option>
                    </select>
                    <button class="toggle-btn" id="previewVoiceBtn" title="Preview Voice" style="padding: 5px 8px; font-size: 0.85rem;">
                        🔊
                    </button>
                </div>

                <!-- English Filter Checkbox -->
                <label style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: var(--text-muted); cursor: pointer; user-select: none;">
                    <input type="checkbox" id="englishOnlyCheckbox" checked style="cursor: pointer; accent-color: var(--primary);">
                    <span>English Only</span>
                </label>

                <!-- Offline Only Filter Checkbox -->
                <label style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: var(--text-muted); cursor: pointer; user-select: none;" title="Show only offline voices that work without internet connection on localhost">
                    <input type="checkbox" id="offlineOnlyCheckbox" checked style="cursor: pointer; accent-color: var(--primary);">
                    <span>Offline Only</span>
                </label>
                
                <!-- Speed controls -->
                <div class="slider-wrapper">
                    <span>Speed:</span>
                    <input type="range" id="rateRange" min="0.5" max="2.0" step="0.1" value="1.0">
                    <span id="speedVal" style="min-width: 28px; text-align: right;">1.0x</span>
                </div>

                <!-- Speak Toggle Button (Mute/Unmute output) -->
                <button class="toggle-btn active" id="speechToggle" title="Toggle Voice Response">
                    <svg viewBox="0 0 24 24" id="volumeIcon">
                        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
                    </svg>
                </button>
            </div>
        </header>

        <!-- Message logs -->
        <div class="messages-container" id="messages">
            <div class="message-row assistant">
                <div class="bubble">
                    <p>Hello! I am your voice-enabled AI assistant. How can I help you today? You can type your request or click the microphone to speak.</p>
                </div>
            </div>
        </div>

        <!-- Chat controls input -->
        <div class="input-container">
            <!-- Voice Input toggle button -->
            <button class="circle-btn mic-btn" id="micBtn" title="Speak to assistant">
                <svg viewBox="0 0 24 24" id="micIcon">
                    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/>
                </svg>
            </button>

            <!-- Text entry panel -->
            <div class="chat-input-wrapper">
                <input type="text" class="chat-input" id="userInput" placeholder="Ask something..." autocomplete="off">
                <div class="listening-visualizer" id="listeningIndicator">
                    <span class="mic-active-dot"></span>
                    <span>Listening...</span>
                </div>
            </div>

            <!-- Send button -->
            <button class="circle-btn send-btn" id="sendBtn" title="Send message" disabled>
                <svg viewBox="0 0 24 24">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
            </button>
        </div>
    </div>

    <script>
        const messagesContainer = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const micBtn = document.getElementById('micBtn');
        const voiceSelect = document.getElementById('voiceSelect');
        const rateRange = document.getElementById('rateRange');
        const speedVal = document.getElementById('speedVal');
        const previewVoiceBtn = document.getElementById('previewVoiceBtn');
        const englishOnlyCheckbox = document.getElementById('englishOnlyCheckbox');
        const offlineOnlyCheckbox = document.getElementById('offlineOnlyCheckbox');
        const speechToggle = document.getElementById('speechToggle');
        const listeningIndicator = document.getElementById('listeningIndicator');
        const toastNotification = document.getElementById('toastNotification');
        const toastMessage = document.getElementById('toastMessage');
        const toastIcon = document.getElementById('toastIcon');

        // Toast Helper
        let toastTimeout = null;
        function showToast(message, isError = true, duration = 4500) {
            if (!toastNotification) return;
            if (toastTimeout) clearTimeout(toastTimeout);
            
            toastMessage.textContent = message;
            toastIcon.textContent = isError ? '⚠️' : 'ℹ️';
            
            toastNotification.className = 'toast-notification';
            if (isError) {
                toastNotification.classList.add('error');
            } else {
                toastNotification.classList.add('success');
            }
            
            toastNotification.classList.add('show');
            
            toastTimeout = setTimeout(() => {
                toastNotification.classList.remove('show');
            }, duration);
        }

        // FIX: Trigger words injected from Python backend — single source of truth.
        // Avoids duplicating the list between needs_internet() and the JS frontend.
        const contextTriggerWords = {{ trigger_words | safe }};

        // App state
        let chatHistory = [];
        let isGenerating = false;
        let isListening = false;
        let ttsEnabled = true;
        let synthesisQueue = [];
        let activeUtterances = []; // Store active utterance references to prevent Chrome GC bug
        let isSpeaking = false;
        let currentUtterance = null;
        let speechBuffer = '';
        let silenceTimer = null;
        let finalTranscript = '';
        let manuallyStopped = true;  // tracks whether the user intended to stop mic
        let ttsKeepaliveTimer = null; // Chrome TTS stall prevention
        // FIX: Implemented isGenerating timeout guard (was mentioned in comment but never wired up).
        // If the SSE stream silently drops mid-response, the UI would be permanently locked.
        let generatingTimeoutId = null;
        const GENERATING_TIMEOUT_MS = 90000; // 90 seconds

        // Initialize Web Speech Synthesis
        const synth = window.speechSynthesis;
        
        // Populate browser voice dropdown list with filtering, sorting, and persistence
        function loadVoices() {
            if (!synth) return;
            
            // Capture the current selection or saved preference to maintain it
            const currentlySelected = voiceSelect.value || localStorage.getItem('preferredVoiceName') || '';
            const filterEnglish = englishOnlyCheckbox.checked;
            const filterOffline = offlineOnlyCheckbox.checked;
            
            let voices = synth.getVoices();
            
            // Filter out non-English languages if "English Only" is checked
            if (filterEnglish) {
                voices = voices.filter(voice => voice.lang.toLowerCase().startsWith('en'));
            }
            
            // Filter out cloud/online voices if "Offline Only" is checked
            if (filterOffline) {
                voices = voices.filter(voice => voice.localService === true);
            }
            
            // Sort voices: offline/local voices first
            voices.sort((a, b) => {
                if (a.localService && !b.localService) return -1;
                if (!a.localService && b.localService) return 1;
                return a.name.localeCompare(b.name);
            });

            voiceSelect.innerHTML = '<option value="">Default System Voice</option>';
            
            voices.forEach(voice => {
                const option = document.createElement('option');
                const isLocal = voice.localService === true;
                option.textContent = `${voice.name} (${voice.lang}) ${isLocal ? '🔒 [Local]' : '☁️ [Cloud]'}`;
                option.value = voice.name;
                
                // Prioritize keeping the current/saved selection
                if (voice.name === currentlySelected) {
                    option.selected = true;
                } else if (!currentlySelected && voice.default) {
                    option.selected = true;
                }
                
                voiceSelect.appendChild(option);
            });
        }
        
        // Browser voices are loaded asynchronously
        if (synth) {
            loadVoices();
            if (synth.onvoiceschanged !== undefined) {
                synth.onvoiceschanged = () => {
                    console.log("[TTS] onvoiceschanged triggered, reloading voices...");
                    loadVoices();
                };
            }
        }

        // Save voice preference when manually changed
        voiceSelect.addEventListener('change', () => {
            console.log("[TTS] Voice changed to:", voiceSelect.value);
            localStorage.setItem('preferredVoiceName', voiceSelect.value);
        });

        // Load checkbox preferences from localStorage
        const savedEnglishOnly = localStorage.getItem('preferredEnglishOnly');
        if (savedEnglishOnly !== null) {
            englishOnlyCheckbox.checked = savedEnglishOnly === 'true';
        }
        
        const savedOfflineOnly = localStorage.getItem('preferredOfflineOnly');
        if (savedOfflineOnly !== null) {
            offlineOnlyCheckbox.checked = savedOfflineOnly === 'true';
        }

        // Trigger voice reload when English-only checkbox changes
        englishOnlyCheckbox.addEventListener('change', () => {
            localStorage.setItem('preferredEnglishOnly', englishOnlyCheckbox.checked);
            loadVoices();
        });

        // Trigger voice reload when Offline-only checkbox changes
        offlineOnlyCheckbox.addEventListener('change', () => {
            localStorage.setItem('preferredOfflineOnly', offlineOnlyCheckbox.checked);
            loadVoices();
        });

        // Persist and load speed settings
        const savedSpeed = localStorage.getItem('preferredSpeechSpeed');
        if (savedSpeed) {
            rateRange.value = savedSpeed;
            speedVal.textContent = parseFloat(savedSpeed).toFixed(1) + 'x';
        }

        rateRange.addEventListener('input', () => {
            const currentSpeed = parseFloat(rateRange.value).toFixed(1);
            speedVal.textContent = currentSpeed + 'x';
            localStorage.setItem('preferredSpeechSpeed', rateRange.value);
        });

        // Interactive voice preview functionality
        previewVoiceBtn.addEventListener('click', () => {
            if (!synth) return;
            
            // If the preview utterance is already speaking, stop all speech
            if (synth.speaking && previewVoiceBtn.textContent === '⏹') {
                stopSpeaking();
                return;
            }
            
            stopSpeaking(); // Mute/stop any current chatbot speech
            
            const previewText = "Hello! This is a preview of my voice.";
            const previewUtterance = new SpeechSynthesisUtterance(previewText);
            
            // Set the voice chosen
            const selectedVoiceName = voiceSelect.value;
            let matchedVoice = null;
            if (selectedVoiceName) {
                const voices = synth.getVoices();
                matchedVoice = voices.find(v => v.name === selectedVoiceName);
                if (matchedVoice) {
                    previewUtterance.voice = matchedVoice;
                    previewUtterance.lang = matchedVoice.lang; // Sync language
                }
            }
            
            previewUtterance.rate = parseFloat(rateRange.value) || 1.0;
            
            // Keep reference to prevent GC
            activeUtterances.push(previewUtterance);
            
            previewUtterance.onstart = () => {
                previewVoiceBtn.textContent = '⏹';
                previewVoiceBtn.title = "Stop preview";
                isSpeaking = true;
            };
            
            previewUtterance.onend = () => {
                activeUtterances = activeUtterances.filter(u => u !== previewUtterance);
                previewVoiceBtn.textContent = '🔊';
                previewVoiceBtn.title = "Preview Voice";
                isSpeaking = false;
            };
            
            previewUtterance.onerror = (e) => {
                console.error("[TTS Preview] Speech error:", e.error);
                activeUtterances = activeUtterances.filter(u => u !== previewUtterance);
                
                // Fallback logic for preview
                const isCloudVoice = matchedVoice && (matchedVoice.localService === false || matchedVoice.name.includes('Google') || matchedVoice.name.includes('Natural'));
                if (isCloudVoice && (e.error === 'network' || e.error === 'voice-unavailable' || e.error === 'synthesis-failed')) {
                    showToast(`Voice preview failed (${e.error}). Trying offline local voice...`, true);
                    
                    const fallbackUtterance = new SpeechSynthesisUtterance(previewText);
                    fallbackUtterance.rate = previewUtterance.rate;
                    
                    const voices = synth.getVoices();
                    const localVoice = voices.find(v => v.lang.toLowerCase().startsWith('en') && v.localService === true);
                    if (localVoice) {
                        fallbackUtterance.voice = localVoice;
                        fallbackUtterance.lang = localVoice.lang;
                    }
                    
                    activeUtterances.push(fallbackUtterance);
                    
                    fallbackUtterance.onstart = () => {
                        previewVoiceBtn.textContent = '⏹';
                        isSpeaking = true;
                    };
                    fallbackUtterance.onend = () => {
                        activeUtterances = activeUtterances.filter(u => u !== fallbackUtterance);
                        previewVoiceBtn.textContent = '🔊';
                        isSpeaking = false;
                    };
                    fallbackUtterance.onerror = () => {
                        activeUtterances = activeUtterances.filter(u => u !== fallbackUtterance);
                        previewVoiceBtn.textContent = '🔊';
                        isSpeaking = false;
                    };
                    
                    synth.speak(fallbackUtterance);
                } else {
                    previewVoiceBtn.textContent = '🔊';
                    previewVoiceBtn.title = "Preview Voice";
                    isSpeaking = false;
                }
            };
            
            synth.speak(previewUtterance);
        });

        // Speech Toggle Mode
        speechToggle.addEventListener('click', () => {
            ttsEnabled = !ttsEnabled;
            speechToggle.classList.toggle('active', ttsEnabled);
            if (!ttsEnabled) {
                stopSpeaking();
            } else {
                speakText("Voice response enabled.");
            }
        });

        // Enable/Disable send button based on text
        userInput.addEventListener('input', () => {
            sendBtn.disabled = userInput.value.trim().length === 0 || isGenerating;
        });

        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !sendBtn.disabled) {
                sendMessage();
            }
        });

        // Submit message when clicking the Send button
        sendBtn.addEventListener('click', () => {
            if (!sendBtn.disabled) {
                sendMessage();
            }
        });

        // Text to Speech Queue Management
        function speakText(text) {
            if (!synth || !ttsEnabled) return;

            // FIX: Improved markdown stripping for cleaner TTS audio.
            // v6 only handled ### headings; v7 handles H1-H6, bold/italic markers,
            // underscores, inline code, and converts [text](url) links to spoken text only.
            let cleanText = text
                .replace(/\\*+/g, '')                              // Remove * and ** (bold/italic)
                .replace(/_{1,2}([^_]+)_{1,2}/g, '$1')            // Un-wrap _italic_ and __bold__
                .replace(/#{1,6}\\s*/g, '')                        // Remove # through ###### headings
                .replace(/`+/g, '')                                // Remove inline code backticks
                .replace(/\\[([^\\]]+)\\]\\([^)]+\\)/g, '$1')     // [text](url) → text only
                .trim();

            if (!cleanText) return;

            const utterance = new SpeechSynthesisUtterance(cleanText);
            
            // Keep a reference to prevent garbage collection in Chromium browsers
            activeUtterances.push(utterance);
            
            // Set user preferences
            const selectedVoiceName = voiceSelect.value;
            let matchedVoice = null;
            if (selectedVoiceName) {
                const voices = synth.getVoices();
                matchedVoice = voices.find(v => v.name === selectedVoiceName);
                if (matchedVoice) {
                    utterance.voice = matchedVoice;
                    utterance.lang = matchedVoice.lang; // Sync language code
                }
            }
            
            utterance.rate = parseFloat(rateRange.value) || 1.0;
            
            utterance.onend = () => {
                // Remove reference
                activeUtterances = activeUtterances.filter(u => u !== utterance);
                isSpeaking = false;
                processSpeechQueue();
            };
            
            utterance.onerror = (e) => {
                console.error("Speech Synthesis Error: ", e.error);
                
                // If cloud/network voice fails, attempt local fallback
                const isCloudVoice = matchedVoice && (matchedVoice.localService === false || matchedVoice.name.includes('Google') || matchedVoice.name.includes('Natural'));
                if (isCloudVoice && (e.error === 'network' || e.error === 'voice-unavailable' || e.error === 'synthesis-failed')) {
                    showToast(`Voice "${matchedVoice.name}" failed (${e.error}). Switching to offline local voice...`, true);
                    
                    const fallbackUtterance = new SpeechSynthesisUtterance(cleanText);
                    fallbackUtterance.rate = utterance.rate;
                    
                    const voices = synth.getVoices();
                    const localVoice = voices.find(v => v.lang.toLowerCase().startsWith('en') && v.localService === true);
                    if (localVoice) {
                        fallbackUtterance.voice = localVoice;
                        fallbackUtterance.lang = localVoice.lang;
                    }
                    
                    activeUtterances.push(fallbackUtterance);
                    
                    fallbackUtterance.onend = () => {
                        activeUtterances = activeUtterances.filter(u => u !== fallbackUtterance);
                        isSpeaking = false;
                        processSpeechQueue();
                    };
                    fallbackUtterance.onerror = (err) => {
                        activeUtterances = activeUtterances.filter(u => u !== fallbackUtterance);
                        isSpeaking = false;
                        processSpeechQueue();
                    };
                    
                    synthesisQueue.unshift(fallbackUtterance); // Insert to front of queue
                }
                
                activeUtterances = activeUtterances.filter(u => u !== utterance);
                isSpeaking = false;
                processSpeechQueue();
            };

            synthesisQueue.push(utterance);
            if (!isSpeaking) {
                processSpeechQueue();
            }
        }

        function processSpeechQueue() {
            if (synthesisQueue.length === 0) {
                isSpeaking = false;
                if (ttsKeepaliveTimer) { clearInterval(ttsKeepaliveTimer); ttsKeepaliveTimer = null; }
                document.querySelectorAll('.speak-wave').forEach(w => w.remove());
                return;
            }
            
            isSpeaking = true;
            currentUtterance = synthesisQueue.shift();
            
            // Show dynamic voice visualization next to the active message bubble
            const assistantRows = document.querySelectorAll('.message-row.assistant');
            if (assistantRows.length > 0) {
                const latestBubble = assistantRows[assistantRows.length - 1].querySelector('.bubble');
                if (latestBubble && !latestBubble.querySelector('.speak-wave')) {
                    const waveEl = document.createElement('div');
                    waveEl.className = 'speak-wave';
                    waveEl.innerHTML = `
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                    `;
                    latestBubble.appendChild(waveEl);
                }
            }
            
            synth.speak(currentUtterance);

            // Chrome bug: SpeechSynthesis silently stalls/pauses after ~15s.
            // Calling resume() on a regular interval prevents this.
            if (ttsKeepaliveTimer) clearInterval(ttsKeepaliveTimer);
            ttsKeepaliveTimer = setInterval(() => {
                if (synth && synth.speaking) {
                    synth.pause();
                    synth.resume();
                } else if (!synth.speaking && synthesisQueue.length === 0) {
                    clearInterval(ttsKeepaliveTimer);
                    ttsKeepaliveTimer = null;
                }
            }, 10000);
        }

        function stopSpeaking() {
            if (synth) {
                synth.cancel();
            }
            if (ttsKeepaliveTimer) { clearInterval(ttsKeepaliveTimer); ttsKeepaliveTimer = null; }
            activeUtterances = [];
            synthesisQueue = [];
            isSpeaking = false;
            document.querySelectorAll('.speak-wave').forEach(w => w.remove());
            if (previewVoiceBtn) {
                previewVoiceBtn.textContent = '🔊';
                previewVoiceBtn.title = "Preview Voice";
            }
        }

        // Browser Web Speech Recognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let recognition = null;

        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = true; // Keep listening to allow user pauses/breaths
            recognition.interimResults = true; // Show words on screen while user is speaking
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                isListening = true;
                micBtn.classList.add('listening');
                userInput.placeholder = "";
                listeningIndicator.style.display = 'flex';
                stopSpeaking(); // Mute bot if it is speaking when user tries to talk
                if (silenceTimer) clearTimeout(silenceTimer);
                
                // Only clear transcripts if this is a brand new intended session (not an auto-restart)
                if (manuallyStopped) {
                    finalTranscript = '';
                    userInput.value = '';
                }

                // CRITICAL FIX: Reset manuallyStopped to false so that if Chrome
                // ends the session unexpectedly the auto-restart logic will fire.
                manuallyStopped = false;
            };

            recognition.onresult = (event) => {
                let interimTranscript = '';
                
                // Reset silence timer on any detected voice input activity
                if (silenceTimer) clearTimeout(silenceTimer);
                
                // Reconstruct full transcript from results
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    const segment = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += segment + ' ';
                    } else {
                        interimTranscript += segment;
                    }
                }
                
                // Display text typing itself in real-time
                const currentText = finalTranscript + interimTranscript;
                userInput.value = currentText.trim();
                
                if (userInput.value.trim().length > 0) {
                    sendBtn.disabled = false;
                }
                
                // Start a 2.5 second silence timer.
                // If the user goes silent for 2.5 seconds, we auto-stop recognition and submit.
                silenceTimer = setTimeout(() => {
                    if (isListening) {
                        console.log("[Speech] Silence timeout reached. Auto-submitting prompt.");
                        manuallyStopped = true; // We intend to stop and send
                        try {
                            recognition.stop();
                        } catch (e) {
                            console.error("Error stopping recognition on timeout:", e);
                        }
                    }
                }, 2500);
            };

            recognition.onerror = (event) => {
                console.error("[Speech] Recognition Error:", event.error);
                if (silenceTimer) clearTimeout(silenceTimer);

                // On network errors, attempt auto-restart if the user hasn't manually stopped
                const retriableErrors = ['network', 'service-not-allowed', 'audio-capture'];
                if (!manuallyStopped && isListening && retriableErrors.includes(event.error)) {
                    console.log(`[Speech] Retriable error (${event.error}). Attempting restart...`);
                    setTimeout(() => {
                        if (isListening && !manuallyStopped) {
                            try {
                                recognition.start();
                            } catch (e) {
                                console.error("Error restarting after error:", e);
                                resetMicState();
                            }
                        }
                    }, 300);
                } else {
                    resetMicState();
                }
            };

            recognition.onend = () => {
                if (silenceTimer) clearTimeout(silenceTimer);
                
                if (!manuallyStopped && isListening) {
                    // Browser ended session unexpectedly (e.g. Chrome's server-side silence limit)
                    // Auto-restart to keep microphone capture active and accumulate transcript
                    console.log("[Speech] Browser ended session unexpectedly. Auto-restarting...");
                    setTimeout(() => {
                        try {
                            recognition.start();
                        } catch (e) {
                            console.error("Error auto-restarting recognition:", e);
                            resetMicState();
                        }
                    }, 100);
                } else {
                    resetMicState();
                    
                    // Automatically send message on microphone turn end if content exists
                    const speechContent = userInput.value.trim();
                    if (speechContent.length > 0) {
                        sendMessage();
                    }
                }
            };
        } else {
            micBtn.style.display = 'none'; // Hide mic if not supported (e.g., Firefox)
            console.log("Speech recognition not supported in this browser.");
        }

        function resetMicState() {
            isListening = false;
            micBtn.classList.remove('listening');
            userInput.placeholder = "Ask something...";
            listeningIndicator.style.display = 'none';
        }

        micBtn.addEventListener('click', () => {
            if (!recognition) return;
            if (isListening) {
                // User is manually stopping the mic - set flag BEFORE calling stop()
                // so that onend knows not to auto-restart
                manuallyStopped = true;
                isListening = false; // Prevent race where onend fires before flag is read
                try {
                    recognition.stop();
                } catch (e) {
                    console.error("Error stopping recognition:", e);
                    resetMicState();
                }
            } else {
                // Starting a fresh new session
                manuallyStopped = true; // stays true until onstart resets it, ensuring fresh transcripts
                stopSpeaking(); // Cancel any synthesis to release audio device
                
                // Allow a small delay for the browser to close the output audio stream
                setTimeout(() => {
                    try {
                        recognition.start();
                    } catch (e) {
                        console.error("Error starting recognition:", e);
                        resetMicState();
                    }
                }, 200);
            }
        });

        // FIX: XSS protection — use textContent instead of innerHTML for user-supplied content.
        // Previously: bubble.innerHTML = `<p>${content}</p>` allowed HTML injection.
        // Now all dynamic text is set via textContent which is always treated as plain text.
        function appendMessageRow(role, content = '', isTypingPlaceholder = false) {
            const row = document.createElement('div');
            row.className = `message-row ${role}`;
            
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            
            if (isTypingPlaceholder) {
                bubble.innerHTML = `
                    <div class="typing-dots">
                        <span class="dot"></span>
                        <span class="dot"></span>
                        <span class="dot"></span>
                    </div>
                `;
            } else if (content) {
                const p = document.createElement('p');
                p.textContent = content;  // Safe: treats content as plain text, not HTML
                bubble.appendChild(p);
            }
            
            row.appendChild(bubble);
            messagesContainer.appendChild(row);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            return row;
        }

        // Submits the query to the server
        async function sendMessage() {
            const messageText = userInput.value.trim();
            if (!messageText) return;

            if (isGenerating) {
                console.warn("[Chat] sendMessage called while isGenerating=true. Ignoring.");
                return;
            }

            // Stop recognition and prevent auto-restart
            if (isListening && recognition) {
                manuallyStopped = true;
                isListening = false;
                try {
                    recognition.stop();
                } catch (e) {
                    console.error("Error stopping recognition on send:", e);
                }
            }

            // Stop any ongoing narration
            stopSpeaking();

            // Clear input and reset transcript accumulator for next voice prompt
            finalTranscript = '';
            userInput.value = '';
            sendBtn.disabled = true;

            // Render user bubble
            appendMessageRow('user', messageText);

            // Check if any context trigger words are present (uses Python-injected list)
            const lowerText = messageText.toLowerCase();
            const hasContextHint = contextTriggerWords.some(w => lowerText.includes(w));

            // Create Assistant Typing Placeholder Bubble
            const assistantRow = appendMessageRow('assistant', '', true);
            const assistantBubble = assistantRow.querySelector('.bubble');
            
            if (hasContextHint) {
                const hint = document.createElement('div');
                hint.className = 'context-fetch-indicator';
                hint.innerHTML = `
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                    </svg>
                    <span>Fetching real-time updates...</span>
                `;
                assistantBubble.appendChild(hint);
            }

            isGenerating = true;
            speechBuffer = '';

            // FIX: Actually implement the 90s isGenerating timeout guard.
            // Previously commented about it but never wired it up.
            // If the SSE stream silently drops, this releases the UI lock automatically.
            generatingTimeoutId = setTimeout(() => {
                if (isGenerating) {
                    console.warn('[Chat] isGenerating timeout (90s) reached. Releasing UI lock.');
                    isGenerating = false;
                    sendBtn.disabled = userInput.value.trim().length === 0;
                }
            }, GENERATING_TIMEOUT_MS);
            
            // Build the body payload
            const payload = {
                message: messageText,
                history: chatHistory
            };

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
                }

                // Remove typing dots placeholder
                const dotsEl = assistantBubble.querySelector('.typing-dots');
                if (dotsEl) dotsEl.remove();

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let botReply = '';
                
                // Content paragraph element — text is appended via textContent (XSS-safe)
                const responseTextEl = document.createElement('p');
                assistantBubble.insertBefore(responseTextEl, assistantBubble.firstChild);

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.substring(6));

                                // FIX: Handle server-side errors streamed back from the LLM generator.
                                // Previously these were silently swallowed; now displayed to the user.
                                if (data.error) {
                                    console.error('[Chat] Server-side LLM error:', data.error);
                                    responseTextEl.textContent = `⚠ Server error: ${data.error}`;
                                    responseTextEl.style.color = '#ef4444';
                                    break;
                                }

                                if (data.content) {
                                    const rawToken = data.content;
                                    // textContent += is XSS-safe for streaming token append
                                    responseTextEl.textContent += rawToken;
                                    botReply += rawToken;
                                    
                                    // Process speech synthesis chunk buffer
                                    if (ttsEnabled) {
                                        speechBuffer += rawToken;
                                        // Match sentence endings for real-time TTS chunking.
                                        // Avoids splitting on abbreviations perfectly but handles
                                        // standard sentence endings well for live narration.
                                        const sentenceMatch = speechBuffer.match(/([^.?!\\n\\r]+[.?!\\n])/);
                                        if (sentenceMatch) {
                                            const completedSentence = sentenceMatch[0].trim();
                                            speechBuffer = speechBuffer.substring(sentenceMatch[0].length);
                                            speakText(completedSentence);
                                        }
                                    }
                                }
                            } catch (err) {
                                // Ignore JSON parsing errors on partial/incomplete SSE chunks
                            }
                        }
                    }
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }

                // Speak any remaining leftover text in buffer
                if (ttsEnabled && speechBuffer.trim()) {
                    speakText(speechBuffer.trim());
                }

                // Save final assistant reply to chat history
                chatHistory.push({ role: "user", content: messageText });
                chatHistory.push({ role: "assistant", content: botReply });

            } catch (err) {
                console.error("Fetch API error:", err);
                const dotsEl = assistantBubble.querySelector('.typing-dots');
                if (dotsEl) dotsEl.remove();
                
                const errText = document.createElement('p');
                errText.style.color = '#ef4444';
                errText.textContent = "Error: Failed to connect to local Llama backend. Verify the terminal logs.";
                assistantBubble.insertBefore(errText, assistantBubble.firstChild);
            } finally {
                isGenerating = false;
                // FIX: Always clear the timeout guard when the request finishes normally
                if (generatingTimeoutId) {
                    clearTimeout(generatingTimeoutId);
                    generatingTimeoutId = null;
                }
                sendBtn.disabled = userInput.value.trim().length === 0;
                // Remove the fetching context indicator
                const hintEl = assistantBubble.querySelector('.context-fetch-indicator');
                if (hintEl) hintEl.remove();
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    # FIX: Inject CONTEXT_TRIGGER_WORDS from Python into the JS template via Jinja2.
    # This eliminates the duplicated list that previously existed independently in both
    # needs_internet() and the frontend JS contextTriggerWords array.
    return render_template_string(HTML_TEMPLATE, trigger_words=json.dumps(CONTEXT_TRIGGER_WORDS))

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    user_input = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Build system/history format
    # Start with System Prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Append past history context
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Fetch context if query warrants it
    realtime_context = ""
    if needs_internet(user_input):
        print(f"[Web Server] Fetching real-time context for: '{user_input}'")
        realtime_context = fetch_realtime_context(user_input)

    user_content = user_input
    if realtime_context:
        user_content = f"Context (real-time data):\n{realtime_context}\n\nUser Question: {user_input}"

    messages.append({"role": "user", "content": user_content})

    # Manage token window
    messages = manage_context_window(messages, N_CTX, buffer=2048)

    def generate_reply():
        # FIX: Acquire the thread lock before any llama-cpp inference call.
        # llama-cpp-python is NOT thread-safe. Without this, concurrent requests
        # (e.g., two browser tabs) will corrupt internal model state and crash.
        with llm_lock:
            try:
                stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=2048,
                    stream=True
                )
                for chunk in stream:
                    if "content" in chunk["choices"][0]["delta"]:
                        text = chunk["choices"][0]["delta"]["content"]
                        yield f"data: {json.dumps({'content': text})}\n\n"
            except Exception as e:
                print(f"[Web Server] Llama inference error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # Return stream response
    return Response(generate_reply(), mimetype="text/event-stream")

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Web GUI Voice Assistant (v7) at: http://127.0.0.1:5000")
    print("=" * 60)
    # Turn off Flask terminal debugger/reloader logs to prevent double loading the model
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

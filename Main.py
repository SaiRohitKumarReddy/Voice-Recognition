import streamlit as st
import requests
import json
import os
import tempfile
import base64
import uuid
import logging
import subprocess  # Required for pip install
import sys
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="🎙️ Voice AI Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Auto-install missing packages ---
def install_and_import(package, package_name=None):
    package_name = package_name or package
    try:
        __import__(package_name)
        return True
    except ImportError:
        try:
            # Update pip first
            st.info("🔧 Updating pip to latest version...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ])

            # Install package
            st.info(f"📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-cache-dir", package
            ])
            return True
        except Exception as e:
            st.error(f"❌ Failed to install `{package}`: {str(e)}")
            return False

# Required packages: (pip_name, import_name)
required_packages = {
    "groq": "groq",
    "transformers": "transformers",
    "torch": "torch",
    "PySoundFile": "soundfile",  
    "librosa": "librosa",
    "pydub": "pydub",
    "audio-recorder-streamlit": "audio_recorder_streamlit",
    "edge-tts": "edge_tts"
}

# Install each package if not already present
for pkg, imp in required_packages.items():
    if pkg not in sys.modules:
        success = install_and_import(pkg, imp)
        if not success:
            st.error(f"🚨 Critical failure: Could not install `{pkg}`. App may not work.")
            st.stop()

# --- Now Import All Packages ---
try:
    import soundfile as sf
    SNDFILE_AVAILABLE = True
except ImportError:
    SNDFILE_AVAILABLE = False
    st.error("❌ Failed to import soundfile")
    st.stop()

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("❌ Failed to import torch")
    st.stop()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.error("❌ Failed to import groq")
    st.stop()

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("❌ Failed to import transformers")
    st.stop()

try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False
    st.error("❌ Failed to import audio_recorder_streamlit")
    st.stop()

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    st.warning("⚠️ TTS (edge-tts) not available")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    st.warning("⚠️ librosa not available")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    st.warning("⚠️ pydub not available")

# --- Constants ---
MAX_TTS_LENGTH = 800
MAX_TOKENS_DETAIL = 1000
MAX_TOKENS_SIMPLE = 300
MAX_CACHE_SIZE = 5
API_TIMEOUT = 30

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Session State Management ---
def get_session_var(key, default=None):
    return st.session_state.get(key, default)

def set_session_var(key, value):
    st.session_state[key] = value

def init_session_state():
    defaults = {
        'conversation_history': [],
        'total_interactions': 0,
        'api_initialized': False,
        'groq_client': None,
        'audio_cache': {},
        'voice_options': {'en': 'en-US-AriaNeural'},
        'wav2vec_processor': None,
        'wav2vec_model': None
    }
    for key, default_value in defaults.items():
        if get_session_var(key) is None:
            set_session_var(key, default_value)

init_session_state()

# --- Enhanced CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.8rem;
        border-radius: 18px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
        word-wrap: break-word;
        max-width: 100%;
        width: 100%;
        color: white;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 6px solid #4f46e5;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-left: 6px solid #ec4899;
    }
    .audio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        width: 100%;
    }
    .feature-status {
        background: rgba(255,255,255,0.15);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.6rem 0;
        font-size: 1rem;
        text-align: center;
        font-weight: 600;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 12px;
        border-radius: 12px;
    }
    .stButton>button {
        height: 100%;
        width: 100%;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Voice Assistant Class ---
class StreamlitVoiceAssistant:
    def __init__(self):
        self.setup_groq()
        self.setup_wav2vec()

    def setup_groq(self):
        try:
            api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, 'secrets') else os.environ.get("GROQ_API_KEY")
            if not api_key:
                with st.sidebar:
                    st.markdown("### 🔑 Groq API Key")
                    api_key = st.text_input("Enter your Groq API Key", type="password", placeholder="gsk_xxx...")
                    st.markdown("[Get Free API Key](https://console.groq.com/)")
                if not api_key:
                    set_session_var('api_initialized', False)
                    return False
            client = Groq(api_key=api_key)
            # Test request
            client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model="llama3-70b-8192",
                max_tokens=5,
                timeout=10
            )
            set_session_var('groq_client', client)
            st.sidebar.success("✅ Groq Connected!")
            set_session_var('api_initialized', True)
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Groq init failed: {str(e)}")
            set_session_var('api_initialized', False)
            return False

    def setup_wav2vec(self):
        try:
            model_name = "facebook/wav2vec2-large-960h-lv60-self"
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            set_session_var('wav2vec_processor', processor)
            set_session_var('wav2vec_model', model)
            st.sidebar.success("🎙️ Wav2Vec Ready!")
            logger.info("Wav2Vec model loaded")
        except Exception as e:
            logger.error(f"Wav2Vec load error: {e}")
            st.sidebar.error("❌ Failed to load Wav2Vec model")

    def transcribe_audio(self, audio_bytes):
        try:
            processor = get_session_var('wav2vec_processor')
            model = get_session_var('wav2vec_model')
            if not processor or not model:
                return None
            if not audio_bytes or len(audio_bytes) < 1000:
                return None

            # Save audio to temp file
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_bytes)
                    temp_path = f.name
                audio_input, sample_rate = sf.read(temp_path)
                if len(audio_input.shape) > 1:
                    audio_input = audio_input.mean(axis=1)  # Convert to mono
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                if not LIBROSA_AVAILABLE:
                    st.warning("⚠️ Cannot resample: librosa not available")
                    return None
                audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            # Transcribe
            inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0].strip()
            if not transcription:
                return None

            return {
                'text': transcription,
                'language': 'en',
                'confidence': 'High',
                'duration': f"{len(audio_input) / sample_rate:.2f}s"
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            st.error("❌ Transcription failed")
            return None

    def get_ai_response(self, query, language='en'):
        try:
            client = get_session_var('groq_client')
            if not client:
                return None
            detail = len(query) > 50 or any(k in query.lower() for k in ['explain', 'how', 'why'])
            max_tokens = MAX_TOKENS_DETAIL if detail else MAX_TOKENS_SIMPLE
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Answer clearly in English."},
                    {"role": "user", "content": query}
                ],
                model="llama3-70b-8192",
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error("❌ AI response failed")
            logger.error(f"AI response error: {e}")
            return None

    def text_to_speech_sync(self, text, language='en'):
        if not EDGE_TTS_AVAILABLE:
            return None
        try:
            voice = get_session_var('voice_options', {}).get('en', 'en-US-AriaNeural')
            cache_key = f"{hash(text)}_{voice}"
            cache = get_session_var('audio_cache', {})
            if cache_key in cache:
                return cache[cache_key]

            if len(text) > MAX_TTS_LENGTH:
                text = '. '.join(text.split('. ')[:3]) + '.'

            import asyncio
            communicate = edge_tts.Communicate(text, voice)
            mp3_data = b""

            async def stream_audio():
                nonlocal mp3_data
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_data += chunk["data"]

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    with ThreadPoolExecutor() as executor:
                        executor.submit(lambda: asyncio.run(stream_audio())).result()
                else:
                    asyncio.run(stream_audio())
            except RuntimeError:
                asyncio.run(stream_audio())

            if mp3_data:
                if len(cache) >= MAX_CACHE_SIZE:
                    oldest = list(cache.keys())[0]
                    cache.pop(oldest)
                cache[cache_key] = mp3_data
                set_session_var('audio_cache', cache)
                return mp3_data
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    def create_audio_player(self, audio_bytes, unique_id=None):
        if not audio_bytes:
            return
        try:
            b64 = base64.b64encode(audio_bytes).decode()
            st.markdown(f"""
            <div style="margin: 15px 0;">
                <audio controls style="width:100%; max-width:100%;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    Your browser does not support audio.
                </audio>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error("❌ Audio player failed")
            logger.error(f"Audio player error: {e}")

# --- UI Functions ---
def display_feature_status():
    st.markdown("### 🛠️ Feature Status")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f'<div class="feature-status">🎙️ Voice: {"✅" if AUDIO_RECORDER_AVAILABLE else "❌"}</div>',
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="feature-status">🔊 TTS: {"✅" if EDGE_TTS_AVAILABLE else "❌"}</div>',
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="feature-status">🧠 AI: {"✅" if get_session_var("api_initialized") else "❌"}</div>',
                    unsafe_allow_html=True)

def process_user_input(assistant, user_text, lang, conf, tts):
    set_session_var('total_interactions', get_session_var('total_interactions', 0) + 1)
    uid = str(uuid.uuid4())[:8]
    st.markdown('<div class="chat-message user-message">', unsafe_allow_html=True)
    st.markdown(f"**🎤 You:** {user_text}")
    if conf != 'Text Input':
        st.markdown(f"*Confidence: {conf}*")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🧠 Thinking..."):
        response = assistant.get_ai_response(user_text, lang)
    if response:
        st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
        st.markdown("**🤖 AI Assistant:**")
        st.markdown(response)
        if tts and EDGE_TTS_AVAILABLE:
            with st.spinner("🔊 Generating speech..."):
                audio = assistant.text_to_speech_sync(response)
                if audio:
                    st.markdown("**🎧 Audio Response:**")
                    assistant.create_audio_player(audio, f"tts_{uid}")
        st.markdown('</div>', unsafe_allow_html=True)

    history = get_session_var('conversation_history', [])
    history.append({
        'timestamp': datetime.now().strftime("%H:%M"),
        'user_text': user_text,
        'ai_response': response,
        'confidence': conf
    })
    set_session_var('conversation_history', history[-10:])

def display_history():
    history = get_session_var('conversation_history', [])
    if history:
        st.markdown("---")
        st.markdown("### 🕰️ Recent Chats")
        for i, h in enumerate(reversed(history[-5:])):
            with st.expander(f"💬 Chat {len(history) - i} - {h['timestamp']}"):
                st.markdown(f"**You:** {h['user_text']}")
                st.markdown(f"**AI:** {h['ai_response']}")

# --- Main App ---
def main():
    init_session_state()
    st.markdown('<h1 class="main-header">🎙️ Voice AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#555;">Ask in voice — powered by Groq & Wav2Vec2</p>',
                unsafe_allow_html=True)

    assistant = StreamlitVoiceAssistant()
    display_feature_status()

    if not get_session_var('api_initialized'):
        st.info("🔑 Please enter your Groq API key in the sidebar.")
        st.stop()

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        enable_tts = st.checkbox("🔊 Text-to-Speech", EDGE_TTS_AVAILABLE)
        st.metric("🗨️ Chats", get_session_var('total_interactions', 0))
        if st.button("🗑️ Clear History"):
            for k in ['conversation_history', 'total_interactions', 'audio_cache']:
                set_session_var(k, [] if 'history' in k else (0 if 'interactions' in k else {}))
            st.rerun()

    # --- Voice Input ---
    if AUDIO_RECORDER_AVAILABLE:
        st.markdown('<div class="audio-container">', unsafe_allow_html=True)
        st.markdown("### 🎤 Speak Your Question")
        st.markdown("🔊 Speak clearly into the mic.")
        audio_bytes = audio_recorder(
            text="🎙️ Hold to Record",
            recording_color="#e74c3c",
            neutral_color="#27ae60",
            icon_name="microphone",
            icon_size="2x",
            key=f"rec_{get_session_var('total_interactions')}"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if audio_bytes and len(audio_bytes) > 500:
            st.audio(audio_bytes, format="audio/wav")
            with st.spinner("📝 Transcribing..."):
                result = assistant.transcribe_audio(audio_bytes)
            if result:
                process_user_input(
                    assistant,
                    result['text'],
                    result['language'],
                    result['confidence'],
                    enable_tts
                )

    # --- Text Input ---
    st.markdown("### ✍️ Or Type Your Question")
    col1, col2 = st.columns([4, 1])
    with col1:
        text_input = st.text_input("Your query:", placeholder="Explain quantum computing", key="text_input")
    with col2:
        send_btn = st.button("🚀 Send", key="send")

    if send_btn and text_input.strip():
        process_user_input(
            assistant,
            text_input.strip(),
            'en',
            'Text Input',
            enable_tts
        )

    display_history()
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#777;'>🎙️ Voice AI Assistant | Powered by Groq & Local Wav2Vec2</p>",
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()

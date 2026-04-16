"""Audio Service for handling speech-to-text and text-to-speech operations."""

import openai
import io
import base64
from typing import Optional, AsyncGenerator
from ai_service.config import settings


class AudioService:
    """Service for audio processing using OpenAI Whisper and TTS."""
    
    def __init__(self):
        """Initialize the audio service."""
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.whisper_model = settings.whisper_model
        self.tts_voice = settings.elevenlabs_voice_id  # This will be used for TTS voice selection
    
    async def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio data to text using OpenAI Whisper."""
        try:
            if not audio_data:
                return None
            
            # Create a file-like object from audio data
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"
            
            # Transcribe using Whisper
            transcript = self.client.audio.transcriptions.create(
                model=self.whisper_model,
                file=audio_file,
                response_format="text"
            )
            
            return transcript.strip() if transcript else None
            
        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            return None
    
    async def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using OpenAI TTS."""
        try:
            if not text:
                return None
            
            # Generate speech using OpenAI TTS
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",  # Using OpenAI's built-in voice
                input=text
            )
            
            # Convert response to bytes
            audio_data = response.content
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Error converting text to speech: {e}")
            return None
    
    async def text_to_speech_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """Convert text to speech with streaming output."""
        try:
            if not text:
                return
            
            # Generate streaming speech
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="pcm"  # Use PCM for streaming
            )
            
            # Stream audio data
            for chunk in response.iter_bytes(chunk_size=1024):
                yield chunk
                
        except Exception as e:
            print(f"❌ Error in streaming text to speech: {e}")
    
    def convert_audio_format(self, audio_data: bytes, target_format: str = "wav") -> bytes:
        """Convert audio data to target format."""
        try:
            # This is a placeholder for audio format conversion
            # In a real implementation, you might use libraries like pydub or ffmpeg
            return audio_data
            
        except Exception as e:
            print(f"❌ Error converting audio format: {e}")
            return audio_data
    
    def validate_audio_data(self, audio_data: bytes) -> bool:
        """Validate audio data format and size."""
        try:
            # Check if audio data is not empty
            if not audio_data:
                return False
            
            # Check audio data size (max 25MB for Whisper)
            max_size = 25 * 1024 * 1024  # 25MB
            if len(audio_data) > max_size:
                print(f"⚠️ Audio data too large: {len(audio_data)} bytes")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating audio data: {e}")
            return False

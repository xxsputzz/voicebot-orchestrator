"""
Advanced TTS Features
Additional practical features for production-ready TTS system
"""
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import unicodedata
from datetime import datetime, timedelta

class TextQuality(Enum):
    BASIC = "basic"
    ENHANCED = "enhanced" 
    PROFESSIONAL = "professional"

class OutputFormat(Enum):
    WAV = "wav"
    MP3 = "mp3" 
    OGG = "ogg"
    FLAC = "flac"

@dataclass
class ProcessingPreferences:
    """User preferences for text processing"""
    auto_punctuation: bool = True
    expand_abbreviations: bool = True
    number_pronunciation: bool = True
    emoji_handling: str = "describe"  # ignore, describe, skip
    profanity_filter: bool = False
    reading_speed_optimization: bool = True

class TextPreprocessor:
    """Advanced text preprocessing for optimal TTS results"""
    
    def __init__(self):
        # Abbreviation expansions
        self.abbreviations = {
            "Mr.": "Mister",
            "Mrs.": "Missus", 
            "Dr.": "Doctor",
            "Prof.": "Professor",
            "St.": "Street",
            "Ave.": "Avenue",
            "Blvd.": "Boulevard",
            "Inc.": "Incorporated",
            "Corp.": "Corporation",
            "Ltd.": "Limited",
            "etc.": "et cetera",
            "vs.": "versus",
            "e.g.": "for example",
            "i.e.": "that is",
            "CEO": "Chief Executive Officer",
            "CFO": "Chief Financial Officer",
            "CTO": "Chief Technology Officer",
            "AI": "Artificial Intelligence",
            "API": "Application Programming Interface",
            "URL": "Uniform Resource Locator",
            "HTML": "HyperText Markup Language",
            "CSS": "Cascading Style Sheets",
            "JavaScript": "JavaScript",
            "SQL": "Structured Query Language"
        }
        
        # Number patterns
        self.number_patterns = {
            r'\b(\d{1,3}(?:,\d{3})*)\b': self._expand_large_numbers,
            r'\b(\d+)%\b': lambda m: f"{m.group(1)} percent",
            r'\$(\d+(?:\.\d{2})?)\b': lambda m: f"{m.group(1)} dollars",
            r'\b(\d+):(\d+)\b': lambda m: f"{m.group(1)} {m.group(2)}",  # Time
            r'\b(\d+)/(\d+)/(\d+)\b': lambda m: f"{m.group(1)}/{m.group(2)}/{m.group(3)}",  # Date
        }
        
        # Emoji descriptions
        self.emoji_descriptions = {
            "😀": "grinning face",
            "😃": "grinning face with big eyes", 
            "😄": "grinning face with smiling eyes",
            "😁": "beaming face with smiling eyes",
            "😊": "smiling face with smiling eyes",
            "😍": "smiling face with heart-eyes",
            "😎": "smiling face with sunglasses",
            "😢": "crying face",
            "😭": "loudly crying face",
            "😡": "angry red face",
            "👍": "thumbs up",
            "👎": "thumbs down",
            "❤️": "red heart",
            "🎉": "party popper",
            "🔥": "fire",
            "💡": "light bulb",
            "🚀": "rocket",
            "⭐": "star",
            "✅": "check mark",
            "❌": "cross mark"
        }
        
        # Profanity filter (basic list)
        self.profanity_words = {
            # Add common profanity words and their replacements
            "damn": "darn",
            "hell": "heck",
            # Add more as needed
        }
    
    def preprocess(self, text: str, preferences: ProcessingPreferences) -> str:
        """
        Comprehensive text preprocessing for TTS optimization
        
        Args:
            text: Raw input text
            preferences: Processing preferences
            
        Returns:
            Processed text optimized for TTS
        """
        processed_text = text
        
        # Basic cleanup
        processed_text = self._clean_text(processed_text)
        
        # Handle emojis
        if preferences.emoji_handling == "describe":
            processed_text = self._describe_emojis(processed_text)
        elif preferences.emoji_handling == "skip":
            processed_text = self._remove_emojis(processed_text)
        
        # Expand abbreviations
        if preferences.expand_abbreviations:
            processed_text = self._expand_abbreviations(processed_text)
        
        # Handle numbers
        if preferences.number_pronunciation:
            processed_text = self._process_numbers(processed_text)
        
        # Auto punctuation
        if preferences.auto_punctuation:
            processed_text = self._add_punctuation(processed_text)
        
        # Profanity filter
        if preferences.profanity_filter:
            processed_text = self._filter_profanity(processed_text)
        
        # Reading speed optimization
        if preferences.reading_speed_optimization:
            processed_text = self._optimize_reading_speed(processed_text)
        
        return processed_text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def _describe_emojis(self, text: str) -> str:
        """Replace emojis with text descriptions"""
        for emoji, description in self.emoji_descriptions.items():
            text = text.replace(emoji, f" {description} ")
        
        # Handle any remaining emojis with generic description
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(" emoji ", text)
        
        return text
    
    def _remove_emojis(self, text: str) -> str:
        """Remove all emojis from text"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub("", text)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        for abbrev, expansion in self.abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _process_numbers(self, text: str) -> str:
        """Process numbers for better pronunciation"""
        for pattern, replacement in self.number_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _expand_large_numbers(self, match):
        """Expand large numbers with commas"""
        number_str = match.group(1).replace(',', '')
        number = int(number_str)
        
        if number >= 1000000000:
            billions = number // 1000000000
            remainder = number % 1000000000
            if remainder == 0:
                return f"{billions} billion"
            else:
                return f"{billions} billion {remainder}"
        elif number >= 1000000:
            millions = number // 1000000
            remainder = number % 1000000
            if remainder == 0:
                return f"{millions} million"
            else:
                return f"{millions} million {remainder}"
        elif number >= 1000:
            thousands = number // 1000
            remainder = number % 1000
            if remainder == 0:
                return f"{thousands} thousand"
            else:
                return f"{thousands} thousand {remainder}"
        else:
            return str(number)
    
    def _add_punctuation(self, text: str) -> str:
        """Add punctuation for better speech flow"""
        # Add periods after sentences that don't end with punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        processed_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[-1] in '.!?':
                sentence += '.'
            processed_sentences.append(sentence)
        
        return ' '.join(processed_sentences)
    
    def _filter_profanity(self, text: str) -> str:
        """Filter profanity words"""
        for profane, replacement in self.profanity_words.items():
            pattern = r'\b' + re.escape(profane) + r'\b'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _optimize_reading_speed(self, text: str) -> str:
        """Optimize text for natural reading speed"""
        # Add pauses after certain patterns
        text = re.sub(r'([.!?])\s+', r'\1 ... ', text)  # Longer pause after sentences
        text = re.sub(r'([,:;])\s+', r'\1 .. ', text)   # Short pause after clauses
        
        # Break up very long sentences
        long_sentence_pattern = r'([^.!?]{100,}?)([,:;])'
        text = re.sub(long_sentence_pattern, r'\1\2 ... ', text)
        
        return text

class SSMLGenerator:
    """Generate SSML markup for advanced speech control"""
    
    def __init__(self):
        self.break_strengths = {
            "none": "none",
            "x-weak": "x-weak", 
            "weak": "weak",
            "medium": "medium",
            "strong": "strong",
            "x-strong": "x-strong"
        }
    
    def generate_ssml(
        self,
        text: str,
        voice: str = "default",
        rate: str = "medium",
        pitch: str = "medium",
        volume: str = "medium",
        emphasis_words: List[str] = None,
        break_positions: List[Tuple[int, str]] = None
    ) -> str:
        """
        Generate SSML markup for advanced speech control
        
        Args:
            text: Input text
            voice: Voice name
            rate: Speaking rate (x-slow, slow, medium, fast, x-fast)
            pitch: Pitch level (x-low, low, medium, high, x-high)
            volume: Volume level (silent, x-soft, soft, medium, loud, x-loud)
            emphasis_words: List of words to emphasize
            break_positions: List of (position, strength) for pauses
            
        Returns:
            SSML markup string
        """
        # Start with root element
        ssml = f'<speak version="1.0" xml:lang="en-US">'
        
        # Add voice selection
        ssml += f'<voice name="{voice}">'
        
        # Add prosody control
        ssml += f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">'
        
        # Process text with emphasis and breaks
        processed_text = self._apply_ssml_markup(text, emphasis_words, break_positions)
        ssml += processed_text
        
        # Close tags
        ssml += '</prosody></voice></speak>'
        
        return ssml
    
    def _apply_ssml_markup(
        self, 
        text: str, 
        emphasis_words: List[str] = None,
        break_positions: List[Tuple[int, str]] = None
    ) -> str:
        """Apply SSML markup to text"""
        processed_text = text
        
        # Apply emphasis to words
        if emphasis_words:
            for word in emphasis_words:
                pattern = r'\b' + re.escape(word) + r'\b'
                processed_text = re.sub(
                    pattern, 
                    f'<emphasis level="strong">{word}</emphasis>',
                    processed_text,
                    flags=re.IGNORECASE
                )
        
        # Apply breaks at specified positions
        if break_positions:
            # Sort by position in reverse order to maintain correct positions
            break_positions = sorted(break_positions, key=lambda x: x[0], reverse=True)
            
            for position, strength in break_positions:
                if 0 <= position <= len(processed_text):
                    break_tag = f'<break strength="{strength}"/>'
                    processed_text = (
                        processed_text[:position] + 
                        break_tag + 
                        processed_text[position:]
                    )
        
        return processed_text

class BatchProcessor:
    """Batch processing for multiple texts"""
    
    def __init__(self, tts_engine):
        self.tts_engine = tts_engine
        
    async def process_batch(
        self,
        texts: List[str],
        voice: str = "default",
        emotion: str = "neutral",
        output_format: OutputFormat = OutputFormat.WAV,
        quality: TextQuality = TextQuality.ENHANCED,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch
        
        Args:
            texts: List of texts to synthesize
            voice: Voice to use
            emotion: Emotion style
            output_format: Output audio format
            quality: Processing quality level
            progress_callback: Optional progress callback function
            
        Returns:
            List of results with audio data and metadata
        """
        results = []
        preprocessor = TextPreprocessor()
        
        # Set processing preferences based on quality
        preferences = self._get_quality_preferences(quality)
        
        for i, text in enumerate(texts):
            try:
                # Preprocess text
                processed_text = preprocessor.preprocess(text, preferences)
                
                # Synthesize
                audio_bytes = await self.tts_engine.synthesize_speech(
                    text=processed_text,
                    voice=voice,
                    emotion=emotion,
                    high_quality=(quality == TextQuality.PROFESSIONAL)
                )
                
                result = {
                    "index": i,
                    "original_text": text,
                    "processed_text": processed_text,
                    "audio_bytes": audio_bytes,
                    "status": "success",
                    "metadata": {
                        "voice": voice,
                        "emotion": emotion,
                        "format": output_format.value,
                        "quality": quality.value,
                        "audio_size": len(audio_bytes),
                        "text_length": len(text)
                    }
                }
                
                results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(texts))
                    
            except Exception as e:
                result = {
                    "index": i,
                    "original_text": text,
                    "status": "error",
                    "error": str(e)
                }
                results.append(result)
        
        return results
    
    def _get_quality_preferences(self, quality: TextQuality) -> ProcessingPreferences:
        """Get processing preferences based on quality level"""
        if quality == TextQuality.BASIC:
            return ProcessingPreferences(
                auto_punctuation=False,
                expand_abbreviations=False,
                number_pronunciation=False,
                emoji_handling="ignore",
                profanity_filter=False,
                reading_speed_optimization=False
            )
        elif quality == TextQuality.ENHANCED:
            return ProcessingPreferences(
                auto_punctuation=True,
                expand_abbreviations=True,
                number_pronunciation=True,
                emoji_handling="describe",
                profanity_filter=False,
                reading_speed_optimization=True
            )
        else:  # PROFESSIONAL
            return ProcessingPreferences(
                auto_punctuation=True,
                expand_abbreviations=True,
                number_pronunciation=True,
                emoji_handling="describe",
                profanity_filter=True,
                reading_speed_optimization=True
            )

class TTSAnalytics:
    """Analytics and usage tracking for TTS system"""
    
    def __init__(self):
        self.usage_stats = {
            "total_syntheses": 0,
            "total_characters": 0,
            "total_audio_generated": 0,
            "voice_usage": {},
            "emotion_usage": {},
            "error_count": 0,
            "average_processing_time": 0.0
        }
        
        self.session_stats = {
            "session_start": datetime.now(),
            "syntheses_this_session": 0,
            "characters_this_session": 0
        }
    
    def record_synthesis(
        self,
        text_length: int,
        audio_size: int,
        processing_time: float,
        voice: str,
        emotion: str,
        success: bool = True
    ):
        """Record synthesis statistics"""
        if success:
            self.usage_stats["total_syntheses"] += 1
            self.usage_stats["total_characters"] += text_length
            self.usage_stats["total_audio_generated"] += audio_size
            
            # Update voice usage
            if voice not in self.usage_stats["voice_usage"]:
                self.usage_stats["voice_usage"][voice] = 0
            self.usage_stats["voice_usage"][voice] += 1
            
            # Update emotion usage
            if emotion not in self.usage_stats["emotion_usage"]:
                self.usage_stats["emotion_usage"][emotion] = 0
            self.usage_stats["emotion_usage"][emotion] += 1
            
            # Update average processing time
            current_avg = self.usage_stats["average_processing_time"]
            total_syntheses = self.usage_stats["total_syntheses"]
            self.usage_stats["average_processing_time"] = (
                (current_avg * (total_syntheses - 1) + processing_time) / total_syntheses
            )
            
            # Session stats
            self.session_stats["syntheses_this_session"] += 1
            self.session_stats["characters_this_session"] += text_length
        else:
            self.usage_stats["error_count"] += 1
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        session_duration = datetime.now() - self.session_stats["session_start"]
        
        # Calculate rates
        chars_per_synthesis = (
            self.usage_stats["total_characters"] / max(self.usage_stats["total_syntheses"], 1)
        )
        
        audio_per_char = (
            self.usage_stats["total_audio_generated"] / max(self.usage_stats["total_characters"], 1)
        )
        
        # Top voices and emotions
        top_voices = sorted(
            self.usage_stats["voice_usage"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_emotions = sorted(
            self.usage_stats["emotion_usage"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "overall_stats": self.usage_stats,
            "session_stats": {
                **self.session_stats,
                "session_duration_minutes": session_duration.total_seconds() / 60
            },
            "calculated_metrics": {
                "average_chars_per_synthesis": chars_per_synthesis,
                "average_audio_bytes_per_char": audio_per_char,
                "success_rate": (
                    (self.usage_stats["total_syntheses"] / 
                     max(self.usage_stats["total_syntheses"] + self.usage_stats["error_count"], 1)) * 100
                )
            },
            "top_voices": top_voices,
            "top_emotions": top_emotions,
            "report_generated": datetime.now().isoformat()
        }

# Export classes for use in other modules
__all__ = [
    'TextPreprocessor', 'ProcessingPreferences', 'TextQuality',
    'SSMLGenerator', 'BatchProcessor', 'TTSAnalytics', 'OutputFormat'
]

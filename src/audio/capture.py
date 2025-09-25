import numpy as np
import threading
import queue
import time
import subprocess
import platform
from typing import Optional, Callable


class AudioCapture:
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 channels: int = 1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        self.ffmpeg_process = None
        
    def start_recording(self):
        """Start audio recording using system audio tools"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio_system)
        self.recording_thread.start()
        
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process = None
        if self.recording_thread:
            self.recording_thread.join()
            
    def _record_audio_system(self):
        """Record audio using system tools (ffmpeg/sox)"""
        try:
            # Try ffmpeg first
            if self._check_command('ffmpeg'):
                self._record_with_ffmpeg()
            else:
                raise Exception("ffmpeg가 설치되지 않았습니다. 'brew install ffmpeg'로 설치해주세요.")
                
        except Exception as e:
            print(f"오디오 캡처 오류: {e}")
            raise e
    
    def _check_command(self, command):
        """Check if a command exists"""
        try:
            subprocess.run([command, '-version'], capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def _record_with_ffmpeg(self):
        """Record using ffmpeg"""
        try:
            # macOS의 경우 기본 마이크 사용
            if platform.system() == "Darwin":
                input_device = ":default"  # macOS default input
            else:
                input_device = "default"
                
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-f', 'avfoundation' if platform.system() == "Darwin" else 'alsa',
                '-i', input_device,
                '-ar', str(self.sample_rate),
                '-ac', str(self.channels),
                '-f', 'f32le',  # Output as 32-bit float
                '-'  # Output to stdout
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            
            while self.is_recording and self.ffmpeg_process.poll() is None:
                # Read chunk_size samples (4 bytes per float32 sample)
                chunk_bytes = self.chunk_size * 4
                data = self.ffmpeg_process.stdout.read(chunk_bytes)
                
                if len(data) == chunk_bytes:
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    self.audio_queue.put(audio_data)
                else:
                    break
                    
        except Exception as e:
            print(f"FFmpeg 녹음 오류: {e}")
            raise e
    
                
    def get_audio_data(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get audio data from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_audio_buffer(self, duration_seconds: float) -> np.ndarray:
        """Get audio buffer of specified duration"""
        samples_needed = int(self.sample_rate * duration_seconds)
        buffer = np.array([])
        
        start_time = time.time()
        while len(buffer) < samples_needed:
            if time.time() - start_time > duration_seconds + 2.0:
                break
                
            chunk = self.get_audio_data(timeout=0.1)
            if chunk is not None:
                buffer = np.concatenate([buffer, chunk])
                
        return buffer[:samples_needed] if len(buffer) >= samples_needed else buffer
        
    def __del__(self):
        self.stop_recording()


class AudioFileLoader:
    """Load audio from file for testing"""
    
    @staticmethod
    def load_audio(file_path: str, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]:
        """Load audio file and resample if needed"""
        import soundfile as sf
        import librosa
        
        audio_data, sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Resample if needed
        if sample_rate != target_sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
            
        return audio_data.astype(np.float32), target_sample_rate
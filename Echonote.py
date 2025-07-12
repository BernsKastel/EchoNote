import whisper
from pathlib import Path
import warnings
import subprocess
import os
import time
import math
from datetime import datetime, timedelta
import shutil
import torch
import numpy as np
from typing import List, Tuple, Optional

# Suprime advertencias
warnings.filterwarnings("ignore")

class TranscriptorWhisperOptimizado:
    def __init__(self, audio_path: str):
        self.audio_path = Path(audio_path)
        self.desktop_path = Path.home() / "Desktop"
        
        # Nombre seguro para carpetas
        safe_name = self.audio_path.stem[:50]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.project_folder = self.desktop_path / f"Transcripcion_{safe_name}_{timestamp}"
        
        # Configuración optimizada
        self.chunk_duration = 300  # 5 minutos
        self.overlap_duration = 10  # Reducido para mayor velocidad
        
        # Variables de estado
        self.model = None
        self.total_chunks = 0
        self.start_time = None
        self.processed_chunks = 0
        self.chunk_transcriptions = []
        
        # Configuración optimizada para FFmpeg
        self.ffmpeg_base_options = [
            "-hide_banner", "-loglevel", "error", "-y"
        ]
        
    def print_progress_bar(self, current: int, total: int, prefix: str = "", suffix: str = "", length: int = 50):
        """Muestra barra de progreso optimizada"""
        if self.start_time is None:
            self.start_time = time.time()
            
        elapsed_time = time.time() - self.start_time
        if current > 0:
            estimated_total = elapsed_time * total / current
            remaining_time = estimated_total - elapsed_time
            eta = str(timedelta(seconds=int(remaining_time)))
        else:
            eta = "Calculando..."
            
        percent = (current / total) * 100
        filled_length = int(length * current // total)
        bar = "█" * filled_length + "░" * (length - filled_length)
        
        print(f"\r{prefix} |{bar}| {percent:.1f}% {current}/{total} - ETA: {eta} {suffix}", end="", flush=True)
        if current == total:
            print()
    
    def get_audio_duration(self) -> float:
        """Obtiene duración del audio de forma optimizada"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(self.audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            print(f"❌ Error obteniendo duración: {e}")
            return 0
    
    def create_chunks_batch(self, duration: float) -> List[Path]:
        """Crea todos los chunks en una sola operación por lotes"""
        print(f"⚡ Creando chunks de audio de {duration/60:.1f} minutos...")
        
        # Crear directorios
        self.project_folder.mkdir(parents=True, exist_ok=True)
        chunks_folder = self.project_folder / "chunks_audio"
        chunks_folder.mkdir(parents=True, exist_ok=True)
        
        self.total_chunks = math.ceil(duration / self.chunk_duration)
        chunk_paths = []
        
        print(f"📊 Procesando {self.total_chunks} chunks...")
        
        # Crear chunks secuencialmente pero optimizado
        for i in range(self.total_chunks):
            start_time = i * self.chunk_duration
            chunk_name = f"chunk_{i+1:03d}.wav"
            chunk_path = chunks_folder / chunk_name
            
            # Calcular duración real del chunk
            remaining_duration = duration - start_time
            actual_duration = min(self.chunk_duration, remaining_duration)
            
            # Comando FFmpeg optimizado
            cmd = [
                "ffmpeg", *self.ffmpeg_base_options,
                "-i", str(self.audio_path),
                "-ss", str(start_time),
                "-t", str(actual_duration),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-avoid_negative_ts", "make_zero",
                str(chunk_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, timeout=120)
                
                # Verificar que el chunk se creó correctamente
                if chunk_path.exists() and chunk_path.stat().st_size > 1000:
                    chunk_paths.append(chunk_path)
                    self.print_progress_bar(len(chunk_paths), self.total_chunks, 
                                          "🔄 Creando chunks:", f"✅ {i+1}")
                else:
                    print(f"\n❌ Chunk {i+1} inválido o muy pequeño")
                    
            except subprocess.TimeoutExpired:
                print(f"\n⏱️ Timeout en chunk {i+1}")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error en chunk {i+1}: {e}")
        
        print(f"\n✅ {len(chunk_paths)} chunks creados exitosamente")
        return chunk_paths
    
    def load_model_optimized(self) -> bool:
        """Carga el modelo Whisper de forma optimizada"""
        print("🤖 Cargando modelo Whisper...")
        
        # Limpiar memoria GPU si está disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = "cuda"
            print("🚀 Usando GPU para aceleración")
        else:
            device = "cpu"
            print("💻 Usando CPU")
        
        try:
            # Usar modelo "base" como mejor balance velocidad/precisión
            self.model = whisper.load_model("base", device=device)
            print("✅ Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            # Fallback a CPU si GPU falla
            if device == "cuda":
                try:
                    self.model = whisper.load_model("base", device="cpu")
                    print("✅ Modelo cargado en CPU como fallback")
                    return True
                except Exception as e2:
                    print(f"❌ Error en fallback CPU: {e2}")
            return False
    
    def transcribe_chunk_fast(self, chunk_path: Path, chunk_index: int) -> str:
        """Transcribe un chunk de forma optimizada"""
        try:
            # Configuración optimizada para velocidad
            result = self.model.transcribe(
                str(chunk_path),
                language="es",
                task="transcribe",
                verbose=False,
                fp16=torch.cuda.is_available(),  # Usar fp16 solo en GPU
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )
            
            text = result.get("text", "").strip()
            if not text:
                return f"[SIN AUDIO] Chunk {chunk_index}"
            
            # Limpiar texto
            text = ' '.join(text.split())
            return text
            
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"\n❌ Error en chunk {chunk_index}: {error_msg}")
            return f"[ERROR] Chunk {chunk_index}: {error_msg}"
    
    def save_chunk_transcription(self, chunk_index: int, transcription: str, 
                               start_time_seconds: float) -> Optional[Path]:
        """Guarda la transcripción de un chunk"""
        try:
            chunk_folder = self.project_folder / "transcripciones_por_chunk"
            chunk_folder.mkdir(parents=True, exist_ok=True)
            
            start_time_str = str(timedelta(seconds=int(start_time_seconds)))
            end_time_str = str(timedelta(seconds=int(start_time_seconds + self.chunk_duration)))
            
            filename = f"chunk_{chunk_index:03d}_transcripcion.txt"
            filepath = chunk_folder / filename
            
            content = f"""# TRANSCRIPCIÓN CHUNK {chunk_index}
# Archivo: {self.audio_path.name}
# Tiempo: {start_time_str} - {end_time_str}
# Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Caracteres: {len(transcription)}

{transcription}
"""
            
            filepath.write_text(content, encoding="utf-8")
            return filepath
            
        except Exception as e:
            print(f"❌ Error guardando chunk {chunk_index}: {e}")
            return None
    
    def process_chunks_sequentially(self, chunk_paths: List[Path]) -> bool:
        """Procesa chunks secuencialmente de forma optimizada"""
        print(f"\n🚀 Transcribiendo {len(chunk_paths)} chunks...")
        
        self.start_time = time.time()
        self.processed_chunks = 0
        self.chunk_transcriptions = []
        
        for i, chunk_path in enumerate(chunk_paths):
            chunk_index = i + 1
            start_time_seconds = i * self.chunk_duration
            
            # Transcribir chunk
            transcription = self.transcribe_chunk_fast(chunk_path, chunk_index)
            
            # Guardar transcripción individual
            chunk_file = self.save_chunk_transcription(chunk_index, transcription, start_time_seconds)
            
            # Almacenar resultado
            self.chunk_transcriptions.append({
                'index': chunk_index,
                'transcription': transcription,
                'start_time': start_time_seconds,
                'file_path': chunk_file,
                'is_valid': not transcription.startswith('[ERROR]') and not transcription.startswith('[SIN AUDIO]')
            })
            
            self.processed_chunks += 1
            
            # Actualizar progreso
            status = "✅" if self.chunk_transcriptions[-1]['is_valid'] else "❌"
            self.print_progress_bar(self.processed_chunks, len(chunk_paths),
                                  "📝 Transcribiendo:", f"{status} Chunk {chunk_index}")
            
            # Limpiar memoria entre chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n✅ Transcripción completada")
        return True
    
    def create_final_transcription(self) -> Optional[Path]:
        """Crea la transcripción final combinada"""
        print("\n🔗 Creando transcripción final...")
        
        # Separar transcripciones válidas e inválidas
        valid_transcriptions = [t for t in self.chunk_transcriptions if t['is_valid']]
        invalid_transcriptions = [t for t in self.chunk_transcriptions if not t['is_valid']]
        
        # Crear contenido final
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.audio_path.stem[:30]
        filename = f"Transcripcion_COMPLETA_{safe_name}_{timestamp}.txt"
        output_path = self.project_folder / filename
        
        # Estadísticas
        total_time = time.time() - self.start_time if self.start_time else 0
        total_chunks = len(self.chunk_transcriptions)
        valid_chunks = len(valid_transcriptions)
        error_chunks = len(invalid_transcriptions)
        
        # Contenido del archivo
        content_parts = []
        
        # Cabecera
        content_parts.append(f"""# TRANSCRIPCIÓN COMPLETA - WHISPER OPTIMIZADO
# Archivo: {self.audio_path.name}
# Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Chunks procesados: {total_chunks}
# Chunks válidos: {valid_chunks}
# Chunks con errores: {error_chunks}
# Éxito: {(valid_chunks/total_chunks)*100:.1f}%
# Tiempo de procesamiento: {total_time/60:.1f} minutos

{'='*100}
TRANSCRIPCIÓN COMPLETA
{'='*100}
""")
        
        # Transcripciones válidas
        for chunk_data in valid_transcriptions:
            start_time_str = str(timedelta(seconds=int(chunk_data['start_time'])))
            end_time_str = str(timedelta(seconds=int(chunk_data['start_time'] + self.chunk_duration)))
            
            content_parts.append(f"""
{'='*80}
[{start_time_str} - {end_time_str}] CHUNK {chunk_data['index']}
{'='*80}
{chunk_data['transcription']}
""")
        
        # Errores si los hay
        if invalid_transcriptions:
            content_parts.append(f"""
{'='*100}
CHUNKS CON ERRORES
{'='*100}
""")
            for chunk_data in invalid_transcriptions:
                start_time_str = str(timedelta(seconds=int(chunk_data['start_time'])))
                content_parts.append(f"CHUNK {chunk_data['index']} ({start_time_str}): {chunk_data['transcription']}")
        
        # Guardar archivo
        try:
            full_content = '\n'.join(content_parts)
            output_path.write_text(full_content, encoding="utf-8")
            print(f"✅ Transcripción guardada: {filename}")
            return output_path
        except Exception as e:
            print(f"❌ Error guardando transcripción final: {e}")
            return None
    
    def process_audio_optimized(self) -> bool:
        """Procesa el audio completo de forma optimizada"""
        print(f"🎵 Procesando: {self.audio_path.name}")
        
        # Verificaciones iniciales
        if not self.audio_path.exists():
            print(f"❌ Archivo no encontrado: {self.audio_path}")
            return False
        
        file_size_mb = self.audio_path.stat().st_size / (1024*1024)
        print(f"📊 Tamaño: {file_size_mb:.1f} MB")
        
        # 1. Obtener duración
        duration = self.get_audio_duration()
        if duration <= 0:
            print("❌ No se pudo obtener duración del archivo")
            return False
        
        print(f"⏱️ Duración: {duration/60:.1f} minutos")
        
        # 2. Cargar modelo Whisper
        if not self.load_model_optimized():
            print("❌ Error cargando modelo Whisper")
            return False
        
        # 3. Crear chunks
        chunk_paths = self.create_chunks_batch(duration)
        if not chunk_paths:
            print("❌ No se pudieron crear chunks")
            return False
        
        # 4. Procesar chunks secuencialmente
        if not self.process_chunks_sequentially(chunk_paths):
            print("❌ Error procesando chunks")
            return False
        
        # 5. Crear transcripción final
        final_file = self.create_final_transcription()
        if final_file is None:
            print("❌ Error creando transcripción final")
            return False
        
        # 6. Limpieza opcional
        cleanup = input("\n🗑️ ¿Eliminar chunks de audio temporales? (s/n): ").lower().strip()
        if cleanup == 's':
            try:
                chunks_folder = self.project_folder / "chunks_audio"
                if chunks_folder.exists():
                    shutil.rmtree(chunks_folder)
                    print("🗑️ Chunks temporales eliminados")
            except Exception as e:
                print(f"⚠️ Error eliminando temporales: {e}")
        
        # Resumen final
        total_time = time.time() - self.start_time
        valid_chunks = len([t for t in self.chunk_transcriptions if t['is_valid']])
        total_chunks = len(self.chunk_transcriptions)
        
        print(f"\n🎉 ¡Proceso completado!")
        print(f"📁 Carpeta: {self.project_folder}")
        print(f"📄 Éxito: {valid_chunks}/{total_chunks} chunks ({(valid_chunks/total_chunks)*100:.1f}%)")
        print(f"⏱️ Tiempo total: {total_time/60:.1f} minutos")
        print(f"⚡ Velocidad: {(duration/60)/(total_time/60):.1f}x tiempo real")
        
        return True

def main():
    print("🎙️ TRANSCRIPTOR WHISPER OPTIMIZADO")
    print("=" * 50)
    print("🚀 Versión optimizada para máximo rendimiento")
    print("📋 Procesamiento secuencial: Audio → Chunks → Transcripción")
    print("⚡ Configuración optimizada para velocidad")
    
    # Solicitar archivo
    while True:
        audio_path = input("\n📂 Ruta del archivo de audio: ").strip()
        if audio_path:
            audio_path = audio_path.strip('"').strip("'")
            if Path(audio_path).exists():
                break
            else:
                print("❌ Archivo no encontrado")
        else:
            print("❌ Ingresa una ruta válida")
    
    # Procesar
    start_time = time.time()
    transcriptor = TranscriptorWhisperOptimizado(audio_path)
    success = transcriptor.process_audio_optimized()
    
    total_time = time.time() - start_time
    
    if success:
        print(f"\n🎊 ¡Transcripción completada en {total_time/60:.1f} minutos!")
        print("📝 Revisa la carpeta del proyecto para todos los archivos")
    else:
        print(f"\n❌ Error en la transcripción después de {total_time/60:.1f} minutos")
        print("💡 Verifica que el archivo de audio sea válido")
    
    input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    main()
    

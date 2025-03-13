#!/usr/bin/env python
"""
VideoClipExtractor especializado en extraer solo kills y momentos graciosos,
con títulos y descripciones concisos generados por IA.
"""

import cv2
import numpy as np
import os
import subprocess
import time
import re
import json
import argparse
import requests
import logging
import librosa
import soundfile as sf
import tempfile
import concurrent.futures
import traceback
import pickle
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm

# Importar AudioAnalyzer si está disponible
try:
    from audio import AudioAnalyzer, numpy_to_python
    AUDIO_ANALYZER_AVAILABLE = True
except ImportError:
    AUDIO_ANALYZER_AVAILABLE = False
    print("AudioAnalyzer no está disponible. Se creará una versión básica integrada.")

# Intenta importar whisper, pero proporciona una alternativa si falla
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper no está disponible. Se utilizará análisis de audio alternativo.")

# Definir clase AudioAnalyzer integrada en caso de que el módulo no esté disponible
if not AUDIO_ANALYZER_AVAILABLE:
    def numpy_to_python(obj):
        """Convierte tipos de NumPy a tipos nativos de Python para serialización JSON."""
        if hasattr(obj, "item"):
            return obj.item()  # Convierte arrays de un solo elemento
        elif hasattr(obj, "tolist"):
            return obj.tolist()  # Convierte arrays
        elif isinstance(obj, dict):
            return {key: numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [numpy_to_python(item) for item in obj]
        else:
            return obj

    class AudioAnalyzer:
        """Versión básica integrada del AudioAnalyzer enfocada en risas"""
        def __init__(self, temp_folder="temp", max_workers=4, log_level="INFO"):
            self.logger = logging.getLogger(__name__)
            self.temp_folder = Path(temp_folder)
            self.max_workers = max_workers
            self.temp_folder.mkdir(exist_ok=True)
            
        def extract_audio(self, video_path):
            """Extrae el audio del video utilizando FFmpeg"""
            self.logger.info(f"Extrayendo audio optimizado del video: {video_path}")
            audio_path = self.temp_folder / f"{Path(video_path).stem}_audio_opt.wav"
            
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-af", "dynaudnorm",
                "-y", str(audio_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"Audio extraído correctamente: {audio_path}")
            return audio_path
            
        def process_audio(self, video_path, segment_duration=5.0, threshold=0.65):
            """Procesa el audio para detectar momentos de risa"""
            self.logger.info("Usando análisis de audio básico enfocado en risas")
            highlights = []
            
            try:
                # Extraer y cargar el audio
                audio_path = self.extract_audio(video_path)
                y, sr = librosa.load(str(audio_path), sr=16000)
                
                # Analizar audio en segmentos
                segment_samples = int(segment_duration * sr)
                hop_samples = segment_samples // 2  # 50% de superposición
                
                for i in range(0, len(y) - segment_samples, hop_samples):
                    start_sample = i
                    end_sample = i + segment_samples
                    segment = y[start_sample:end_sample]
                    
                    # Calcular características básicas
                    rms = np.sqrt(np.mean(segment**2))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(segment)[0])
                    
                    # Detectar momentos de risa - alta variabilidad en energía + alto ZCR
                    if rms > 0.1 and zcr > 0.1:  # Alta energía y alta tasa de cruces por cero = posible risa
                        start_time = start_sample / sr
                        end_time = end_sample / sr
                        
                        # Ajustar confianza según intensidad de energía y ZCR
                        confianza = min(0.6 + rms + zcr, 0.95)
                        
                        if confianza >= threshold:
                            highlights.append({
                                "start": float(start_time),
                                "end": float(end_time),
                                "tipo": "risa",
                                "descripción": "Risas o momentos de diversión detectados en el audio",
                                "confianza": float(confianza),
                                "origen": "audio"
                            })
            
            except Exception as e:
                self.logger.error(f"Error en análisis de audio para risas: {e}")
                
            return highlights

# Clase principal para procesamiento de video
class VideoClipExtractor:
    def __init__(self, 
                 model_name: str = "llava:latest", 
                 temp_folder: str = "temp", 
                 output_folder: str = "clips",
                 checkpoint_folder: str = "checkpoints",
                 max_workers: int = 4,
                 timeout: int = 60,  # Timeout para requests en segundos
                 high_performance: bool = True,
                 use_audio_analyzer: bool = True,
                 log_level: str = "INFO"):
        """
        Inicializa el extractor de clips de video enfocado en kills y momentos graciosos.
        
        Args:
            model_name: Nombre del modelo de Ollama a utilizar
            temp_folder: Carpeta para archivos temporales
            output_folder: Carpeta donde se guardarán los clips
            checkpoint_folder: Carpeta para guardar checkpoints de progreso
            max_workers: Número máximo de workers para procesamiento paralelo
            timeout: Tiempo máximo para esperar respuestas de la API (en segundos)
            high_performance: Si es True, usa configuraciones de alto rendimiento
            use_audio_analyzer: Si es True, usa AudioAnalyzer para análisis de audio
            log_level: Nivel de logging
        """
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("videoclip_extractor.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.model_name = model_name
        self.temp_folder = Path(temp_folder)
        self.output_folder = Path(output_folder)
        self.checkpoint_folder = Path(checkpoint_folder)
        self.max_workers = max_workers
        self.timeout = timeout
        self.high_performance = high_performance
        
        # Manejar interrupciones (Ctrl+C)
        signal.signal(signal.SIGINT, self._handle_interrupt)
        self.interrupted = False
        
        # Crear carpetas necesarias
        self.temp_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        self.checkpoint_folder.mkdir(exist_ok=True)
        
        # Configurar analizador de audio
        self.use_audio_analyzer = use_audio_analyzer
        if use_audio_analyzer:
            self.audio_analyzer = AudioAnalyzer(
                temp_folder=temp_folder,
                max_workers=max_workers,
                log_level=log_level
            )
            
        # Verificar que Ollama esté funcionando
        try:
            self._check_ollama()
        except Exception as e:
            self.logger.error(f"Error al verificar Ollama: {e}")
            if high_performance:
                self.logger.warning("Continuando sin verificar Ollama en modo alto rendimiento")
            else:
                raise
    
    def _handle_interrupt(self, sig, frame):
        """Maneja la interrupción del proceso (Ctrl+C)"""
        self.logger.warning("Interrupción detectada. Guardando progreso y finalizando...")
        self.interrupted = True
    
    def _check_ollama(self) -> None:
        """Verifica que Ollama esté funcionando y el modelo esté disponible."""
        try:
            # Verificar que Ollama esté corriendo
            response = requests.get("http://localhost:11434/api/tags", timeout=self.timeout)
            if response.status_code != 200:
                raise ConnectionError("No se pudo conectar a Ollama. Asegúrate de que esté ejecutándose.")
            
            # Verificar que el modelo exista
            models = response.json().get("models", [])
            model_exists = any(model.get("name") == self.model_name for model in models)
            
            if not model_exists:
                self.logger.info(f"El modelo {self.model_name} no está disponible. Intentando descargarlo...")
                subprocess.run(["ollama", "pull", self.model_name], check=True)
                self.logger.info(f"Modelo {self.model_name} descargado correctamente.")
            
        except requests.exceptions.ConnectionError:
            self.logger.error("No se pudo conectar a Ollama. Asegúrate de que esté ejecutándose en http://localhost:11434")
            raise ConnectionError("No se pudo conectar a Ollama. Asegúrate de que esté ejecutándose en http://localhost:11434")
        except subprocess.CalledProcessError:
            self.logger.error(f"No se pudo descargar el modelo {self.model_name}.")
            raise RuntimeError(f"No se pudo descargar el modelo {self.model_name}. Verifica que el nombre sea correcto.")
    
    def save_checkpoint(self, data: Dict, name: str) -> None:
        """Guarda un checkpoint del procesamiento actual"""
        try:
            checkpoint_path = self.checkpoint_folder / f"{name}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Checkpoint guardado: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error al guardar checkpoint: {e}")
    
    def load_checkpoint(self, name: str) -> Optional[Dict]:
        """Carga un checkpoint guardado previamente"""
        checkpoint_path = self.checkpoint_folder / f"{name}.pkl"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
                self.logger.info(f"Checkpoint cargado: {checkpoint_path}")
                return data
            except Exception as e:
                self.logger.error(f"Error al cargar checkpoint: {e}")
        return None
        
    def extract_frames(self, 
                       video_path: str, 
                       sample_rate: float = 0.5,
                       adaptive_sampling: bool = True,
                       max_frames: int = 1000,  # Límite máximo de frames a extraer
                      ) -> List[Tuple[float, Path]]:
        """
        Extrae frames del video de manera optimizada.
        
        Args:
            video_path: Ruta al archivo de video
            sample_rate: Cantidad de frames a extraer por segundo
            adaptive_sampling: Si es True, utiliza muestreo adaptativo
            max_frames: Número máximo de frames a extraer
            
        Returns:
            Lista de tuplas (timestamp, frame_path)
        """
        # Verificar si existe un checkpoint
        checkpoint_name = f"frames_{Path(video_path).stem}"
        checkpoint_data = self.load_checkpoint(checkpoint_name)
        if checkpoint_data:
            self.logger.info(f"Reanudando extracción de frames desde checkpoint")
            return checkpoint_data['frames_info']
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        self.logger.info(f"Video: {video_path}")
        self.logger.info(f"Duración: {duration:.2f} segundos")
        self.logger.info(f"FPS: {fps}")
        self.logger.info(f"Total de frames: {total_frames}")
        
        # En modo alto rendimiento, limitar el número total de frames a analizar
        if self.high_performance:
            # Calcular cuántos frames saltamos para llegar al máximo deseado
            if total_frames > max_frames:
                step = total_frames // max_frames
                self.logger.info(f"Modo alto rendimiento: Analizando 1 de cada {step} frames")
            else:
                step = 1
        else:
            # Calcular cada cuántos frames guardar uno para análisis basado en sample_rate
            step = max(1, int(fps / sample_rate))
            
        frames_info = []
        count = 0
        prev_frame = None
        
        # Usar tqdm para mostrar una barra de progreso
        with tqdm(total=total_frames, desc="Extrayendo frames") as pbar:
            while count < total_frames:
                if self.interrupted:
                    break
                    
                # Leer el frame
                ret, frame = video.read()
                if not ret:
                    break
                
                # Decidir si procesamos este frame
                process_frame = False
                
                if count % step == 0:
                    # Por defecto, procesamos según el intervalo
                    process_frame = True
                elif adaptive_sampling and prev_frame is not None and not self.high_performance:
                    # Solo usar muestreo adaptativo en modo normal
                    diff = cv2.absdiff(frame, prev_frame)
                    non_zero_count = np.count_nonzero(diff)
                    if non_zero_count > frame.shape[0] * frame.shape[1] * 0.1:  # 10% de cambio
                        process_frame = True
                
                # Guardar el frame actual para la próxima comparación si usamos adaptativo
                if adaptive_sampling and not self.high_performance:
                    prev_frame = frame.copy()
                
                # Procesar el frame si es necesario
                if process_frame:
                    timestamp = count / fps
                    frame_path = self.temp_folder / f"frame_{count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frames_info.append((timestamp, frame_path))
                    
                    # En modo alto rendimiento, guardar checkpoint cada 100 frames
                    if self.high_performance and len(frames_info) % 100 == 0:
                        self.save_checkpoint({'frames_info': frames_info}, checkpoint_name)
                
                count += 1
                pbar.update(1)
                
                # Si ya tenemos suficientes frames en modo alto rendimiento, salir
                if self.high_performance and len(frames_info) >= max_frames:
                    self.logger.info(f"Alcanzado límite de {max_frames} frames. Finalizando extracción.")
                    break
        
        video.release()
        
        # Guardar el checkpoint final
        self.save_checkpoint({'frames_info': frames_info}, checkpoint_name)
        
        self.logger.info(f"Se extrajeron {len(frames_info)} frames para análisis (de {total_frames} totales)")
        return frames_info
            
    def analyze_frame(self, frame_path: Path, retries: int = 2) -> Dict:
        """
        Analiza un frame con el modelo de Ollama para detectar kills y eventos interesantes.
        
        Args:
            frame_path: Ruta al archivo de imagen del frame
            retries: Número de reintentos si falla
            
        Returns:
            Dict con el análisis del frame
        """
        # Prompt enfocado específicamente en detectar kills y momentos graciosos
        prompt = """
        Analiza esta imagen de un videojuego y determina si muestra un "kill" (eliminación de un enemigo/jugador)
        o un momento gracioso. Busca específicamente:
        1. Eliminaciones de enemigos/jugadores (kills)
        2. Situaciones cómicas o ridículas
        3. Movimientos impresionantes
        
        Responde SOLO en formato JSON con esta estructura, limitando la descripción a 100 caracteres máximo:
        {
            "es_destacable": true/false,
            "tipo_momento": "kill"/"risa",
            "confianza": 0.0-1.0,
            "descripción": "breve descripción en 100 caracteres o menos"
        }
        """
        
        if self.high_performance:
            # En modo alto rendimiento, usamos un prompt aún más simple
            prompt = """
            ¿Muestra un "kill" o un momento gracioso en un videojuego? Responde solo JSON:
            {"es_destacable": true/false, "tipo_momento": "kill"/"risa", "confianza": 0.0-1.0, "descripción": "máximo 100 caracteres"}
            """
        
        # Preparar la solicitud para Ollama
        import base64

        # Función para realizar la solicitud con reintentos
        def make_request(retry_count=0):
            try:
                with open(frame_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False
                }
                
                response = requests.post(
                    "http://localhost:11434/api/generate", 
                    json=payload,
                    timeout=self.timeout  # Añadir timeout para evitar bloqueos
                )
                response.raise_for_status()
                return response.json().get("response", "")
            except (requests.exceptions.RequestException, IOError) as e:
                if retry_count < retries:
                    self.logger.warning(f"Error en solicitud. Reintentando ({retry_count+1}/{retries})...")
                    time.sleep(1)  # Esperar antes de reintentar
                    return make_request(retry_count + 1)
                else:
                    self.logger.error(f"Error al procesar frame {frame_path} después de {retries} intentos: {e}")
                    return None
        
        result = make_request()
        if result is None:
            return {
                "es_destacable": False, 
                "tipo_momento": "normal", 
                "confianza": 0.0, 
                "descripción": "Error de conexión"
            }
            
        # Intentar extraer JSON de la respuesta con métodos robustos
        try:
            # Limpiar la respuesta
            cleaned_result = result.replace('\n', ' ')
            
            # Eliminar bloques de código Markdown
            json_code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned_result)
            if json_code_block:
                cleaned_result = json_code_block.group(1)
            
            # Buscar cualquier objeto JSON en el texto
            json_match = re.search(r'\{.*\}', cleaned_result)
            if json_match:
                json_str = json_match.group(0)
                
                # Limpiar caracteres no válidos
                json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
                
                try:
                    parsed_result = json.loads(json_str)
                    
                    # Validar campos básicos - enfocados en kills y risas
                    tipo = parsed_result.get("tipo_momento", "normal")
                    
                    # Asegurarnos de que solo aceptamos "kill" o "risa"
                    if tipo not in ["kill", "risa"]:
                        # Convertir otros tipos a "normal" para filtrarlos más tarde
                        tipo = "normal"
                    
                    # Limitar descripción a 100 caracteres
                    descripcion = parsed_result.get("descripción", "Sin descripción")
                    if len(descripcion) > 100:
                        descripcion = descripcion[:97] + "..."
                    
                    return {
                        "es_destacable": parsed_result.get("es_destacable", False) and tipo in ["kill", "risa"],
                        "tipo_momento": tipo,
                        "confianza": max(0.0, min(1.0, parsed_result.get("confianza", 0.0))),
                        "descripción": descripcion
                    }
                except json.JSONDecodeError:
                    # Si falla, continuamos con la extracción manual
                    pass
                        
            # Extracción manual cuando todo lo demás falla
            es_destacable = "true" in cleaned_result.lower() and "es_destacable" in cleaned_result.lower()
            
            # Extraer tipo_momento - enfocado en kill o risa
            tipo_match = re.search(r'"tipo_momento":\s*"([^"]+)"', cleaned_result)
            tipo_momento = tipo_match.group(1) if tipo_match else "normal"
            
            # Asegurar que solo usamos "kill" o "risa"
            if tipo_momento not in ["kill", "risa"]:
                tipo_momento = "normal"
                es_destacable = False
            
            # Extraer confianza
            confianza_match = re.search(r'"confianza":\s*([\d.]+)', cleaned_result)
            confianza = float(confianza_match.group(1)) if confianza_match else 0.0
            
            # Extraer descripción y limitarla a 100 caracteres
            desc_match = re.search(r'"descripción":\s*"([^"]+(?:"[^"]+)*)"', cleaned_result)
            descripcion = desc_match.group(1) if desc_match else "Sin descripción"
            if len(descripcion) > 100:
                descripcion = descripcion[:97] + "..."
            
            return {
                "es_destacable": es_destacable and tipo_momento in ["kill", "risa"],
                "tipo_momento": tipo_momento,
                "confianza": max(0.0, min(1.0, confianza)),
                "descripción": descripcion
            }
                
        except Exception as e:
            # Capturar cualquier error y devolver un resultado seguro
            self.logger.error(f"Error al procesar respuesta: {e}")
            return {
                "es_destacable": False, 
                "tipo_momento": "normal", 
                "confianza": 0.0, 
                "descripción": "Error de procesamiento"
            }
    
    def process_frames_in_parallel(self, 
                                  frames_info: List[Tuple[float, Path]], 
                                  threshold: float,
                                  batch_size: int = 20  # Tamaño de lote más pequeño
                                 ) -> List[Dict]:
        """
        Procesa múltiples frames en paralelo con manejo de errores mejorado.
        
        Args:
            frames_info: Lista de tuplas (timestamp, frame_path)
            threshold: Umbral de confianza para considerar un momento destacable
            batch_size: Tamaño de los lotes para procesamiento
            
        Returns:
            Lista de momentos destacables
        """
        # Verificar si existe un checkpoint
        video_name = frames_info[0][1].parent.name if frames_info else "unknown"
        checkpoint_name = f"visual_highlights_{video_name}"
        checkpoint_data = self.load_checkpoint(checkpoint_name)
        if checkpoint_data:
            self.logger.info(f"Reanudando análisis de frames desde checkpoint")
            return checkpoint_data['visual_highlights']
            
        self.logger.info(f"Analizando {len(frames_info)} frames en paralelo con {self.max_workers} workers...")
        
        # Resultados de análisis de frames
        frames_analysis = []
        
        # Función para procesar un frame individual
        def process_frame(frame_data):
            try:
                timestamp, frame_path = frame_data
                analysis = self.analyze_frame(frame_path)
                return timestamp, analysis
            except Exception as e:
                self.logger.error(f"Error al procesar frame: {e}")
                return frame_data[0], {
                    "es_destacable": False, 
                    "tipo_momento": "normal", 
                    "confianza": 0.0, 
                    "descripción": f"Error: {str(e)}"
                }
        
        # Procesar frames en paralelo por lotes pequeños
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            total_batches = (len(frames_info) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                if self.interrupted:
                    self.logger.warning("Interrupción detectada. Guardando progreso...")
                    break
                    
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(frames_info))
                current_batch = frames_info[start_idx:end_idx]
                
                batch_desc = f"Lote {batch_idx+1}/{total_batches}"
                self.logger.info(f"Procesando {batch_desc} ({start_idx+1}-{end_idx} de {len(frames_info)})")
                
                # Ejecutar análisis en paralelo
                futures_to_frames = {
                    executor.submit(process_frame, frame_data): i 
                    for i, frame_data in enumerate(current_batch)
                }
                
                # Recolectar resultados usando tqdm para progreso
                with tqdm(total=len(futures_to_frames), desc=f"Analizando {batch_desc}") as pbar:
                    for future in concurrent.futures.as_completed(futures_to_frames):
                        try:
                            timestamp, analysis = future.result()
                            frames_analysis.append((timestamp, analysis))
                        except Exception as e:
                            self.logger.error(f"Error al obtener resultado: {e}")
                        finally:
                            pbar.update(1)
                
                # Guardar checkpoint después de cada lote
                if frames_analysis:
                    # Convertir a highlights para el checkpoint
                    current_highlights = self._frames_to_highlights(frames_analysis, threshold)
                    self.save_checkpoint({'visual_highlights': current_highlights}, checkpoint_name)
        
        # Ordenar por timestamp y convertir a highlights
        frames_analysis.sort(key=lambda x: x[0])
        visual_highlights = self._frames_to_highlights(frames_analysis, threshold)
        
        # Guardar checkpoint final
        self.save_checkpoint({'visual_highlights': visual_highlights}, checkpoint_name)
        
        return visual_highlights
    
    def _frames_to_highlights(self, frames_analysis: List[Tuple[float, Dict]], threshold: float) -> List[Dict]:
        """
        Convierte análisis de frames a highlights, filtrando solo kills y risas.
        
        Args:
            frames_analysis: Lista de tuplas (timestamp, análisis)
            threshold: Umbral de confianza mínimo
            
        Returns:
            Lista de highlights
        """
        highlights = []
        current_highlight = None
        
        for timestamp, analysis in frames_analysis:
            # Verificar que sea un momento destacable y sea kill o risa
            is_valid = (analysis["es_destacable"] and 
                      analysis["confianza"] >= threshold and 
                      analysis["tipo_momento"] in ["kill", "risa"])
            
            if is_valid:
                # Si no hay un highlight en curso, iniciar uno nuevo
                if current_highlight is None:
                    current_highlight = {
                        "start": timestamp,
                        "tipo": analysis["tipo_momento"],
                        "descripción": analysis["descripción"],
                        "confianza": analysis["confianza"]
                    }
                # Si es un momento muy destacado del mismo tipo, extender el highlight
                elif analysis["tipo_momento"] == current_highlight["tipo"]:
                    # Extender el highlight
                    pass
                # Si es un tipo diferente, finalizar el actual e iniciar uno nuevo
                else:
                    current_highlight["end"] = timestamp + 2  # Añadir 2 segundos después del momento
                    highlights.append(current_highlight)
                    current_highlight = {
                        "start": timestamp,
                        "tipo": analysis["tipo_momento"],
                        "descripción": analysis["descripción"],
                        "confianza": analysis["confianza"]
                    }
            elif current_highlight is not None:
                # Si hay un highlight en curso y este frame no es destacable, finalizarlo
                # Calculamos cuánto tiempo ha pasado desde el inicio del highlight
                highlight_duration = timestamp - current_highlight["start"]
                
                # Solo terminamos el highlight si ha durado al menos 2 segundos
                if highlight_duration >= 2:
                    # Añadimos 2 segundos más al final para capturar el desenlace
                    current_highlight["end"] = timestamp + 2
                    highlights.append(current_highlight)
                current_highlight = None
        
        # Si hay un highlight en curso al final, finalizarlo
        if current_highlight is not None and len(frames_analysis) > 0:
            current_highlight["end"] = frames_analysis[-1][0] + 2
            highlights.append(current_highlight)
        
        # Filtrar una vez más para asegurar que solo tenemos kill o risa
        return [h for h in highlights if h["tipo"] in ["kill", "risa"]]
    
    def merge_highlights(self, 
                         visual_highlights: List[Dict], 
                         audio_highlights: List[Dict],
                         max_gap: float = 3.0
                        ) -> List[Dict]:
        """
        Combina los momentos destacables visuales y de audio, filtrando solo kills y risas.
        
        Args:
            visual_highlights: Lista de momentos destacables visuales
            audio_highlights: Lista de momentos destacables de audio
            max_gap: Brecha máxima (en segundos) para considerar fusionar dos highlights
            
        Returns:
            Lista combinada de momentos destacables
        """
        # Verificar si existe un checkpoint
        checkpoint_name = f"combined_highlights"
        checkpoint_data = self.load_checkpoint(checkpoint_name)
        if checkpoint_data and checkpoint_data.get('visual_count') == len(visual_highlights) and \
           checkpoint_data.get('audio_count') == len(audio_highlights):
            self.logger.info(f"Reanudando desde checkpoint de highlights combinados")
            return checkpoint_data['combined_highlights']
        
        # Filtrar primero para asegurar que solo tenemos kill o risa
        filtered_visual = [h for h in visual_highlights if h["tipo"] in ["kill", "risa"]]
        filtered_audio = [h for h in audio_highlights if h["tipo"] in ["kill", "risa"]]
        
        # Combinar todos los highlights
        all_highlights = []
        
        # Añadir origen a cada highlight
        for h in filtered_visual:
            h_copy = h.copy()
            h_copy["origen"] = "visual"
            all_highlights.append(h_copy)
            
        for h in filtered_audio:
            h_copy = h.copy()
            h_copy["origen"] = "audio"
            # Asegúrate de que la transcripción esté disponible
            if "transcripción" not in h_copy:
                h_copy["transcripción"] = ""
            all_highlights.append(h_copy)
        
        # Ordenar por tiempo de inicio
        all_highlights.sort(key=lambda x: x["start"])
        
        # Fusionar highlights que se solapan o están muy cercanos
        merged_highlights = []
        current = None
        
        for h in all_highlights:
            if current is None:
                current = h.copy()
            elif h["start"] <= current["end"] + max_gap:
                # Fusionar highlights
                current["end"] = max(current["end"], h["end"])
                
                # Combinar descripciones si son diferentes (limitando a 100 caracteres)
                if h["descripción"] != current["descripción"]:
                    combined_desc = current["descripción"] + " + " + h["descripción"]
                    if len(combined_desc) > 100:
                        combined_desc = combined_desc[:97] + "..."
                    current["descripción"] = combined_desc
                
                # Combinar transcripciones si las hay
                if h.get("transcripción", "") and current.get("transcripción", ""):
                    current["transcripción"] = current["transcripción"] + " " + h["transcripción"]
                elif h.get("transcripción", ""):
                    current["transcripción"] = h["transcripción"]
                
                # Tomar el tipo con mayor confianza (manteniendo solo kill o risa)
                if h.get("confianza", 0.5) > current.get("confianza", 0.5) + 0.1:
                    current["tipo"] = h["tipo"]
                
                # Marcar como combinado
                if h["origen"] != current["origen"]:
                    current["origen"] = "combinado"
            else:
                merged_highlights.append(current)
                current = h.copy()
        
        # Añadir el último highlight en proceso
        if current is not None:
            merged_highlights.append(current)
        
        # Filtrar una vez más para asegurar que solo tenemos kill o risa
        filtered_merged = [h for h in merged_highlights if h["tipo"] in ["kill", "risa"]]
        
        # Guardar checkpoint
        self.save_checkpoint({
            'combined_highlights': filtered_merged,
            'visual_count': len(visual_highlights),
            'audio_count': len(audio_highlights)
        }, checkpoint_name)
        
        return filtered_merged
    
    def generate_title_description(self, 
                                  tipo: str, 
                                  descripcion: str,
                                  origen: str = "combinado",
                                  transcripcion: str = ""
                                 ) -> Dict[str, str]:
        """
        Genera un título y una descripción llamativa para el clip.
        El título tendrá 20 caracteres y la descripción 100 caracteres.

        Args:
            tipo: Tipo de momento destacado (kill o risa).
            descripcion: Descripción breve del evento.
            origen: Origen del clip (visual, audio o combinado)
            transcripcion: Transcripción del audio si está disponible

        Returns:
            Un diccionario con "titulo" y "descripcion" generados.
        """
        # En modo alto rendimiento, usamos plantillas predefinidas
        if self.high_performance:
            # Títulos predefinidos por tipo (exactamente 20 caracteres)
            titulos = {
                "kill": "¡Kill Impresionante!",
                "risa": "¡Momento Hilarante!"
            }
            
            # Descripciones predefinidas por tipo (exactamente 100 caracteres)
            descripciones = {
                "kill": "Eliminación perfecta que demuestra gran habilidad. Un momento de gloria que todo jugador desea conseguir.",
                "risa": "Situación divertida que provoca carcajadas. Uno de esos momentos únicos que hacen de los juegos algo especial."
            }
            
            # Usar el título y descripción predefinidos
            titulo = titulos.get(tipo, "¡Momento Destacado!")
            descripcion_gen = descripciones.get(tipo, "Un momento único que muestra lo mejor del juego. Situación especial que merece ser compartida.")
            
            return {
                "titulo": titulo,
                "descripcion": descripcion_gen
            }
        
        # En modo normal, generamos con IA
        prompt = f"""
        Eres un creador de contenido para videojuegos. 
        Genera un título llamativo y una descripción atractiva para un clip de videojuego basado en este evento:

        Tipo de clip: {tipo} (kill o momento gracioso)
        Descripción: {descripcion}
        Origen: {origen}
        {f'Transcripción: {transcripcion[:100]}' if transcripcion else ''}

        IMPORTANTE: El título debe tener EXACTAMENTE 20 caracteres.
        La descripción debe tener EXACTAMENTE 100 caracteres.

        Responde ÚNICAMENTE en formato JSON:
        {{
            "titulo": "Título de 20 caracteres",
            "descripcion": "Descripción de exactamente 100 caracteres, ni uno más ni uno menos, contando espacios y puntuación."
        }}
        """

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(
                "http://localhost:11434/api/generate", 
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json().get("response", "")

            # Extraer JSON con métodos robustos
            cleaned_result = result.replace('\n', ' ')
            
            # Buscar cualquier objeto JSON
            json_match = re.search(r'\{.*\}', cleaned_result)
            if json_match:
                json_str = json_match.group(0)
                json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
                
                try:
                    parsed_result = json.loads(json_str)
                    titulo = parsed_result.get("titulo", "")
                    descripcion = parsed_result.get("descripcion", "")
                    
                    # Ajustar longitud si es necesario
                    if len(titulo) > 20:
                        titulo = titulo[:20]
                    elif len(titulo) < 20:
                        titulo = titulo.ljust(20)
                        
                    if len(descripcion) > 100:
                        descripcion = descripcion[:100]
                    elif len(descripcion) < 100:
                        descripcion = descripcion.ljust(100)
                    
                    return {
                        "titulo": titulo,
                        "descripcion": descripcion
                    }
                except Exception:
                    pass
            
            # Si no se pudo extraer, usar plantillas predefinidas
            titulos = {
                "kill": "¡Kill Impresionante!",
                "risa": "¡Momento Hilarante!"
            }
            
            descripciones = {
                "kill": "Eliminación perfecta que demuestra gran habilidad. Un momento de gloria que todo jugador desea conseguir.",
                "risa": "Situación divertida que provoca carcajadas. Uno de esos momentos únicos que hacen de los juegos algo especial."
            }
            
            return {
                "titulo": titulos.get(tipo, "¡Momento Destacado!"),
                "descripcion": descripciones.get(tipo, "Un momento único que muestra lo mejor del juego. Situación especial que merece ser compartida.")
            }

        except Exception as e:
            self.logger.error(f"Error al generar título y descripción: {e}")
            return {
                "titulo": "¡" + tipo.capitalize() + " destacado!".ljust(20)[:20],
                "descripcion": f"Un clip de {tipo} impresionante que muestra un momento especial del juego que vale la pena compartir.".ljust(100)[:100]
            }
    
    def extract_clips(self, 
                      video_path: str, 
                      highlights: List[Dict], 
                      padding: float = 2
                     ) -> List[Tuple[Path, Path]]:
        """
        Extrae los clips destacados del video y genera un archivo .txt con el título y la descripción.

        Args:
            video_path: Ruta al archivo de video
            highlights: Lista de momentos destacables
            padding: Segundos adicionales antes y después del momento

        Returns:
            Lista de rutas a los clips generados
        """
        # Verificar si existe un checkpoint
        checkpoint_name = f"clips_{Path(video_path).stem}"
        checkpoint_data = self.load_checkpoint(checkpoint_name)
        if checkpoint_data and checkpoint_data.get('highlights_count') == len(highlights):
            self.logger.info(f"Reanudando extracción de clips desde checkpoint")
            return checkpoint_data['clips_paths']
            
        if not highlights:
            self.logger.warning("No se encontraron momentos destacables de tipo kill o risa en el video.")
            return []

        video_filename = Path(video_path).stem
        clips_paths = []
        
        # Extraer clips con progreso
        with tqdm(total=len(highlights), desc="Extrayendo clips") as pbar:
            for i, highlight in enumerate(highlights):
                if self.interrupted:
                    self.logger.warning("Interrupción detectada durante extracción de clips")
                    break
                    
                try:
                    start_time = max(0, highlight["start"] - padding)
                    end_time = highlight["end"] + padding
                    duration = end_time - start_time

                    tipo = highlight["tipo"]
                    descripcion = highlight["descripción"]
                    origen = highlight.get("origen", "visual")
                    transcripcion = highlight.get("transcripción", "")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Agregar origen e índice al nombre del archivo para identificar mejor los clips
                    output_filename = f"{video_filename}_{tipo}_{origen}_{i+1}_{timestamp}.mp4"
                    output_path = self.output_folder / output_filename

                    # Generar título y descripción
                    ia_text = self.generate_title_description(tipo, descripcion, origen, transcripcion)

                    # Crear el archivo de texto con el título y la descripción
                    txt_filename = f"{video_filename}_{tipo}_{origen}_{i+1}_{timestamp}.txt"
                    txt_path = self.output_folder / txt_filename

                    with open(txt_path, "w", encoding="utf-8") as txt_file:
                        txt_file.write(f"Título: {ia_text['titulo']}\n")
                        txt_file.write(f"Descripción: {ia_text['descripcion']}\n")
                        txt_file.write(f"Tipo de momento: {tipo}\n")
                        txt_file.write(f"Origen de detección: {origen}\n")
                        txt_file.write(f"Inicio: {start_time:.2f}s\n")
                        txt_file.write(f"Fin: {end_time:.2f}s\n")
                        txt_file.write(f"Descripción original: {descripcion}\n")
                        
                        if transcripcion:
                            # Limitar también la transcripción a 100 caracteres
                            transcripcion_short = transcripcion[:100]
                            txt_file.write(f"\nTranscripción: {transcripcion_short}\n")

                    # Usar FFmpeg para extraer el clip con manejo de errores
                    try:
                        cmd = [
                            "ffmpeg", "-i", video_path,
                            "-ss", str(start_time),
                            "-t", str(duration),
                            "-c:v", "libx264", "-c:a", "aac",
                            "-preset", "veryfast",  # Más rápido, sacrificando un poco de calidad
                            "-crf", "23",           # Calidad ligeramente inferior para velocidad
                            "-y",                   # Sobrescribir si existe
                            str(output_path)
                        ]

                        self.logger.info(f"Extrayendo clip {i+1}/{len(highlights)}: {start_time:.2f}s - {end_time:.2f}s")
                        
                        # Ejecutar FFmpeg con timeout
                        process = subprocess.Popen(
                            cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        # Esperar a que termine con timeout
                        try:
                            stdout, stderr = process.communicate(timeout=300)  # 5 minutos max
                            
                            if process.returncode != 0:
                                self.logger.error(f"Error al extraer clip {i+1}: FFmpeg código {process.returncode}")
                                self.logger.error(f"Error: {stderr}")
                                continue
                                
                            clips_paths.append((output_path, txt_path))
                            
                            # Guardar checkpoint después de cada clip
                            self.save_checkpoint({
                                'clips_paths': clips_paths,
                                'highlights_count': len(highlights)
                            }, checkpoint_name)
                            
                        except subprocess.TimeoutExpired:
                            self.logger.error(f"Timeout al extraer clip {i+1}")
                            process.kill()
                    except Exception as e:
                        self.logger.error(f"Error al ejecutar FFmpeg: {e}")
                except Exception as e:
                    self.logger.error(f"Error al preparar clip {i+1}: {e}")
                finally:
                    pbar.update(1)
                
        # Guardar checkpoint final
        self.save_checkpoint({
            'clips_paths': clips_paths,
            'highlights_count': len(highlights)
        }, checkpoint_name)
            
        return clips_paths

    def cleanup(self, keep_checkpoints: bool = True) -> None:
        """
        Elimina los archivos temporales pero conserva los checkpoints.
        
        Args:
            keep_checkpoints: Si es True, conserva los checkpoints
        """
        try:
            files_count = 0
            for file in self.temp_folder.glob("*"):
                file.unlink()
                files_count += 1
            self.logger.info(f"Limpieza completada: {files_count} archivos temporales eliminados")
            
            if not keep_checkpoints:
                checkpoints_count = 0
                for file in self.checkpoint_folder.glob("*.pkl"):
                    file.unlink()
                    checkpoints_count += 1
                self.logger.info(f"Se eliminaron {checkpoints_count} archivos de checkpoint")
        except Exception as e:
            self.logger.error(f"Error durante la limpieza: {e}")
        
    def process_video(self, 
                      video_path: str, 
                      visual_sample_rate: float = 0.5, 
                      audio_segment_duration: float = 5.0,
                      visual_threshold: float = 0.7,
                      audio_threshold: float = 0.6,
                      padding: float = 2,
                      adaptive_sampling: bool = True,
                      skip_segments: List[Tuple[float, float]] = None,
                      resume: bool = True,
                      max_frames: int = 1000
                     ) -> List[Tuple[Path, Path]]:
        """
        Procesa un video completo para extraer clips destacables basados en visual y audio.
        Incluye soporte para resumir procesamiento y optimizaciones de rendimiento.
        
        Args:
            video_path: Ruta al archivo de video
            visual_sample_rate: Cantidad de frames a analizar por segundo
            audio_segment_duration: Duración de los segmentos de audio a analizar
            visual_threshold: Umbral de confianza para considerar un momento visual destacable
            audio_threshold: Umbral de confianza para considerar un momento de audio destacable
            padding: Segundos adicionales antes y después del momento
            adaptive_sampling: Si es True, utiliza muestreo adaptativo
            skip_segments: Lista de segmentos (inicio, fin) a omitir
            resume: Si es True, intenta reanudar desde checkpoint
            max_frames: Número máximo de frames a analizar
            
        Returns:
            Lista de rutas a los clips generados
        """
        try:
            # Verificar si hay un checkpoint completo para evitar reprocesamiento
            final_checkpoint = f"final_{Path(video_path).stem}"
            final_data = self.load_checkpoint(final_checkpoint) if resume else None
            if final_data and 'clips_paths' in final_data:
                self.logger.info(f"Reanudando desde checkpoint final. Se omite procesamiento.")
                return final_data['clips_paths']
                
            start_time = time.time()
            
            # Extraer frames
            frames_info = self.extract_frames(
                video_path, 
                visual_sample_rate, 
                adaptive_sampling,
                max_frames=max_frames if self.high_performance else max_frames * 2
            )
            
            # Procesar audio según la configuración
            if self.use_audio_analyzer:
                # Usar AudioAnalyzer directamente
                self.logger.info("Procesando audio con AudioAnalyzer...")
                audio_highlights = self.audio_analyzer.process_audio(
                    video_path, 
                    segment_duration=audio_segment_duration,
                    threshold=audio_threshold
                )
            else:
                # Usar procesamiento básico de audio
                # (Este flujo debería ser raro ya que el procesamiento con AudioAnalyzer es mejor)
                self.logger.info("Procesando audio con método básico...")
                audio_path = self.extract_audio(video_path)
                audio_segments = self.split_audio(audio_path, audio_segment_duration)
                audio_highlights = []  # Por defecto vacío en modo rápido
            
            # Omitir segmentos si es necesario
            if skip_segments:
                # Filtrar frames
                frames_info = [
                    (ts, path) for ts, path in frames_info
                    if not any(start <= ts <= end for start, end in skip_segments)
                ]
                
                # Filtrar highlights de audio
                audio_highlights = [
                    h for h in audio_highlights
                    if not any(start <= h["start"] <= end for start, end in skip_segments)
                ]
                
                self.logger.info(f"Se omitieron segmentos en los rangos especificados")
            
            # Procesar frames en paralelo
            visual_highlights = self.process_frames_in_parallel(
                frames_info, 
                visual_threshold,
                batch_size=20 if self.high_performance else 60
            )
            
            self.logger.info(f"Se encontraron {len(visual_highlights)} momentos visuales destacables.")
            self.logger.info(f"Se encontraron {len(audio_highlights)} momentos de audio destacables.")
            
            # Combinar highlights
            combined_highlights = self.merge_highlights(visual_highlights, audio_highlights)
            self.logger.info(f"Se combinaron en {len(combined_highlights)} momentos destacables.")
            
            # Mostrar información de los highlights combinados
            for i, h in enumerate(combined_highlights):
                transcription_preview = h.get("transcripción", "")[:30] + "..." if h.get("transcripción", "") else "No disponible"
                self.logger.info(f"Highlight {i+1}: {h['tipo']} (origen: {h.get('origen', 'visual')}) - " 
                               f"{h['start']:.2f}s a {h['end']:.2f}s - {h['descripción']}")
                self.logger.info(f"  Transcripción: {transcription_preview}")
                
            # Extraer clips
            clips = self.extract_clips(video_path, combined_highlights, padding)
            
            end_time = time.time()
            self.logger.info(f"Proceso completado en {end_time - start_time:.2f} segundos")
            
            # Guardar checkpoint final
            self.save_checkpoint({'clips_paths': clips}, final_checkpoint)
            
            return clips
        except KeyboardInterrupt:
            self.logger.warning("Proceso interrumpido por el usuario.")
            return []
        finally:
            # Mantener checkpoints para posible reanudación
            self.cleanup(keep_checkpoints=True)

def main():
    """
    Función principal para ejecutar el script desde la línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Extractor de clips de videojuegos - solo kills y momentos graciosos")
    parser.add_argument("video_path", help="Ruta al archivo de video MP4")
    parser.add_argument("--model", default="llava:latest", help="Modelo de Ollama a utilizar (por defecto: llava:latest)")
    parser.add_argument("--temp", default="temp", help="Carpeta temporal")
    parser.add_argument("--output", default="clips", help="Carpeta donde guardar los clips")
    parser.add_argument("--checkpoints", default="checkpoints", help="Carpeta para guardar checkpoints")
    
    # Opciones de rendimiento
    parser.add_argument("--high-performance", action="store_true", help="Modo de alto rendimiento (más rápido pero menos preciso)")
    parser.add_argument("--max-frames", type=int, default=1000, help="Máximo número de frames a analizar")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout para peticiones a Ollama (segundos)")
    parser.add_argument("--workers", type=int, default=4, help="Número de workers para procesamiento paralelo")
    
    # Opciones de análisis
    parser.add_argument("--visual-rate", type=float, default=0.5, help="Frames a analizar por segundo")
    parser.add_argument("--audio-duration", type=float, default=5.0, help="Duración de segmentos de audio (segundos)")
    parser.add_argument("--visual-threshold", type=float, default=0.7, help="Umbral para momentos visuales")
    parser.add_argument("--audio-threshold", type=float, default=0.6, help="Umbral para momentos de audio")
    parser.add_argument("--padding", type=float, default=2.0, help="Segundos extra al inicio/fin de clips")
    
    # Opciones de configuración
    parser.add_argument("--no-adaptive", action="store_true", help="Desactivar muestreo adaptativo")
    parser.add_argument("--no-audio-analyzer", action="store_true", help="No usar AudioAnalyzer para audio")
    parser.add_argument("--no-resume", action="store_true", help="No reanudar desde checkpoints")
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Nivel de logging")
    
    # Opciones de modo
    parser.add_argument("--audio-only", action="store_true", help="Analizar solo el audio del video")
    parser.add_argument("--visual-only", action="store_true", help="Analizar solo el contenido visual")
    
    # Opciones de filtrado
    parser.add_argument("--skip-start", type=float, default=None, help="Tiempo de inicio a omitir (segundos)")
    parser.add_argument("--skip-end", type=float, default=None, help="Tiempo de fin a omitir (segundos)")
    
    args = parser.parse_args()
    
    # Preparar segmentos a omitir
    skip_segments = []
    if args.skip_start is not None and args.skip_end is not None:
        skip_segments.append((args.skip_start, args.skip_end))
    
    # Crear el extractor
    extractor = VideoClipExtractor(
        model_name=args.model,
        temp_folder=args.temp,
        output_folder=args.output,
        checkpoint_folder=args.checkpoints,
        max_workers=args.workers,
        timeout=args.timeout,
        high_performance=args.high_performance,
        use_audio_analyzer=not args.no_audio_analyzer,
        log_level=args.log
    )
    
    if args.audio_only:
        # Solo analizar audio (enfocado en risas)
        if not args.no_audio_analyzer:
            # Usar AudioAnalyzer directamente
            analyzer = AudioAnalyzer(
                temp_folder=args.temp,
                max_workers=args.workers,
                log_level=args.log
            )
            audio_highlights = analyzer.process_audio(
                args.video_path, 
                segment_duration=args.audio_duration,
                threshold=args.audio_threshold
            )
        else:
            # Usar procesamiento básico de audio
            audio_highlights = extractor.audio_analyzer.process_audio(
                args.video_path,
                segment_duration=args.audio_duration,
                threshold=args.audio_threshold
            )
            
        clips = extractor.extract_clips(args.video_path, audio_highlights, args.padding)
    elif args.visual_only:
        # Solo analizar visual (enfocado en kills y momentos graciosos)
        frames_info = extractor.extract_frames(
            args.video_path, 
            args.visual_rate, 
            not args.no_adaptive,
            max_frames=args.max_frames
        )
        visual_highlights = extractor.process_frames_in_parallel(frames_info, args.visual_threshold)
        clips = extractor.extract_clips(args.video_path, visual_highlights, args.padding)
    else:
        # Análisis completo (visual + audio)
        clips = extractor.process_video(
            args.video_path, 
            args.visual_rate, 
            args.audio_duration,
            args.visual_threshold,
            args.audio_threshold,
            args.padding,
            not args.no_adaptive,
            skip_segments,
            not args.no_resume,
            args.max_frames
        )
    
    print(f"\nProceso completado. Se generaron {len(clips)} clips:")
    for clip, txt in clips:
        print(f"- Video: {clip}")
        print(f"- Texto: {txt}")
    
    if args.high_performance:
        print("\nNota: Se utilizó el modo de alto rendimiento.")
        print("Los resultados pueden ser menos precisos pero el procesamiento es significativamente más rápido.")
    
    if len(clips) == 0:
        print("\nNo se generaron clips de kills o momentos graciosos. Posibles razones:")
        print("- No se encontraron momentos de este tipo específico en el video")
        print("- El proceso fue interrumpido")
        print("- Hubo errores durante el procesamiento (revisa videoclip_extractor.log)")
        print("\nSugerencias:")
        print("- Ajusta los umbrales (--visual-threshold, --audio-threshold)")
        print("- Aumenta el muestreo visual (--visual-rate)")
        print("- Prueba con otro video de videojuegos con más acción")

if __name__ == "__main__":
    main()
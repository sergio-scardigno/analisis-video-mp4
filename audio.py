"""
Solución alternativa para los errores de Whisper en el VideoClipExtractor.
Este script puede reemplazar o complementar el sistema de análisis de audio.
"""

import librosa
import numpy as np
import os
import soundfile as sf
import subprocess
import json
from pathlib import Path
import logging
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any

# Función auxiliar para convertir tipos numpy a tipos estándar de Python
def numpy_to_python(obj):
    """Convierte tipos de NumPy a tipos nativos de Python para serialización JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

class AudioAnalyzer:
    """
    Clase para analizar audio sin depender de Whisper,
    utilizando características acústicas básicas para detectar momentos destacables.
    """
    
    def __init__(self, 
                temp_folder: str = "temp",
                max_workers: int = 4,
                log_level: str = "INFO"):
        """
        Inicializa el analizador de audio.
        
        Args:
            temp_folder: Carpeta para archivos temporales
            max_workers: Número máximo de workers para procesamiento paralelo
            log_level: Nivel de logging
        """
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        self.temp_folder = Path(temp_folder)
        self.max_workers = max_workers
        
        # Asegurarse de que la carpeta temporal existe
        self.temp_folder.mkdir(exist_ok=True)
    
    def extract_audio(self, video_path: str) -> Path:
        """
        Extrae el audio del video utilizando FFmpeg con parámetros optimizados.
        
        Args:
            video_path: Ruta al archivo de video
            
        Returns:
            Ruta al archivo de audio extraído
        """
        self.logger.info(f"Extrayendo audio optimizado del video: {video_path}")
        audio_path = self.temp_folder / f"{Path(video_path).stem}_audio_opt.wav"
        
        # Usar FFmpeg con parámetros optimizados para calidad de análisis
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",                    # No video
            "-acodec", "pcm_s16le",   # Formato PCM 16-bit
            "-ar", "16000",           # Sample rate 16kHz
            "-ac", "1",               # Mono
            "-af", "dynaudnorm",      # Normalización dinámica de audio
            "-y",                     # Sobrescribir si existe
            str(audio_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"Audio extraído correctamente: {audio_path}")
            return audio_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error al extraer audio: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Error al extraer audio con FFmpeg: {str(e)}")
    
    def split_audio(self, audio_path: Path, segment_duration: float = 5.0,
                   overlap: float = 1.0) -> List[Tuple[float, Path]]:
        """
        Divide el audio en segmentos cortos con superposición.
        
        Args:
            audio_path: Ruta al archivo de audio
            segment_duration: Duración de cada segmento en segundos
            overlap: Superposición entre segmentos en segundos
            
        Returns:
            Lista de tuplas (timestamp_inicio, ruta_segmento)
        """
        self.logger.info(f"Dividiendo audio en segmentos de {segment_duration}s con {overlap}s de superposición")
        
        # Obtener información del audio
        audio_info = sf.info(audio_path)
        total_duration = audio_info.duration
        samplerate = audio_info.samplerate
        
        # Calcular paso entre segmentos
        step = segment_duration - overlap
        
        # Lista para almacenar información de segmentos
        segments = []
        
        # Dividir el audio en segmentos
        for start_time in np.arange(0, total_duration - (segment_duration / 2), step):
            end_time = min(start_time + segment_duration, total_duration)
            actual_duration = end_time - start_time
            
            # Solo procesar segmentos de al menos 1 segundo
            if actual_duration < 1.0:
                continue
                
            # Calcular frames
            start_frame = int(start_time * samplerate)
            num_frames = int(actual_duration * samplerate)
            
            # Leer el segmento de audio
            try:
                audio_data, sr = sf.read(audio_path, start=start_frame, frames=num_frames)
                
                # Verificar que el segmento tenga contenido significativo
                if np.max(np.abs(audio_data)) < 0.01:
                    self.logger.debug(f"Segmento en {start_time:.2f}s tiene nivel muy bajo, omitiendo")
                    continue
                    
                # Guardar en archivo temporal
                segment_path = self.temp_folder / f"audio_seg_opt_{start_time:.2f}.wav"
                sf.write(segment_path, audio_data, sr)
                
                segments.append((float(start_time), segment_path))  # Convertir a float estándar
            except Exception as e:
                self.logger.warning(f"Error al procesar segmento en {start_time:.2f}s: {e}")
        
        self.logger.info(f"Se crearon {len(segments)} segmentos de audio válidos")
        return segments
    
    def analyze_segment(self, segment_path: Path) -> Dict[str, Any]:
        """
        Analiza un segmento de audio utilizando características acústicas.
        
        Args:
            segment_path: Ruta al archivo de audio del segmento
            
        Returns:
            Diccionario con resultados del análisis
        """
        try:
            # Cargar audio
            y, sr = librosa.load(str(segment_path), sr=None)
            
            # Verificar que haya contenido
            if len(y) < sr * 0.5 or np.max(np.abs(y)) < 0.01:
                return {
                    "es_destacable": False,
                    "tipo_momento": "normal",
                    "confianza": 0.0,
                    "descripción": "Audio sin contenido significativo"
                }
            
            # Extraer características
            # 1. Energía (volumen)
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))  # Convertir a float estándar
            rms_std = float(np.std(rms))    # Convertir a float estándar
            rms_max = float(np.max(rms))    # Convertir a float estándar
            
            # 2. Tasa de cruces por cero (ZCR) - útil para detectar ruidos bruscos
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))  # Convertir a float estándar
            zcr_std = float(np.std(zcr))    # Convertir a float estándar
            
            # 3. Centroide espectral - relacionado con el "brillo" del sonido
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            cent_mean = float(np.mean(spec_cent))  # Convertir a float estándar
            cent_std = float(np.std(spec_cent))    # Convertir a float estándar
            
            # 4. Ancho de banda espectral
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            bw_mean = float(np.mean(spec_bw))  # Convertir a float estándar
            
            # 5. Roll-off espectral - frecuencia por debajo de la cual se concentra X% de la energía
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            rolloff_mean = float(np.mean(rolloff))  # Convertir a float estándar
            rolloff_std = float(np.std(rolloff))    # Convertir a float estándar
            
            # 6. Contraste espectral - diferencia entre picos y valles
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = float(np.mean(np.mean(contrast, axis=1)))  # Convertir a float estándar
            
            # 7. Onsets - detección de inicios de sonidos
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_count = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))
            onset_density = float(onset_count / (len(y) / sr))  # Convertir a float estándar
            
            # === Detección de eventos concretos ===
            
            # Detector de risas - alta variabilidad en energía + alto ZCR + componentes específicos espectrales
            is_laughter = (rms_std > 0.03 and zcr_mean > 0.1 and 
                          cent_mean > 1800 and onset_density > 3)
            
            # Detector de victoria/celebración - alto volumen + alto contraste + picos fuertes
            is_victory = (rms_max > 0.2 and contrast_mean > 20 and 
                         rolloff_std > 1000 and onset_density > 2)
            
            # Detector de emociones intensas - alto volumen + alto ZCR + alta variabilidad espectral
            is_excitement = (rms_mean > 0.15 and zcr_std > 0.05 and 
                            cent_std > 500 and rolloff_std > 800)
            
            # Detector de música intensa - patrones rítmicos claros + espectro amplio
            is_music = (bw_mean > 2000 and onset_density > 2.5 and
                       contrast_mean > 15 and rms_std < 0.05)  # La música tiene volumen más constante
            
            # Detector de discurso enfático - volumen moderado + ZCR moderado + centroide bajo
            is_speech = (0.05 < rms_mean < 0.2 and 0.05 < zcr_mean < 0.15 and
                        cent_mean < 1500 and onset_density < 3)
            
            # Determinar tipo principal y confianza
            results = [
                ("risa", is_laughter, 0.7 + min(rms_std * 5, 0.25)),
                ("victoria", is_victory, 0.7 + min(rms_max * 1.5, 0.25)),
                ("emocionante", is_excitement, 0.65 + min(zcr_std * 3, 0.3)),
                ("música", is_music, 0.6 + min(onset_density / 10, 0.3)),
                ("discurso", is_speech, 0.6 + min(rms_mean * 2, 0.2))
            ]
            
            # Seleccionar el tipo con mayor probabilidad
            results = [(tipo, flag, float(conf)) for tipo, flag, conf in results if flag]  # Convertir confianza a float estándar
            
            if not results:
                return {
                    "es_destacable": False,
                    "tipo_momento": "normal",
                    "confianza": 0.0,
                    "descripción": "Audio sin eventos destacables detectados"
                }
            
            # Ordenar por confianza descendente
            results.sort(key=lambda x: x[2], reverse=True)
            tipo_momento, _, confianza = results[0]
            
            # Generar descripción basada en el tipo
            descripciones = {
                "risa": "Risas o momentos de diversión detectados en el audio",
                "victoria": "Celebración o momento de victoria con gritos de alegría",
                "emocionante": "Momento de alta emoción o sorpresa con expresiones intensas",
                "música": "Música intensa o climática que marca un momento importante",
                "discurso": "Discurso enfático o importante con tono destacado"
            }
            
            descripcion = descripciones.get(tipo_momento, f"Audio con características de {tipo_momento}")
            
            # Asegurarse de que todos los valores son tipos Python estándar
            return {
                "es_destacable": bool(confianza > 0.65),
                "tipo_momento": str(tipo_momento),
                "confianza": float(confianza),
                "descripción": str(descripcion)
            }
            
        except Exception as e:
            self.logger.error(f"Error al analizar segmento de audio {segment_path}: {e}")
            return {
                "es_destacable": False,
                "tipo_momento": "normal",
                "confianza": 0.0,
                "descripción": f"Error al analizar audio: {str(e)}"
            }
    
    def process_audio(self, video_path: str, segment_duration: float = 5.0,
                     threshold: float = 0.65) -> List[Dict]:
        """
        Procesa el audio completo del video para detectar momentos destacables.
        
        Args:
            video_path: Ruta al archivo de video
            segment_duration: Duración de cada segmento en segundos
            threshold: Umbral de confianza mínimo para considerar un momento destacable
            
        Returns:
            Lista de momentos destacables
        """
        try:
            # Extraer audio del video
            audio_path = self.extract_audio(video_path)
            
            # Dividir en segmentos
            segments = self.split_audio(audio_path, segment_duration)
            
            # Procesar segmentos en paralelo
            return self.process_segments(segments, threshold)
            
        except Exception as e:
            self.logger.error(f"Error al procesar audio del video: {e}")
            return []
    
    def process_segments(self, segments: List[Tuple[float, Path]], 
                        threshold: float = 0.65) -> List[Dict]:
        """
        Procesa segmentos de audio en paralelo para detectar momentos destacables.
        
        Args:
            segments: Lista de tuplas (timestamp, segment_path)
            threshold: Umbral de confianza mínimo
            
        Returns:
            Lista de momentos destacables
        """
        self.logger.info(f"Analizando {len(segments)} segmentos de audio...")
        
        # Resultados de análisis
        segments_analysis = []
        
        # Función para procesar un segmento
        def process_segment(segment_data):
            try:
                start_time, segment_path = segment_data
                
                # Verificar que el archivo existe
                if not os.path.exists(segment_path):
                    return None
                
                # Analizar segmento
                analysis = self.analyze_segment(segment_path)
                
                # Obtener duración del segmento
                segment_info = sf.info(segment_path)
                end_time = float(start_time + segment_info.duration)  # Convertir a float estándar
                
                return float(start_time), end_time, analysis
                
            except Exception as e:
                self.logger.error(f"Error al procesar segmento: {e}")
                return None
        
        # Procesar segmentos en paralelo por lotes
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            batch_size = min(self.max_workers * 5, len(segments))
            
            for batch_start in range(0, len(segments), batch_size):
                batch_end = min(batch_start + batch_size, len(segments))
                current_batch = segments[batch_start:batch_end]
                
                self.logger.info(f"Procesando lote de audio {batch_start+1}-{batch_end} de {len(segments)}")
                
                # Enviar trabajos
                futures = [executor.submit(process_segment, segment) for segment in current_batch]
                
                # Recolectar resultados
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    if i % 5 == 0 or i == len(current_batch) - 1:
                        self.logger.info(f"Progreso: {i+1}/{len(current_batch)} ({(i+1)/len(current_batch)*100:.1f}%)")
                    
                    try:
                        result = future.result()
                        if result:
                            segments_analysis.append(result)
                    except Exception as e:
                        self.logger.error(f"Error al obtener resultado: {e}")
        
        # Ordenar por tiempo
        segments_analysis.sort(key=lambda x: x[0])
        
        # Filtrar por umbral y convertir a formato de highlights
        highlights = []
        
        for start, end, analysis in segments_analysis:
            if analysis["es_destacable"] and analysis["confianza"] >= threshold:
                # Asegurarse de que todos los valores son tipos Python estándar
                highlights.append({
                    "start": float(start),
                    "end": float(end),
                    "tipo": str(analysis["tipo_momento"]),
                    "descripción": str(analysis["descripción"]),
                    "confianza": float(analysis["confianza"]),
                    "origen": "audio"
                })
        
        # Combinar highlights consecutivos del mismo tipo
        if highlights:
            combined = [highlights[0]]
            
            for h in highlights[1:]:
                prev = combined[-1]
                
                # Si es del mismo tipo y está cerca, combinar
                if h["tipo"] == prev["tipo"] and h["start"] - prev["end"] < 2.0:
                    prev["end"] = h["end"]
                    prev["confianza"] = max(prev["confianza"], h["confianza"])
                else:
                    combined.append(h)
            
            highlights = combined
        
        self.logger.info(f"Se detectaron {len(highlights)} momentos destacables en el audio")
        
        # Convertir valores NumPy a Python estándar para serialización JSON
        return [numpy_to_python(h) for h in highlights]


# Función para serialización personalizada
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Ejemplo de uso
if __name__ == "__main__":
    import argparse
    import time
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Analizador de audio avanzado sin Whisper")
    parser.add_argument("video_path", help="Ruta al archivo de video a analizar")
    parser.add_argument("--duration", type=float, default=5.0, 
                      help="Duración de los segmentos de audio (por defecto: 5.0s)")
    parser.add_argument("--threshold", type=float, default=0.65,
                      help="Umbral de confianza para detectar momentos destacables (por defecto: 0.65)")
    parser.add_argument("--workers", type=int, default=4,
                      help="Número de workers para procesamiento paralelo (por defecto: 4)")
    parser.add_argument("--output", default="resultados_audio.json",
                      help="Archivo de salida para guardar los resultados (por defecto: resultados_audio.json)")
    
    args = parser.parse_args()
    
    # Medir tiempo
    start_time = time.time()
    
    # Crear analizador
    analyzer = AudioAnalyzer(max_workers=args.workers)
    
    # Procesar audio
    highlights = analyzer.process_audio(
        args.video_path, 
        segment_duration=args.duration,
        threshold=args.threshold
    )
    
    # Guardar resultados usando el encoder personalizado
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(highlights, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)
    
    # Mostrar resultados
    print(f"\nSe encontraron {len(highlights)} momentos destacables:")
    for i, h in enumerate(highlights):
        print(f"{i+1}. {h['tipo']} ({h['confianza']:.2f}): {h['start']:.2f}s - {h['end']:.2f}s")
        print(f"   Descripción: {h['descripción']}")
    
    # Mostrar tiempo total
    print(f"\nTiempo total de procesamiento: {time.time() - start_time:.2f} segundos")
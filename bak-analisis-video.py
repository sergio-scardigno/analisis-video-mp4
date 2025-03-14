import cv2
import numpy as np
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import json
import argparse
import requests
import logging
from typing import List, Dict, Tuple, Optional

class VideoClipExtractor:
    def __init__(self, 
                 model_name: str = "llava:latest", 
                 temp_folder: str = "temp", 
                 output_folder: str = "clips",
                 log_level: str = "INFO"):
        """
        Inicializa el extractor de clips de video con análisis de IA.
        
        Args:
            model_name: Nombre del modelo de Ollama a utilizar
            temp_folder: Carpeta para archivos temporales
            output_folder: Carpeta donde se guardarán los clips
            log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.model_name = model_name
        self.temp_folder = Path(temp_folder)
        self.output_folder = Path(output_folder)
        
        # Crear carpetas si no existen
        self.temp_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        
        # Verificar que Ollama esté funcionando
        self._check_ollama()
    
    def _check_ollama(self) -> None:
        """Verifica que Ollama esté funcionando y el modelo esté disponible."""
        try:
            # Verificar que Ollama esté corriendo
            response = requests.get("http://localhost:11434/api/tags")
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
    
    def extract_frames(self, 
                       video_path: str, 
                       sample_rate: float = 1.0
                      ) -> List[Tuple[float, Path]]:
        """
        Extrae frames del video para analizar.
        
        Args:
            video_path: Ruta al archivo de video
            sample_rate: Cantidad de frames a extraer por segundo
            
        Returns:
            Lista de tuplas (timestamp, frame_path)
        """
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
        
        # Calcular cada cuántos frames guardar uno para análisis
        frame_interval = max(1, int(fps / sample_rate))
        
        frames_info = []
        count = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            # Procesar solo los frames según el intervalo
            if count % frame_interval == 0:
                timestamp = count / fps
                frame_path = self.temp_folder / f"frame_{count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames_info.append((timestamp, frame_path))
                
            count += 1
            
            # Mostrar progreso
            if count % 100 == 0:
                self.logger.info(f"Procesando frames: {count}/{total_frames} ({count/total_frames*100:.1f}%)")
        
        video.release()
        return frames_info
    
    def analyze_frame(self, frame_path: Path) -> Dict:
        """
        Analiza un frame con el modelo de Ollama para detectar eventos interesantes.
        
        Args:
            frame_path: Ruta al archivo de imagen del frame
            
        Returns:
            Dict con el análisis del frame
        """
        prompt = """
        Analiza esta imagen de un video y determina si muestra un momento destacable.
        Busca específicamente:
        1. Momentos de alta acción
        2. Eventos únicos o sorprendentes
        3. Transiciones dramáticas
        4. Momentos emocionantes o relevantes
        
        Responde en formato JSON con la siguiente estructura:
        {
            "es_destacable": true/false,
            "tipo_momento": "acción"/"evento"/"drama"/"emocionante",
            "confianza": 0.0-1.0,
            "descripción": "breve descripción de lo que ocurre"
        }
        """
        
        # Preparar la solicitud para Ollama
        import base64

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
            
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()
            result = response.json().get("response", "")
            
            # Extraer solo el JSON de la respuesta
            try:
                # Buscar el primer '{' y el último '}'
                start_idx = result.find('{')
                end_idx = result.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = result[start_idx:end_idx]
                    parsed_result = json.loads(json_str)
                    
                    # Validar que todos los campos esperados estén presentes
                    return {
                        "es_destacable": parsed_result.get("es_destacable", False),
                        "tipo_momento": parsed_result.get("tipo_momento", "normal"),
                        "confianza": max(0.0, min(1.0, parsed_result.get("confianza", 0.0))),
                        "descripción": parsed_result.get("descripción", "Sin descripción")
                    }
                else:
                    self.logger.warning(f"No se pudo extraer JSON de la respuesta: {result}")
                    return {
                        "es_destacable": False, 
                        "tipo_momento": "normal", 
                        "confianza": 0.0, 
                        "descripción": "No se pudo analizar"
                    }
            except json.JSONDecodeError:
                self.logger.error(f"Error al analizar la respuesta JSON: {result}")
                return {
                    "es_destacable": False, 
                    "tipo_momento": "normal", 
                    "confianza": 0.0, 
                    "descripción": "Error de formato"
                }
                
        except (requests.exceptions.RequestException, IOError) as e:
            self.logger.error(f"Error al procesar frame {frame_path}: {e}")
            return {
                "es_destacable": False, 
                "tipo_momento": "normal", 
                "confianza": 0.0, 
                "descripción": f"Error de procesamiento: {str(e)}"
            }
    
    def find_highlights(self, 
                        video_path: str, 
                        sample_rate: float = 1.0, 
                        threshold: float = 0.7
                       ) -> List[Dict]:
        """
        Encuentra momentos destacables en el video.
        
        Args:
            video_path: Ruta al archivo de video
            sample_rate: Cantidad de frames a analizar por segundo
            threshold: Umbral de confianza mínimo para considerar un momento destacable
            
        Returns:
            Lista de momentos destacables (timestamp, duración, tipo, descripción)
        """
        self.logger.info(f"Analizando video: {video_path}")
        frames_info = self.extract_frames(video_path, sample_rate)
        
        highlights = []
        current_highlight = None
        
        self.logger.info(f"Analizando {len(frames_info)} frames con IA...")
        
        for i, (timestamp, frame_path) in enumerate(frames_info):
            self.logger.debug(f"Analizando frame {i+1}/{len(frames_info)} ({timestamp:.2f}s)")
            analysis = self.analyze_frame(frame_path)
            
            if analysis["es_destacable"] and analysis["confianza"] >= threshold:
                # Si no hay un highlight en curso, iniciar uno nuevo
                if current_highlight is None:
                    current_highlight = {
                        "start": timestamp,
                        "tipo": analysis["tipo_momento"],
                        "descripción": analysis["descripción"]
                    }
                # Si es un momento muy destacado, considerar finalizar clip
                elif analysis["tipo_momento"] in ["evento", "drama", "emocionante"]:
                    current_highlight["end"] = timestamp + 3  # Añadir 3 segundos después del momento
                    highlights.append(current_highlight)
                    current_highlight = None
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
        
        # Si hay un highlight en curso al final del video, finalizarlo
        if current_highlight is not None:
            current_highlight["end"] = frames_info[-1][0] + 2
            highlights.append(current_highlight)
        
        return highlights
    
    def generate_title_description(self, 
                                   tipo: str, 
                                   descripcion: str
                                  ) -> Dict[str, str]:
        """
        Genera un título y una descripción llamativa para el clip usando Ollama.

        Args:
            tipo: Tipo de momento destacado.
            descripcion: Descripción breve del evento.

        Returns:
            Un diccionario con "titulo" y "descripcion" generados.
        """
        prompt = f"""
        Eres un creador de contenido para plataformas como YouTube y TikTok. 
        Genera un título llamativo y una descripción atractiva para un clip de video basado en este evento:

        Tipo de clip: {tipo}
        Descripción: {descripcion}

        El título debe ser corto, llamativo y atractivo. 
        La descripción debe expandir un poco lo que sucede, con algo de emoción.

        Formato de respuesta en JSON:
        {{
            "titulo": "Aquí el título generado en español",
            "descripcion": "Aquí la descripción generada en español, no más de 100 caracteres"
        }}
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()
            result = response.json().get("response", "")

            # Extraer solo el JSON de la respuesta
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {
                    "titulo": "Momento destacado increíble", 
                    "descripcion": "Un clip que no te puedes perder"
                }

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error al comunicarse con Ollama: {e}")
            return {
                "titulo": "Momento destacado", 
                "descripcion": "Un clip único y sorprendente"
            }
    
    def extract_clips(self, 
                      video_path: str, 
                      highlights: List[Dict], 
                      padding: float = 2
                     ) -> List[Tuple[Path, Path]]:
        """
        Extrae los clips destacados del video y genera un archivo .txt con el título y la descripción usando IA.

        Args:
            video_path: Ruta al archivo de video
            highlights: Lista de momentos destacables
            padding: Segundos adicionales antes y después del momento

        Returns:
            Lista de rutas a los clips generados
        """
        if not highlights:
            self.logger.warning("No se encontraron momentos destacables en el video.")
            return []

        video_filename = Path(video_path).stem
        clips_paths = []

        for i, highlight in enumerate(highlights):
            start_time = max(0, highlight["start"] - padding)
            end_time = highlight["end"] + padding
            duration = end_time - start_time

            tipo = highlight["tipo"]
            descripcion = highlight["descripción"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{video_filename}_{tipo}_{i+1}_{timestamp}.mp4"
            output_path = self.output_folder / output_filename

            # Generar título y descripción con Ollama
            ia_text = self.generate_title_description(tipo, descripcion)

            # Crear el archivo de texto con el título y la descripción
            txt_filename = f"{video_filename}_{tipo}_{i+1}_{timestamp}.txt"
            txt_path = self.output_folder / txt_filename

            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(f"Título: {ia_text['titulo']}\n")
                txt_file.write(f"Descripción: {ia_text['descripcion']}\n")
                txt_file.write(f"Inicio: {start_time:.2f}s\n")
                txt_file.write(f"Fin: {end_time:.2f}s\n")

            # Usar FFmpeg para extraer el clip
            cmd = [
                "ffmpeg", "-i", video_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-c:v", "libx264", "-c:a", "aac",
                "-preset", "fast", "-crf", "22",
                str(output_path)
            ]

            self.logger.info(f"Extrayendo clip {i+1}/{len(highlights)}: {start_time:.2f}s - {end_time:.2f}s")
            subprocess.run(cmd, check=True)

            clips_paths.append((output_path, txt_path))  # Guardar tanto el video como el .txt

        return clips_paths

    def cleanup(self) -> None:
        """Elimina los archivos temporales."""
        for file in self.temp_folder.glob("*"):
            file.unlink()
        
    def process_video(self, 
                      video_path: str, 
                      sample_rate: float = 1.0, 
                      threshold: float = 0.7, 
                      padding: float = 2
                     ) -> List[Tuple[Path, Path]]:
        """
        Procesa un video completo para extraer clips destacables.
        
        Args:
            video_path: Ruta al archivo de video
            sample_rate: Cantidad de frames a analizar por segundo
            threshold: Umbral de confianza para considerar un momento destacable
            padding: Segundos adicionales antes y después del momento
            
        Returns:
            Lista de rutas a los clips generados
        """
        try:
            highlights = self.find_highlights(video_path, sample_rate, threshold)
            self.logger.info(f"Se encontraron {len(highlights)} momentos destacables.")
            
            for i, h in enumerate(highlights):
                self.logger.info(f"Highlight {i+1}: {h['tipo']} - {h['start']:.2f}s a {h['end']:.2f}s - {h['descripción']}")
                
            clips = self.extract_clips(video_path, highlights, padding)
            return clips
        finally:
            self.cleanup()

def main():
    """
    Función principal para ejecutar el script desde la línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Extractor de clips de video con IA")
    parser.add_argument("video_path", help="Ruta al archivo de video MP4")
    parser.add_argument("--model", default="llava:latest", help="Modelo de Ollama a utilizar (por defecto: llava:latest)")
    parser.add_argument("--rate", type=float, default=0.5, help="Cantidad de frames a analizar por segundo (por defecto: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Umbral de confianza para detectar momentos destacables (por defecto: 0.7)")
    parser.add_argument("--padding", type=float, default=2, help="Segundos adicionales al inicio y fin de cada clip (por defecto: 2)")
    parser.add_argument("--output", default="clips", help="Carpeta donde guardar los clips (por defecto: 'clips')")
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Nivel de logging (por defecto: INFO)")
    
    args = parser.parse_args()
    
    extractor = VideoClipExtractor(
        model_name=args.model, 
        output_folder=args.output,
        log_level=args.log
    )
    
    clips = extractor.process_video(
        args.video_path, 
        args.rate, 
        args.threshold, 
        args.padding
    )
    
    print(f"\nProceso completado. Se generaron {len(clips)} clips:")
    for clip, txt in clips:
        print(f"- Video: {clip}")
        print(f"- Texto: {txt}")

if __name__ == "__main__":
    main()

#python analisis-video.py eliminacion.mp4 --model llava:latest --rate 0.5  --threshold 0.7 --padding 2 --output clips_generados --log INFO
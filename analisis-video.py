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

class FortniteClipExtractor:
    def __init__(self, model_name="llava:latest", temp_folder="temp", output_folder="clips"):
        """
        Inicializa el extractor de clips de Fortnite.
        
        Args:
            model_name: Nombre del modelo de Ollama a utilizar
            temp_folder: Carpeta para archivos temporales
            output_folder: Carpeta donde se guardarán los clips
        """
        self.model_name = model_name
        self.temp_folder = Path(temp_folder)
        self.output_folder = Path(output_folder)
        
        # Crear carpetas si no existen
        self.temp_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        
        # Verificar que Ollama esté funcionando
        self._check_ollama()
    
    def _check_ollama(self):
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
                print(f"El modelo {self.model_name} no está disponible. Intentando descargarlo...")
                subprocess.run(["ollama", "pull", self.model_name], check=True)
                print(f"Modelo {self.model_name} descargado correctamente.")
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError("No se pudo conectar a Ollama. Asegúrate de que esté ejecutándose en http://localhost:11434")
        except subprocess.CalledProcessError:
            raise RuntimeError(f"No se pudo descargar el modelo {self.model_name}. Verifica que el nombre sea correcto.")
    
    def extract_frames(self, video_path, sample_rate=1):
        """
        Extrae frames del video para analizar.
        
        Args:
            video_path: Ruta al archivo de video
            sample_rate: Cantidad de frames a extraer por segundo
            
        Returns:
            List de tuplas (timestamp, frame_path)
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video: {video_path}")
        print(f"Duración: {duration:.2f} segundos")
        print(f"FPS: {fps}")
        print(f"Total de frames: {total_frames}")
        
        # Calcular cada cuántos frames guardar uno para análisis
        frame_interval = int(fps / sample_rate)
        
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
                print(f"Procesando frames: {count}/{total_frames} ({count/total_frames*100:.1f}%)")
        
        video.release()
        return frames_info
    
    def analyze_frame(self, frame_path):
        """
        Analiza un frame con el modelo de Ollama para detectar eventos interesantes.
        
        Args:
            frame_path: Ruta al archivo de imagen del frame
            
        Returns:
            Dict con el análisis del frame
        """
        prompt = """
        Analiza esta imagen de una partida de Fortnite y determina si muestra un momento destacable.
        Busca específicamente:
        1. Eliminaciones de enemigos (kills)
        2. Victorias (Victory Royale)
        3. Momentos de alta acción (tiroteos intensos)
        4. Uso de habilidades o elementos especiales
        
        Responde en formato JSON con la siguiente estructura:
        {
            "es_destacable": true/false,
            "tipo_momento": "eliminación"/"victoria"/"acción"/"habilidad"/"normal",
            "confianza": 0.0-1.0,
            "descripción": "breve descripción de lo que ocurre"
        }
        """
        
        # Preparar la solicitud para Ollama
        with open(frame_path, "rb") as img_file:
            img_base64 = img_file.read()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        }
        
        try:
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
                    return json.loads(json_str)
                else:
                    return {"es_destacable": False, "tipo_momento": "normal", "confianza": 0.0, "descripción": "No se pudo analizar"}
            except json.JSONDecodeError:
                print(f"Error al analizar la respuesta JSON: {result}")
                return {"es_destacable": False, "tipo_momento": "normal", "confianza": 0.0, "descripción": "Error de formato"}
                
        except requests.exceptions.RequestException as e:
            print(f"Error al comunicarse con Ollama: {e}")
            return {"es_destacable": False, "tipo_momento": "normal", "confianza": 0.0, "descripción": "Error de comunicación"}
    
    def find_highlights(self, video_path, sample_rate=1, threshold=0.7):
        """
        Encuentra momentos destacables en el video.
        
        Args:
            video_path: Ruta al archivo de video
            sample_rate: Cantidad de frames a analizar por segundo
            threshold: Umbral de confianza mínimo para considerar un momento destacable
            
        Returns:
            Lista de momentos destacables (timestamp, duración, tipo, descripción)
        """
        print(f"Analizando video: {video_path}")
        frames_info = self.extract_frames(video_path, sample_rate)
        
        highlights = []
        current_highlight = None
        
        print(f"Analizando {len(frames_info)} frames con IA...")
        
        for i, (timestamp, frame_path) in enumerate(frames_info):
            print(f"Analizando frame {i+1}/{len(frames_info)} ({timestamp:.2f}s)")
            analysis = self.analyze_frame(frame_path)
            
            if analysis["es_destacable"] and analysis["confianza"] >= threshold:
                # Si no hay un highlight en curso, iniciar uno nuevo
                if current_highlight is None:
                    current_highlight = {
                        "start": timestamp,
                        "tipo": analysis["tipo_momento"],
                        "descripción": analysis["descripción"]
                    }
                # Si es una victoria, finalizar inmediatamente el clip
                elif analysis["tipo_momento"] == "victoria":
                    current_highlight["end"] = timestamp + 5  # Añadir 5 segundos después de la victoria
                    current_highlight["tipo"] = "victoria"
                    current_highlight["descripción"] = "Victory Royale"
                    highlights.append(current_highlight)
                    current_highlight = None
            elif current_highlight is not None:
                # Si hay un highlight en curso y este frame no es destacable, finalizarlo
                # Calculamos cuánto tiempo ha pasado desde el inicio del highlight
                highlight_duration = timestamp - current_highlight["start"]
                
                # Solo terminamos el highlight si ha durado al menos 3 segundos
                if highlight_duration >= 3:
                    # Añadimos 2 segundos más al final para capturar el desenlace
                    current_highlight["end"] = timestamp + 2
                    highlights.append(current_highlight)
                current_highlight = None
        
        # Si hay un highlight en curso al final del video, finalizarlo
        if current_highlight is not None:
            current_highlight["end"] = frames_info[-1][0] + 2
            highlights.append(current_highlight)
        
        return highlights
    
def generate_title_description(self, tipo, descripcion):
    """
    Genera un título y una descripción llamativa para el clip usando Ollama.

    Args:
        tipo: Tipo de momento destacado (ej. "eliminación", "victoria", etc.).
        descripcion: Descripción breve del evento.

    Returns:
        Un diccionario con "titulo" y "descripcion" generados.
    """
    prompt = f"""
    Eres un creador de contenido para plataformas como YouTube y TikTok. 
    Genera un título llamativo y una descripción atractiva para un clip de Fortnite basado en este evento:

    Tipo de clip: {tipo}
    Descripción: {descripcion}

    El título debe ser corto, llamativo y divertido. 
    La descripción debe expandir un poco lo que sucede, con algo de emoción.

    Formato de respuesta en JSON:
    {{
        "titulo": "Aquí el título generado",
        "descripcion": "Aquí la descripción generada"
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
            return {"titulo": "Clip épico en Fortnite", "descripcion": "Mira este increíble momento en Fortnite."}

    except requests.exceptions.RequestException as e:
        print(f"Error al comunicarse con Ollama: {e}")
        return {"titulo": "Momento destacado en Fortnite", "descripcion": "No se pudo generar una descripción personalizada."}


def extract_clips(self, video_path, highlights, padding=2):
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
        print("No se encontraron momentos destacables en el video.")
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

        print(f"Extrayendo clip {i+1}/{len(highlights)}: {start_time:.2f}s - {end_time:.2f}s")
        subprocess.run(cmd, check=True)

        clips_paths.append((output_path, txt_path))  # Guardar tanto el video como el .txt

    return clips_paths



    def cleanup(self):
        """Elimina los archivos temporales."""
        for file in self.temp_folder.glob("*"):
            file.unlink()
        
    def process_video(self, video_path, sample_rate=1, threshold=0.7, padding=2):
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
            print(f"Se encontraron {len(highlights)} momentos destacables.")
            
            for i, h in enumerate(highlights):
                print(f"Highlight {i+1}: {h['tipo']} - {h['start']:.2f}s a {h['end']:.2f}s - {h['descripción']}")
                
            clips = self.extract_clips(video_path, highlights, padding)
            return clips
        finally:
            self.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extractor de clips de Fortnite con IA")
    parser.add_argument("video_path", help="Ruta al archivo de video MP4")
    parser.add_argument("--model", default="llava:latest", help="Modelo de Ollama a utilizar (por defecto: llava:latest)")
    parser.add_argument("--rate", type=float, default=0.5, help="Cantidad de frames a analizar por segundo (por defecto: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Umbral de confianza para detectar momentos destacables (por defecto: 0.7)")
    parser.add_argument("--padding", type=float, default=2, help="Segundos adicionales al inicio y fin de cada clip (por defecto: 2)")
    parser.add_argument("--output", default="clips", help="Carpeta donde guardar los clips (por defecto: 'clips')")
    
    args = parser.parse_args()
    
    extractor = FortniteClipExtractor(model_name=args.model, output_folder=args.output)
    clips = extractor.process_video(args.video_path, args.rate, args.threshold, args.padding)
    
    print(f"\nProceso completado. Se generaron {len(clips)} clips:")
    for clip in clips:
        print(f"- {clip}")
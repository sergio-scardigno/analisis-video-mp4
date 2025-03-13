# analisis-video-mp4

pip install -r requirements.txt
pip install git+https://github.com/openai/whisper.git

python analisis.py tu_video.mp4 --padding 5 --visual-threshold 0.8 --audio-threshold 0.7 --workers 8

Comando Completo recomendado

python analisis.py tu_video.mp4 --model llava:latest --padding 5 --visual-threshold 0.8 --audio-threshold 0.7 --workers 8 --visual-rate 1.0 --output clips_personalizados

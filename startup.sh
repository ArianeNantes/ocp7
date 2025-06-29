echo "🔧 [Startup] Création de venv..."
python -m venv /home/site/venv

echo "📦 [Startup] Activation de venv..."
source /home/site/venv/bin/activate

echo "⬇️ [Startup] Installation des requirements..."
pip install --upgrade pip
pip install -r /home/site/wwwroot/requirements.txt

echo "🚀 [Startup] Lancement de l'app..."
exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind=0.0.0.0:8000

# Procfile
web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker api_llm_cluster:app --bind 0.0.0.0:$PORT

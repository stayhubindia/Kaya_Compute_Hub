import os
import sys
from pathlib import Path

# Add services/api and services/worker to Python path
base_dir = Path(__file__).resolve().parent.parent.parent
api_dir = base_dir / 'services' / 'api'
worker_dir = base_dir / 'services' / 'worker'

for path_dir in [str(api_dir), str(worker_dir)]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

import django
django.setup()

from config.celery import app as celery_app

__all__ = ('celery_app',)

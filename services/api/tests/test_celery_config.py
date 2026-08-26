import pytest
from django.conf import settings
from config.celery import app as celery_app

def test_celery_configuration_loading():
    assert celery_app.main == 'kaya'
    assert settings.CELERY_TASK_SERIALIZER == 'json'
    assert settings.CELERY_RESULT_SERIALIZER == 'json'
    assert 'json' in settings.CELERY_ACCEPT_CONTENT
    assert settings.CELERY_TASK_ACKS_LATE is True

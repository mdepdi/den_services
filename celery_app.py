import os
from celery import Celery
from kombu import Queue
from core.config import settings

# Create Celery instance
celery_app = Celery(
    settings.PROJECT_NAME,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        'tasks.intersite_celery',
        'tasks.fwa_celery'
        ],
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='Asia/Jakarta',
    enable_utc=False,  
    
    # Worker configuration
    # worker_concurrency=4,
    # worker_max_memory_per_child=4 * 1024 * 1024 * 1024,
    worker_max_tasks_per_child=10,
    worker_prefetch_multiplier=1,
    
    # Connection settings
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    broker_heartbeat=10,
    broker_pool_limit=10,

    # Queue settings
    task_queues=(
        Queue('concurrent', routing_key='tasks.concurrent.#'),
        Queue('heavy', routing_key='tasks.heavy.#'),
    ),
    task_default_queue='concurrent',
    task_routes={
        'tasks.concurrent.*': {'queue': 'concurrent', 'routing_key': 'tasks.concurrent.default'},
        'tasks.heavy.*':      {'queue': 'heavy',      'routing_key': 'tasks.heavy.default'},
    },
)

if __name__ == '__main__':
    print("📋 Available tasks:")
    for task_name in sorted(celery_app.tasks.keys()):
        print(f"  - {task_name}")
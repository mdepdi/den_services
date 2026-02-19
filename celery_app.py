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
        'tasks.graphhopper_celery',
        'tasks.fwa_celery'
        ],
)

# Configure Celery
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Time
    timezone="Asia/Jakarta",
    enable_utc=False,

    # Worker fairness
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
    task_track_started=True,
    task_acks_late=False,
    task_reject_on_worker_lost=False,

    # Limits
    task_soft_time_limit=10*60,
    task_time_limit=15*60,

    # Broker reliability
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=3,
    broker_heartbeat=10,
    broker_pool_limit=5,

    # Event
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Results
    result_expires=60*60*24,

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
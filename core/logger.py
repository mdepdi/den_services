import logging
import os
import sys
import shutil
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from core.config import settings

LOG_DIR = settings.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

# ---- Force UTF-8 for Windows ----
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# COLOR FORMATTER (ONLY FOR CONSOLE)
# ============================================================
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG":    "\033[90m",   # grey
        "INFO":     "\033[37m",   # soft green
        "WARNING":  "\033[93m",   # yellow
        "ERROR":    "\033[91m",   # red
        "CRITICAL": "\033[95m",   # magenta
        "RESET": "\033[0m",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        msg = super().format(record)
        return f"{color}{msg}{reset}"


# ============================================================
# CLEAN MODULE NAME
# ============================================================
def clean_name(module_path: str):
    """
    Returns a clean filename-like module name.
    """
    cleaned = os.path.splitext(os.path.basename(module_path))[0]
    return cleaned


# ============================================================
# AUTO CLEAN OLD LOGS (>14 days)
# ============================================================
def cleanup_old_logs(days: int = 14):
    now = datetime.now()
    for folder in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, folder)
        if not os.path.isdir(path):
            continue
        try:
            folder_date = datetime.strptime(folder, "%Y-%m-%d")
            if folder_date < now - timedelta(days=days):
                shutil.rmtree(path)
        except:
            continue


# ============================================================
# MAIN LOGGER
# ============================================================
def create_logger(module_path: str, silent=False):
    """
    Upgraded logger:
      - ALL logs to console (DEBUG → CRITICAL)
      - info.log only stores INFO → CRITICAL
      - Colored console
      - Rotating UTF-8 logs
      - Auto-clean old logs
      - No duplicate handlers
    """
    name = clean_name(module_path)
    logger = logging.getLogger(name)

    # Prevent double handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # --- date folder ---
    today = datetime.now().strftime("%Y-%b-%d")
    date_dir = os.path.join(LOG_DIR, today)
    os.makedirs(date_dir, exist_ok=True)

    info_log = os.path.join(date_dir, f"{name}.log")

    common_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%b-%d %H:%M:%S"

    formatter = logging.Formatter(common_fmt, datefmt=date_fmt)

    # ============================================================
    # FILE HANDLER — keep only INFO and above
    # ============================================================
    info_handler = RotatingFileHandler(
        info_log,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8"
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    # ============================================================
    # CONSOLE HANDLER — show ALL logs (DEBUG → CRITICAL)
    # ============================================================
    if not silent:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG)
        console.setFormatter(ColorFormatter(common_fmt, date_fmt))
        logger.addHandler(console)

    cleanup_old_logs()

    return logger

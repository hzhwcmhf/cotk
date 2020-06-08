import os
import sys
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.INFO)
FORMAT = logging.Formatter("%(levelname)s: %(message)s")
SH = logging.StreamHandler(stream=sys.stdout)
SH.setFormatter(FORMAT)
LOGGER.addHandler(SH)

CONFIG_FILE = os.path.join(str(Path.home()), '.cotk_config')
DASHBOARD_URL = os.getenv("COTK_DASHBOARD_URL", "http://coai.cs.tsinghua.edu.cn/dashboard")

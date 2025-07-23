# Copyright (C) 2025 SHF Co., Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from dotenv import load_dotenv

load_dotenv()

# MariaDB接続情報
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "detection_db")
}

# 画像保存ディレクトリ
IMAGES_DIR = "uploads/result"

# SORACOM用設定
SORACOM_API_ENDPOINT = "https://api.soracom.io/v1"
CAMERA_ID = os.getenv("CAMERA_ID", "")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# 公開先設定
WEBSITE_DOMAINS = os.getenv("WEBSITE_DOMAINS")

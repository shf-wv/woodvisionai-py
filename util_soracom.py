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

import httpx
import os
import asyncio
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
from config import SORACOM_API_ENDPOINT, CAMERA_ID, API_KEY, API_SECRET, IMAGES_DIR

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("camera_capture.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# リトライ設定
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒


async def get_auth_token() -> Tuple[Optional[str], Optional[str]]:
    """ソラコムAPIの認証トークンを取得する"""
    auth_url = f"{SORACOM_API_ENDPOINT}/auth"
    payload = {
        "authKey": API_SECRET,
        "authKeyId": API_KEY
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(auth_url, json=payload)
                response.raise_for_status()
                return (response.json().get("apiKey"), response.json().get("token"))
        except httpx.HTTPStatusError as e:
            logger.error(f"認証エラー (ステータスコード: {e.response.status_code}, リトライ {attempt+1}/{MAX_RETRIES}): {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"認証リクエストエラー (リトライ {attempt+1}/{MAX_RETRIES}): {e}")
        except Exception as e:
            logger.error(f"認証中の予期しないエラー (リトライ {attempt+1}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY)

    return None, None


async def request_image_export(api_key: str, token: str) -> Optional[str]:
    """静止画エクスポートをリクエストする"""
    now = datetime.now()
    export_url = f"{SORACOM_API_ENDPOINT}/sora_cam/devices/{CAMERA_ID}/images/exports"
    headers = {"X-Soracom-API-Key": api_key, "X-Soracom-Token": token}
    payload = {
        "imageFilters": ["wide_angle_correction"],
        "time": int(now.timestamp() * 1000)
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(export_url, headers=headers, json=payload)
                response.raise_for_status()
                export_data = response.json()
                export_id = export_data.get('exportId')
                return export_id
        except httpx.HTTPStatusError as e:
            logger.error(f"エクスポートリクエストエラー (ステータスコード: {e.response.status_code}, リトライ {attempt+1}/{MAX_RETRIES}): {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"エクスポートリクエストエラー (リトライ {attempt+1}/{MAX_RETRIES}): {e}")
        except Exception as e:
            logger.error(f"エクスポート中の予期しないエラー (リトライ {attempt+1}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY)

    return None


async def get_image_url(api_key: str, token: str, export_id: str) -> Optional[str]:
    """静止画エクスポートのURLを取得する"""
    url = f"{SORACOM_API_ENDPOINT}/sora_cam/devices/{CAMERA_ID}/images/exports/{export_id}"
    headers = {"X-Soracom-API-Key": api_key, "X-Soracom-Token": token}

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                export_data = response.json()
                image_url = export_data.get('url')
                if image_url:
                    return image_url

        except httpx.HTTPStatusError as e:
            logger.error(f"URL取得エラー (ステータスコード: {e.response.status_code}, リトライ {attempt+1}/{MAX_RETRIES}): {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"URL取得リクエストエラー (リトライ {attempt+1}/{MAX_RETRIES}): {e}")
        except Exception as e:
            logger.error(f"URL取得中の予期しないエラー (リトライ {attempt+1}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY)

    return None


async def check_export_status(api_key: str, token: str, export_id: str, max_retries: int = 12, retry_interval: int = 5) -> Optional[str]:
    """
    エクスポート状態を確認し、URLが利用可能になるまで待機する

    Args:
        api_key: ソラコムAPIキー
        token: ソラコムトークン
        export_id: エクスポートID
        max_retries: 最大リトライ回数
        retry_interval: リトライ間隔（秒）

    Returns:
        Optional[str]: 画像URL、取得失敗時はNone
    """
    for attempt in range(max_retries):
        image_url = await get_image_url(api_key, token, export_id)
        if image_url:
            return image_url

        await asyncio.sleep(retry_interval)

    logger.error(f"エクスポートタイムアウト: {max_retries}回の試行後もURLを取得できませんでした")
    return None


async def capture_image(schedule_id: int = 0, attempt: int = 1) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    画像を撮影してファイルに保存し、ファイルパスを返却

    Args:
        schedule_id: スケジュールID（オプション）
        attempt: 撮影試行回数（オプション）

    Returns:
        Tuple[bool, Optional[str], Dict[str, Any]]: 
            (撮影成功/失敗, ファイルパス, メタデータ)
    """
    # 現在の日時をタイムスタンプとして使用
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_id = f"{timestamp}_schedule_{schedule_id}_attempt_{attempt}.jpg"
    filepath = os.path.join(IMAGES_DIR, image_id)  # 保存先のファイルパス

    # ディレクトリが存在するか確認し、なければ作成
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # メタデータを準備
    metadata = {
        'timestamp': timestamp,
        'schedule_id': schedule_id,
        'attempt': attempt,
        'image_id': image_id,
        'filepath': filepath,  # ファイルパスをメタデータに含める
        'camera_id': CAMERA_ID,
        'source': 'soracom'
    }

    # 認証関係の初期化
    api_key = None
    token = None
    export_id = None
    image_url = None

    # 認証トークンの取得
    api_key, token = await get_auth_token()
    if not api_key or not token:
        logger.error("認証トークンの取得に失敗しました。API情報を確認してください。")
        return False, None, metadata

    # 静止画エクスポートのリクエスト
    export_id = await request_image_export(api_key, token)
    if not export_id:
        logger.error("静止画エクスポートのリクエストに失敗しました。")
        return False, None, metadata

    # メタデータにエクスポートIDを追加
    metadata['export_id'] = export_id

    # エクスポート状態を確認し、URLが利用可能になるまで待機
    image_url = await check_export_status(api_key, token, export_id)
    if not image_url:
        logger.error("静止画エクスポートのURL取得に失敗しました。")
        return False, None, metadata

    # 画像データの取得とファイルへの保存
    for attempt_dl in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                image_data = response.content

                # ファイルに保存
                with open(filepath, 'wb') as f:
                    f.write(image_data)

                # メタデータに追加情報を格納
                metadata['content_type'] = response.headers.get('content-type')
                metadata['content_length'] = len(image_data)

                return True, filepath, metadata  # ファイルパスを返す

        except httpx.HTTPStatusError as e:
            logger.error(f"画像取得HTTPエラー (ステータスコード: {e.response.status_code}, リトライ {attempt_dl+1}/{MAX_RETRIES}): {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"画像取得リクエストエラー (リトライ {attempt_dl+1}/{MAX_RETRIES}): {e}")
        except Exception as e:
            logger.error(f"画像取得中の予期しないエラー (リトライ {attempt_dl+1}/{MAX_RETRIES}): {e}")

        if attempt_dl < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY)

    return False, None, metadata

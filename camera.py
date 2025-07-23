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

import logging
import os
from config import IMAGES_DIR
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def save_image(image_data: bytes, metadata: Dict[str, Any], is_best: bool = False) -> str:
    """
    画像データをファイルに保存する

    Args:
        image_data: 画像データ (bytes)
        metadata: 画像のメタデータ
        is_best: 最適な画像かどうか

    Returns:
        str: 保存されたファイルのパス
    """
    try:
        # ファイル名を決定（最適な画像の場合は_bestを追加）
        filename = metadata['image_id']
        if is_best:
            base, ext = os.path.splitext(filename)
            filename = f"{base}_best{ext}"

        filepath = os.path.join(IMAGES_DIR, filename)

        # ディレクトリが存在することを確認
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 画像データをファイルに書き込み
        with open(filepath, 'wb') as f:
            f.write(image_data)

        logger.info(f"画像を保存しました: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"画像保存エラー: {e}")
        return None

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

import mariadb
from contextlib import contextmanager
from config import DB_CONFIG
import logging

logger = logging.getLogger(__name__)


@contextmanager
def get_db_connection():
    """データベース接続のコンテキストマネージャ"""
    conn = None
    try:
        conn = mariadb.connect(**DB_CONFIG)
        yield conn
    except mariadb.Error as e:
        logger.error(f"データベース接続エラー: {e}")
        raise
    finally:
        if conn:
            conn.close()


@contextmanager
def transaction(conn):
    """トランザクションを管理するコンテキストマネージャ"""
    try:
        # 現在の自動コミット状態を保存
        original_autocommit = conn.autocommit
        # 自動コミットをオフにしてトランザクション開始
        conn.autocommit = False
        yield conn
        # 正常終了時はコミット
        conn.commit()
    except mariadb.Error as e:
        # MariaDB固有のエラー発生時はロールバック
        logger.error(f"MariaDBエラーによりロールバックします: {e}")
        conn.rollback()
        raise
    except Exception as e:
        # その他の例外発生時もロールバック
        logger.error(f"予期せぬエラーによりロールバックします: {e}")
        conn.rollback()
        raise
    finally:
        # 自動コミットモードを元に戻す
        conn.autocommit = original_autocommit


def get_camera_schedules():
    """カメラの撮影スケジュール情報を取得する"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            sql = "SELECT * FROM ms_AutoShooting"
            cursor.execute(sql)
            return cursor.fetchall()
    except mariadb.Error as e:
        logger.error(f"スケジュール情報取得エラー: {e}")
        return []

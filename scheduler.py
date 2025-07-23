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

import asyncio
import datetime
import logging
import uuid
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import List, Optional, Dict, Any
from collections import Counter
from db import get_camera_schedules
from util_soracom import capture_image
from detect_module import detect_for_validation, detect_for_registration
from util_func import convert_time_to_seconds

logger = logging.getLogger(__name__)


class CameraScheduler:
    def __init__(self):
        """カメラスケジューラの初期化"""
        self.scheduler = AsyncIOScheduler()
        self.active_jobs = {}
        self.last_reload_time = None  # 再読み込み時刻の記録用（オプション）

    async def reload_schedules(self):
        """データベースからスケジュールを読み込みなおして設定"""

        # 既存のジョブをクリア
        for job_id in list(self.active_jobs.keys()):
            self.scheduler.remove_job(job_id)
            del self.active_jobs[job_id]

        # 新しいスケジュールを取得して設定
        schedules = get_camera_schedules()
        for schedule in schedules:
            # 時刻をHH:MM形式に変換
            time_obj = schedule['asTime']
            if isinstance(time_obj, datetime.timedelta):
                hours, remainder = divmod(time_obj.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                time_str = f"{hours:02d}:{minutes:02d}"
            elif isinstance(time_obj, datetime.time):
                time_str = time_obj.strftime("%H:%M")
            else:
                time_str = str(time_obj)

            hour, minute = time_str.split(':')

            # スケジュールジョブの登録
            job_id = f"camera_{schedule['xxId']}"
            self.scheduler.add_job(
                self.execute_camera_task,
                CronTrigger(hour=hour, minute=minute),
                id=job_id,
                kwargs={
                    'schedule_id': schedule['xxId'],
                    'retry_count': schedule['asRetryCount'],
                    'retry_interval': convert_time_to_seconds(schedule['asRetryInterval']),
                    'location_id': schedule['lcId'],
                    'camera_id': schedule['cmId']
                }
            )
            self.active_jobs[job_id] = schedule

        # 再読み込み完了時刻を記録（オプション）
        self.last_reload_time = datetime.datetime.now()

    async def execute_camera_task(self, schedule_id: str, retry_count: int, retry_interval: int, location_id: str, camera_id: str) -> None:
        """
        カメラ撮影タスクを実行

        Args:
            schedule_id: スケジュールID
            retry_count: 撮影回数
            retry_interval: 撮影間隔（秒）
            location_id: 場所ID
        """

        # 撮影結果を格納するリスト
        results: List[Dict[str, Any]] = []

        try:
            # retry_count回の撮影を実行
            for attempt in range(1, retry_count + 1):

                # 撮影実行 - 画像をファイルに保存
                success, filepath, metadata = await capture_image(schedule_id, attempt)

                result = {
                    'attempt': attempt,
                    'success': success,
                    'filepath': filepath,
                    'metadata': metadata,
                    'quantity': None
                }

                # 撮影成功の場合は物体検出を実行
                if success and filepath and os.path.exists(filepath):
                    try:
                        # ファイルから物体検出を実行
                        quantity = detect_for_validation(filepath, location_id)
                        result['quantity'] = quantity
                    except Exception as e:
                        logger.error(f"物体検出エラー: {e}")

                # 結果を記録
                results.append(result)

                # 最後の撮影を除き、retry_interval秒待機
                if attempt < retry_count:
                    await asyncio.sleep(retry_interval)

            # 最適な画像を選択
            best_image_index = self.select_best_result(results)

            if best_image_index is not None:
                best_result = results[best_image_index]

                # 最適な画像をUUID4でリネーム
                best_filepath = best_result['filepath']
                if best_filepath and os.path.exists(best_filepath):
                    # UUID4を生成
                    new_uuid = str(uuid.uuid4())
                    new_filename = f"{new_uuid}.jpg"
                    new_filepath = os.path.join(os.path.dirname(best_filepath), new_filename)

                    try:
                        # リネーム
                        os.rename(best_filepath, new_filepath)

                        # 他の画像を削除
                        self.cleanup_other_images(results, best_image_index)
                    except Exception as e:
                        logger.error(f"画像リネームエラー: {e}")

                else:
                    logger.error(f"最適な画像が見つかりません: {best_filepath}")

                if not detect_for_registration(new_uuid, camera_id, new_filepath, location_id):
                    logger.error("データベースへの登録に失敗しました")

            else:
                logger.warning("有効な撮影結果がありません")
                # すべての画像を削除
                self.cleanup_all_images(results)

        except Exception as e:
            logger.error(f"カメラタスク実行エラー: {e}")
            # エラー発生時もすべての画像を削除
            self.cleanup_all_images(results)

    def select_best_result(self, results: List[Dict[str, Any]]) -> Optional[int]:
        """
        撮影結果から最適なものを選択する

        選択基準:
        1. 最も出現頻度が高い検出数の画像
        2. 出現頻度が同じ場合は、検出数が最も高い画像

        Args:
            results: 撮影結果のリスト

        Returns:
            Optional[int]: 最適な撮影結果のインデックス（0から始まる）。有効な結果がない場合はNone
        """
        # 有効な撮影結果（成功かつ物体検出が実行されたもの）のみを抽出
        valid_results = [r for r in results if r['success'] and r['quantity'] is not None]

        if not valid_results:
            return None

        # 検出数の出現頻度をカウント
        quantities = [r['quantity'] for r in valid_results]
        quantity_counter = Counter(quantities)

        # 最も出現頻度が高い検出数を取得
        most_common = quantity_counter.most_common()
        max_frequency = most_common[0][1]

        # 最頻値の検出数（複数ある可能性あり）
        most_frequent_quantities = [q for q, freq in most_common if freq == max_frequency]

        # 最頻値の中で最も大きい検出数
        best_quantity = max(most_frequent_quantities)

        # 元のresultsリストの中から、best_quantityを持つ最初の結果のインデックスを返す
        for i, result in enumerate(results):
            if result['success'] and result['quantity'] == best_quantity:
                return i

        return None

    def cleanup_other_images(self, results: List[Dict[str, Any]], exclude_index: int) -> None:
        """
        指定したインデックス以外の画像を削除する

        Args:
            results: 撮影結果のリスト
            exclude_index: 削除しないインデックス
        """
        for i, result in enumerate(results):
            if i != exclude_index:
                filepath = result.get('filepath')
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logger.error(f"画像削除エラー: {filepath}, {e}")

    def cleanup_all_images(self, results: List[Dict[str, Any]]) -> None:
        """
        すべての画像を削除する

        Args:
            results: 撮影結果のリスト
        """
        for result in results:
            filepath = result.get('filepath')
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.error(f"画像削除エラー: {filepath}, {e}")

    def start(self):
        """スケジューラを開始"""
        if not self.scheduler.running:
            self.scheduler.start()

    def shutdown(self):
        """スケジューラを停止"""
        if self.scheduler.running:
            self.scheduler.shutdown()

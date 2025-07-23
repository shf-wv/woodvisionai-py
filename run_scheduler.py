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
import logging
import signal
import sys
from scheduler import CameraScheduler
import json
import datetime
from util_func import convert_time_to_seconds

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)
camera_scheduler = None


def signal_handler(sig, frame):
    """シグナルハンドラ（Ctrl+Cなどの終了シグナル処理）"""
    global camera_scheduler
    if sig == signal.SIGHUP:
        logger.info("SIGHUP受信: スケジュールを再読み込みします")
        if camera_scheduler:
            asyncio.run(camera_scheduler.reload_schedules())
    elif sig in (signal.SIGINT, signal.SIGTERM):
        logger.info("終了シグナルを受信しました。サービスを停止します。")
        if camera_scheduler:
            camera_scheduler.shutdown()
        sys.exit(0)


async def update_status_file():
    """サービスの状態をファイルに書き出す"""
    global camera_scheduler
    if not camera_scheduler:
        return

    jobs = []
    for job_id, schedule in camera_scheduler.active_jobs.items():
        job = camera_scheduler.scheduler.get_job(job_id)
        if job:
            jobs.append({
                "id": job.id,
                "next_run_time": job.next_run_time.strftime("%Y-%m-%d %H:%M:%S") if job.next_run_time else None,
                "schedule": {
                    "id": schedule["xxId"],
                    "time": str(schedule["asTime"]),
                    "retry_count": schedule["asRetryCount"],
                    "retry_interval": convert_time_to_seconds(schedule["asRetryInterval"])
                }
            })

    status = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scheduler_running": camera_scheduler.scheduler.running,
        "active_schedules": jobs
    }

    with open('status.json', 'w') as f:
        json.dump(status, f, indent=2)


async def periodic_status_update():
    """定期的にステータスを更新する"""
    global camera_scheduler
    while True:
        if camera_scheduler:
            await camera_scheduler.reload_schedules()
        await update_status_file()
        await asyncio.sleep(600)  # 10分ごとに更新


async def main():
    """メイン関数"""
    global camera_scheduler

    # シグナルハンドラの設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    # スケジューラの初期化と開始
    camera_scheduler = CameraScheduler()
    await camera_scheduler.reload_schedules()
    camera_scheduler.start()

    # ステータス更新タスクを開始
    status_task = asyncio.create_task(periodic_status_update())

    try:
        # スケジューラが動き続けるように無限ループで待機
        while True:
            await asyncio.sleep(7200)  # 2時間ごとにチェック
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")
        raise
    finally:
        # クリーンアップ
        status_task.cancel()
        camera_scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

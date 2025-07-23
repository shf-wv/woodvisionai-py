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
import sys
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# FastAPIアプリケーションの作成
app = FastAPI(
    title="Camera Calibration API",
    description="カメラキャリブレーション用API",
    version="0.1.0"
)

origins = os.getenv("WEBSITE_DOMAINS", "")
allow_origins = [origin.strip() for origin in origins.split(",") if origin.strip()]

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Referer", "Origin"],
    expose_headers=["Content-Type", "Authorization"],
    max_age=600,
)


# 返却用データモデル
class CalibrationResponse(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    p1: float
    p2: float
    k1: float
    k2: float
    k3: float
    error: float


@app.post("/api/v1/calibrate-camera", response_model=CalibrationResponse)
async def calibration_background(
    file_path: str = Form(...)
):
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    MIN_VIDEO_LENGTH_SECONDS = 10

    video_path = f"{file_path}"

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"指定されたファイルが見つかりません: {video_path}")
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=400, detail=f"指定されたパスはファイルではありません: {video_path}")

    # 拡張子チェック
    file_extension = Path(file_path).suffix.lower()
    if file_extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="サポートされていないファイル形式です。"
        )

    ffprobe_command = [
        "ffprobe",
        "-v", "error",  # エラーのみ表示
        "-select_streams", "v:0",  # 最初のビデオストリームを選択
        "-show_entries", "stream=duration:format=format_name",  # 長さとフォーマット名を取得
        "-of", "default=noprint_wrappers=1:nokey=1",  # キーなし、ラッパーなしで値のみ出力
        video_path
    ]

    try:
        # ffprobeコマンドを非同期で実行
        process = await asyncio.create_subprocess_exec(
            *ffprobe_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            raise HTTPException(
                status_code=400,
                detail=f"動画ファイルの解析に失敗しました。ファイルが破損しているか、サポートされていない形式です。エラー: {error_msg}"
            )

        ffprobe_output = stdout.decode().strip().split('\n')
        if len(ffprobe_output) < 2:
            raise HTTPException(status_code=400, detail="動画のメタデータ（長さ、フォーマット）の取得に失敗しました。有効な動画ファイルか確認してください。")

        duration_str = ffprobe_output[0]

        try:
            duration_seconds = float(duration_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"動画の長さを数値としてパースできませんでした: {duration_str}")

        # 動画の長さが10秒以上かチェック
        if duration_seconds < MIN_VIDEO_LENGTH_SECONDS:
            raise HTTPException(
                status_code=400,
                detail="動画が短すぎます。最低でも10秒の長さを確保してください。"
            )

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffprobeコマンドが見つかりません。"
        )

    # バリデーションここまで

    BASE_DIR = Path(__file__).parent
    script_path = BASE_DIR / "calib.py"

    calib_command = [sys.executable, str(script_path), video_path]

    try:
        calib_process = await asyncio.create_subprocess_exec(
            *calib_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # タイムアウト付きで実行
        try:
            stdout, stderr = await asyncio.wait_for(
                calib_process.communicate(),
                timeout=180.0
            )
        except asyncio.TimeoutError:
            # プロセスを強制終了
            if calib_process:
                calib_process.kill()
                await calib_process.wait()

            raise HTTPException(
                status_code=408,
                detail="処理がタイムアウトしました"
            )

        if calib_process.returncode == 1:
            raise HTTPException(
                status_code=500,
                detail="内部エラーが発生しました。"
            )

        elif calib_process.returncode == 2:
            raise HTTPException(
                status_code=422,
                detail="キャリブレーション処理に失敗しました。"
            )

        try:
            # スペース区切りの10個の値を期待
            values = stdout.decode().strip().split()
            if len(values) != 10:
                raise HTTPException(
                    status_code=500,
                    detail="キャリブレーション結果の形式が不正です。"
                )

            # 浮動小数点数に変換
            float_values = [float(v) for v in values]

            return CalibrationResponse(
                error=float_values[0],
                cx=float_values[1],
                cy=float_values[2],
                fx=float_values[3],
                fy=float_values[4],
                p1=float_values[5],
                p2=float_values[6],
                k1=float_values[7],
                k2=float_values[8],
                k3=float_values[9]
            )

        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"キャリブレーション結果を数値に変換できませんでした: {str(e)}"
            )

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="キャリブレーションスクリプトが見つかりません。"
        )

    finally:
        # プロセスのクリーンアップ
        if calib_process and calib_process.returncode is None:
            try:
                calib_process.kill()
                await calib_process.wait()
            except:
                pass


@app.get("/")
def read_root():
    return {"message": "Camera Calibration API"}

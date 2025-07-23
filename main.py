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
import io
import uuid
from typing import Optional
import mariadb
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from pathlib import Path

# 現在のファイルのディレクトリを取得
current_dir = Path(__file__).parent.absolute()

# sys.pathに追加（重複チェック付き）
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from processor_calib import WideAngleCorrector

# 環境変数の読み込み
load_dotenv()

# FastAPIアプリケーションの作成
app = FastAPI(
    title="YOLO Object Detection API",
    description="YOLO11を使用した物体検出API",
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

# MariaDB接続情報
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "detection_db")
}

# YOLOモデルの読み込み
try:
    model = YOLO("python_app/best.pt")
except Exception as e:
    print(f"モデルの読み込みに失敗しました: {e}")
    raise


# データベース接続関数
def get_db_connection():
    try:
        return mariadb.connect(**DB_CONFIG)
    except mariadb.Error as e:
        print(f"MariaDBへの接続エラー: {e}")
        raise HTTPException(status_code=500, detail=f"データベース接続エラー: {str(e)}")


# 検出結果のモデル
class Detection(BaseModel):
    x: float
    y: float
    w: float
    h: float
    quantity: int
    conf_score: float


# 返却用データモデル
class DetectionResponse(BaseModel):
    id: str


@app.post("/api/v1/bff/detect/{lcId}", response_model=DetectionResponse)
async def detect_objects(
    lcId: str,
    camera_id: Optional[str] = Form(None),
    enable_detection: bool = Form(True),
    file: UploadFile = File(...)
):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")

    is_fixed = 1

    if not camera_id:
        camera_id = ""
        is_fixed = 0

    try:
        # DBとの接続を取得する
        conn = get_db_connection()
        cursor = conn.cursor()

        corrector = WideAngleCorrector(DB_CONFIG)

        # 検出結果IDを生成
        rsId = str(uuid.uuid4())

        # 画像を読み込む
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        processed_image = corrector.process_image(image, camera_id)

        # 画像を保存
        processed_image.save(f"uploads/result/{rsId}.jpg")

        if enable_detection:  # 物体検出を実施する場合の処理

            # 入り数取得のためのクエリ
            query = "SELECT pkQuantity FROM ms_Packing INNER JOIN ms_Location ON ms_Packing.xxId = ms_Location.pkId WHERE ms_Location.xxId = ?"
            cursor.execute(query, (lcId,))
            res = cursor.fetchone()

            if res is None:
                # 取得できなかったときはデフォルト値として200を採用
                per_quantity = 200
            else:
                per_quantity = int(res[0])

            # YOLO11で物体検出
            results = model.predict(image, conf=0.5)

            # 検出結果の処理
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x, y, w, h = map(float, box.xywh[0])

                    if class_id == 0:  # 梱包
                        inbox_quantity = per_quantity
                    else:
                        inbox_quantity = 1

                    detection = Detection(
                        quantity=inbox_quantity,
                        conf_score=confidence,
                        x=x, y=y, w=w, h=h
                    )
                    detections.append(detection)

            # MariaDBに結果を保存
            total_quantity = 0
            detection_count = 0

            try:
                for detection in detections:
                    bbId = str(uuid.uuid4())
                    detection_count += 1
                    cursor.execute(
                        "INSERT INTO dt_BBoxes (xxId, rsId, bbX, bbY, bbWidth, bbHeight, bbQuantity, bbOriginalX, bbOriginalY, bbOriginalWidth, bbOriginalHeight, bbOriginalQuantity, bbConfScore, bbOrder, bbIsModified, bbIsGenerated, bbIsDeleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (bbId, rsId, detection.x, detection.y, detection.w, detection.h, detection.quantity, detection.x, detection.y, detection.w, detection.h, detection.quantity, detection.conf_score, detection_count, 0, 1, 0)
                    )
                    total_quantity += detection.quantity

                cursor.execute(
                    "INSERT INTO dt_Result (xxId, aiId, lcId, rsDatetime, rsQuantityByAI, rsQuantity, rsAdjustment, rsIsModified, cmId, rsIsFixed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rsId, "01234567-89ab-cdef-0123-456789abcdef", lcId, datetime.now(), total_quantity, total_quantity, 0, 0, camera_id, is_fixed)
                )
                conn.commit()

            except mariadb.Error as e:
                conn.rollback()
                raise HTTPException(status_code=500, detail=f"データベースエラー: {str(e)}")
            finally:
                cursor.close()
                conn.close()

            # 結果を返却
            return DetectionResponse(id=rsId)

        else:  # 物体検出を行わない場合の処理
            try:
                total_quantity = 0  # 検出数0として結果テーブルにのみ保存

                cursor.execute(
                    "INSERT INTO dt_Result (xxId, aiId, lcId, rsDatetime, rsQuantityByAI, rsQuantity, rsAdjustment, rsIsModified, cmId, rsIsFixed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rsId, "01234567-89ab-cdef-0123-456789abcdef", lcId, datetime.now(), total_quantity, total_quantity, 0, 0, camera_id, is_fixed)
                )
                conn.commit()

            except mariadb.Error as e:
                conn.rollback()
                raise HTTPException(status_code=500, detail=f"データベースエラー: {str(e)}")
            finally:
                cursor.close()
                conn.close()

            # 結果を返却
            return DetectionResponse(id=rsId)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"データベースエラー: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "YOLO Object Detection API"}

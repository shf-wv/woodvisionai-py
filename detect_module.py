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
import uuid
from datetime import datetime
import mariadb
from ultralytics import YOLO
from PIL import Image
from pydantic import BaseModel
from db import get_db_connection, transaction


# 検出結果のモデル
class Detection(BaseModel):
    x: float
    y: float
    w: float
    h: float
    quantity: int
    conf_score: float


def detect_for_validation(image_path, lcId):

    # YOLOモデルの読み込み
    try:
        model = YOLO("python_app/best.pt")
    except Exception as e:
        logging.error(f"モデルの読み込みに失敗しました: {e}")
        return None

    try:
        # DBとの接続を取得する
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 入り数取得のためのクエリ
            query = "SELECT pkQuantity FROM ms_Packing INNER JOIN ms_Location ON ms_Packing.xxId = ms_Location.pkId WHERE ms_Location.xxId = ?"
            cursor.execute(query, (lcId,))
            res = cursor.fetchone()

            if res is None:
                logging.error("ロケーションの入数が設定されていません。設定を見直してください。")
                return None

            per_quantity = int(res[0])

            # 画像を読み込む
            image = Image.open(image_path)

            # YOLO11で物体検出
            results = model.predict(image, conf=0.5)

            total_quantity = 0

            # 検出結果の処理
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])

                    if class_id == 0:  # 梱包
                        total_quantity += per_quantity
                    else:
                        total_quantity += 1

            return total_quantity

    except Exception as e:
        logging.error(f"物体検出エラー: {e}")
        return None


def detect_for_registration(rsId, cmId, image_path, lcId):

    # YOLOモデルの読み込み
    try:
        model = YOLO("python_app/best.pt")
    except Exception as e:
        logging.error(f"モデルの読み込みに失敗しました: {e}")
        return False

    try:
        # DBとの接続を取得する
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 入り数取得のためのクエリ
            query = "SELECT pkQuantity FROM ms_Packing INNER JOIN ms_Location ON ms_Packing.xxId = ms_Location.pkId WHERE ms_Location.xxId = ?"
            cursor.execute(query, (lcId,))
            res = cursor.fetchone()

            if res is None:
                logging.error("ロケーションの入数が設定されていません。設定を見直してください。")
                return False

            per_quantity = int(res[0])

            # 画像を読み込む
            image = Image.open(image_path)

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
                with transaction(conn):
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
                        (rsId, "01234567-89ab-cdef-0123-456789abcdef", lcId, datetime.now(), total_quantity, total_quantity, 0, 0, cmId, 1)
                    )

                return True

            except mariadb.Error as e:
                logging.error(f"MariaDBエラー: {e}")
                return False
            except Exception as e:
                logging.error(f"予期せぬエラー: {e}")
                return False

    except Exception as e:
        logging.error(f"物体検出エラー: {e}")
        return False

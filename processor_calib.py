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

import cv2
import numpy as np
from PIL import Image
import mariadb
from mariadb import Error
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationDataManager:
    """キャリブレーションデータの管理クラス"""

    def __init__(self, db_config):
        """
        Args:
            db_config (dict): データベース接続設定
        """
        self.db_config = db_config

    def get_calibration_data(self, cm_id):
        """
        指定されたcmIdの最新のキャリブレーションデータを取得

        Args:
            cm_id (str): カメラID

        Returns:
            dict or None: キャリブレーションパラメータ、見つからない場合はNone
        """
        if not cm_id:
            logger.info("カメラIDが指定されていません")
            return None

        connection = None
        cursor = None

        try:
            # MariaDBに接続
            connection = mariadb.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 3306),
                database=self.db_config.get('database'),
                user=self.db_config.get('user'),
                password=self.db_config.get('password')
            )
            cursor = connection.cursor()

            query = """
            SELECT clFx, clFy, clCx, clCy, clP1, clP2, clK1, clK2, clK3
            FROM dt_Calibration
            WHERE cmId = ?
              AND xxIsDeleted = 0
            ORDER BY clDatetime DESC
            LIMIT 1
            """

            cursor.execute(query, (cm_id,))
            result = cursor.fetchone()

            if result:
                # すべてのパラメータがNoneでないことを確認
                if any(param is None for param in result):
                    logger.warning(f"カメラID '{cm_id}' のキャリブレーションデータに不完全な値が含まれています")
                    return None

                calibration_data = {
                    'fx': float(result[0]),
                    'fy': float(result[1]),
                    'cx': float(result[2]),
                    'cy': float(result[3]),
                    'p1': float(result[4]),
                    'p2': float(result[5]),
                    'k1': float(result[6]),
                    'k2': float(result[7]),
                    'k3': float(result[8])
                }

                return calibration_data
            else:
                return None

        except Error as e:
            logger.error(f"MariaDBエラー: {e}")
            return None
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return None
        finally:
            # リソースの適切なクリーンアップ
            if cursor:
                cursor.close()
            if connection:
                connection.close()


class ImageProcessor:
    """画像処理クラス"""

    @staticmethod
    def pil_to_opencv(pil_image):
        """
        PIL画像をOpenCV形式に変換

        Args:
            pil_image (PIL.Image): PIL形式の画像

        Returns:
            numpy.ndarray: OpenCV形式の画像（BGR）
        """
        # PIL画像をRGB形式に変換してからnumpy配列に変換
        rgb_image = np.array(pil_image.convert('RGB'))

        # RGBからBGRに変換（OpenCVはBGR形式を使用）
        if len(rgb_image.shape) == 3:
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = rgb_image

        return bgr_image

    @staticmethod
    def opencv_to_pil(opencv_image):
        """
        OpenCV画像をPIL形式に変換

        Args:
            opencv_image (numpy.ndarray): OpenCV形式の画像（BGR）

        Returns:
            PIL.Image: PIL形式の画像
        """
        # BGRからRGBに変換
        if len(opencv_image.shape) == 3:
            rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = opencv_image

        # PIL画像に変換
        return Image.fromarray(rgb_image)

    @staticmethod
    def apply_distortion_correction(opencv_image, calibration_data, alpha=0):
        """
        OpenCV画像に歪み補正を適用

        Args:
            opencv_image (numpy.ndarray): OpenCV形式の画像
            calibration_data (dict): キャリブレーションパラメータ
            alpha (float): 補正パラメータ（0=完全クロップ、1=全領域保持）

        Returns:
            numpy.ndarray: 補正後のOpenCV画像
        """
        height, width = opencv_image.shape[:2]

        # カメラ内部パラメータ行列の構築
        camera_matrix = np.array([
            [calibration_data['fx'], 0, calibration_data['cx']],
            [0, calibration_data['fy'], calibration_data['cy']],
            [0, 0, 1]
        ], dtype=np.float64)

        # 歪み係数の構築（OpenCVの標準順序：k1, k2, p1, p2, k3）
        dist_coeffs = np.array([
            calibration_data['k1'],
            calibration_data['k2'],
            calibration_data['p1'],
            calibration_data['p2'],
            calibration_data['k3']
        ], dtype=np.float64)

        # 最適なカメラ行列とROIを計算
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (width, height), alpha, (width, height)
        )

        # 歪み補正を適用
        undistorted_image = cv2.undistort(
            opencv_image, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        # ROIでクロップ（alpha=0の場合、歪みのない領域のみを保持）
        if alpha == 0:
            x, y, w, h = roi
            if w > 0 and h > 0:
                undistorted_image = undistorted_image[y:y+h, x:x+w]
                # 元のサイズにリサイズ
                undistorted_image = cv2.resize(
                    undistorted_image, (width, height), interpolation=cv2.INTER_AREA
                )

        return undistorted_image


class WideAngleCorrector:
    """広角補正メインクラス"""

    def __init__(self, db_config):
        """
        Args:
            db_config (dict): データベース接続設定
        """
        self.calibration_manager = CalibrationDataManager(db_config)
        self.image_processor = ImageProcessor()

    def process_image(self, pil_image, cm_id=None):
        """
        PIL画像に広角補正を適用

        Args:
            pil_image (PIL.Image): 入力PIL画像
            cm_id (str, optional): カメラID

        Returns:
            PIL.Image: 処理後のPIL画像
        """
        # カメラIDが指定されていない場合は補正をスキップ
        if not cm_id:
            return pil_image

        # キャリブレーションデータを取得
        calibration_data = self.calibration_manager.get_calibration_data(cm_id)

        if not calibration_data:
            return pil_image

        try:
            # PIL画像をOpenCV形式に変換
            opencv_image = self.image_processor.pil_to_opencv(pil_image)

            # 広角補正を適用（alpha=0）
            corrected_opencv_image = self.image_processor.apply_distortion_correction(
                opencv_image, calibration_data, alpha=0
            )

            # OpenCV画像をPIL形式に変換
            corrected_pil_image = self.image_processor.opencv_to_pil(corrected_opencv_image)

            return corrected_pil_image

        except Exception as e:
            logger.error(f"広角補正処理中にエラーが発生しました: {e}")
            return pil_image

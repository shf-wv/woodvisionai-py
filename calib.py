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
import argparse
import sys
from typing import Tuple, Optional, Dict, Any


def calculate_basic_blur_score(image: np.ndarray) -> float:
    """
    基本的なラプラシアン分散によるブレ検出

    Args:
        image: グレースケール画像

    Returns:
        ブレスコア（値が大きいほど鮮明）
    """
    if image is None or image.size == 0:
        return 0.0

    # ラプラシアンフィルタを適用して分散を計算
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return float(laplacian.var())


def is_frame_sharp(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
    """
    フレームが鮮明かどうかを判定

    Args:
        image: 入力画像
        threshold: 鮮明度の閾値

    Returns:
        (鮮明かどうか, ブレスコア)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blur_score = calculate_basic_blur_score(gray)
    return blur_score > threshold, blur_score


class CameraCalibrator:
    def __init__(self, 
                 board_size: Tuple[int, int] = (7, 7), 
                 square_size: float = 29.0,
                 min_change_threshold: float = 0.25,
                 sharpness_threshold: float = 100.0):
        """
        カメラキャリブレーションクラス

        Args:
            board_size: チェスボードの内部コーナー数 (width, height)
            square_size: チェスボードの正方形のサイズ（実際の単位）
            min_change_threshold: 類似性判定の閾値（0-1の範囲）
            sharpness_threshold: これ以上のブレだった時に弾く
        """
        self.board_size = board_size
        self.square_size = square_size
        self.min_change_threshold = min_change_threshold
        self.sharpness_threshold = sharpness_threshold
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 3Dオブジェクトポイントの準備
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

        # ストレージ配列
        self.objpoints = []  # 3D点
        self.imgpoints = []  # 2D点

        # 画像サイズ（最初のフレームで設定）
        self.image_size = None

        # 統計情報
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detected_frames': 0,
            'accepted_frames': 0,
            'rejected_similar': 0,
            'similarity_scores': []
        }

    def _calculate_frame_similarity(self, current_corners: np.ndarray, 
                                    previous_corners: np.ndarray,
                                    image_size: tuple) -> dict:
        """
        高度な類似度分析（複数の指標を組み合わせ）

        Returns:
            各種類似度指標の辞書
        """
        if previous_corners is None:
            return {'overall': 0.0, 'position': 0.0, 'scale': 0.0, 'rotation': 0.0}

        # 画像の対角線長（正規化基準）
        image_diagonal = np.sqrt(image_size[0]**2 + image_size[1]**2)

        # 1. 位置変化の評価
        position_distances = np.linalg.norm(current_corners - previous_corners, axis=2)
        mean_position_change = np.mean(position_distances) / image_diagonal
        position_similarity = np.exp(-mean_position_change / (self.min_change_threshold * 0.5))

        # 2. スケール変化の評価
        current_center = np.mean(current_corners.reshape(-1, 2), axis=0)
        previous_center = np.mean(previous_corners.reshape(-1, 2), axis=0)

        current_distances_to_center = np.linalg.norm(
            current_corners.reshape(-1, 2) - current_center, axis=1
        )
        previous_distances_to_center = np.linalg.norm(
            previous_corners.reshape(-1, 2) - previous_center, axis=1
        )

        scale_ratio = np.mean(current_distances_to_center) / np.mean(previous_distances_to_center)
        scale_change = abs(np.log(scale_ratio))
        scale_similarity = np.exp(-scale_change / 0.2)

        # 3. 回転変化の評価
        def get_orientation(corners):
            points = corners.reshape(-1, 2)
            centered_points = points - np.mean(points, axis=0)
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            main_direction = eigenvectors[:, np.argmax(eigenvalues)]
            return np.arctan2(main_direction[1], main_direction[0])

        current_orientation = get_orientation(current_corners)
        previous_orientation = get_orientation(previous_corners)

        angle_diff = abs(current_orientation - previous_orientation)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        rotation_similarity = np.exp(-angle_diff / 0.3)

        # 4. 総合類似度（重み付き平均）
        weights = {'position': 0.5, 'scale': 0.3, 'rotation': 0.2}
        overall_similarity = (
            weights['position'] * position_similarity +
            weights['scale'] * scale_similarity +
            weights['rotation'] * rotation_similarity
        )

        return {
            'overall': overall_similarity,
            'position': position_similarity,
            'scale': scale_similarity,
            'rotation': rotation_similarity
        }

    def extract_calibration_data(self, video_path: str, 
                               frame_interval: int = 10) -> bool:
        """
        改良されたキャリブレーションデータ抽出
        """

        cap = cv2.VideoCapture(video_path)

        # 動画の基本情報を取得
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 画像サイズを取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.image_size = (width, height)

        self.stats['total_frames'] = frame_count

        previous_corners = None
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1

            # 指定間隔でフレームを処理
            if current_frame % frame_interval == 0:
                self.stats['processed_frames'] += 1

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # チェスボードのコーナーを検出
                ret_corners, corners = cv2.findChessboardCorners(gray, self.board_size, None)

                if ret_corners:
                    self.stats['detected_frames'] += 1

                    # サブピクセル精度でコーナーを改善
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                    # ブレ検出の実行
                    is_sharp, blur_score = is_frame_sharp(gray)

                    if not is_sharp:
                        continue  # ブレが大きいため除外

                    # 類似度判定
                    similarity_info = self._calculate_frame_similarity(
                        corners_refined, previous_corners, self.image_size
                    )
                    is_similar = similarity_info['overall'] > (1.0 - self.min_change_threshold)
                    similarity_score = similarity_info['overall']

                    self.stats['similarity_scores'].append(similarity_score)

                    if not is_similar:
                        self.objpoints.append(self.objp)
                        self.imgpoints.append(corners_refined)
                        previous_corners = corners_refined.copy()
                        self.stats['accepted_frames'] += 1

                    else:
                        self.stats['rejected_similar'] += 1

        cap.release()

        return self.stats['accepted_frames'] >= 10

    def calibrate(self, image_size: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        カメラキャリブレーションを実行

        Args:
            image_size: 画像サイズ (width, height)

        Returns:
            キャリブレーション結果の辞書、失敗時はNone
        """
        if not self.objpoints or not self.imgpoints:
            print("エラー: キャリブレーション用のポイントが設定されていません", file=sys.stderr)
            return None

        # カメラキャリブレーションを実行
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )

        if not ret:
            print("エラー: カメラキャリブレーションに失敗しました", file=sys.stderr)
            return None

        total_squared_error = 0
        total_points = 0

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], 
                                            camera_matrix, dist_coeffs)

            # 形状を合わせる（必要に応じて）
            imgpoints2 = imgpoints2.reshape(-1, 2)
            detected_points = self.imgpoints[i].reshape(-1, 2)

            # 各点の二乗誤差を計算
            squared_errors = np.sum((detected_points - imgpoints2) ** 2, axis=1)
            total_squared_error += np.sum(squared_errors)
            total_points += len(squared_errors)

        mean_error = np.sqrt(total_squared_error / total_points)

        # 結果を整理
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        # 歪み係数（OpenCVの標準的な順序）
        dist_flat = dist_coeffs.flatten()
        k1, k2, p1, p2 = dist_flat[0], dist_flat[1], dist_flat[2], dist_flat[3]
        k3 = dist_flat[4] if len(dist_flat) > 4 else 0.0

        calibration_result = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "k1": float(k1),
            "k2": float(k2),
            "k3": float(k3),
            "p1": float(p1),
            "p2": float(p2),
            "reprojection_error": float(mean_error),
        }

        return calibration_result


def main():
    parser = argparse.ArgumentParser(description="動画ファイルからカメラキャリブレーションを実行")
    parser.add_argument("video_path", help="動画ファイルのパス")
    parser.add_argument("--board-width", type=int, default=7, 
                       help="チェスボードの幅（内部コーナー数）")
    parser.add_argument("--board-height", type=int, default=7, 
                       help="チェスボードの高さ（内部コーナー数）")
    parser.add_argument("--square-size", type=float, default=29, 
                       help="チェスボードの正方形サイズ")
    parser.add_argument("--frame-interval", type=int, default=10, 
                       help="フレーム抽出間隔")
    parser.add_argument("--min-change", type=float, default=0.25, 
                       help="最小変化量の閾値")

    args = parser.parse_args()

    try:
        # カメラキャリブレーターを初期化
        calibrator = CameraCalibrator(
            board_size=(args.board_width, args.board_height),
            square_size=args.square_size,
            min_change_threshold=args.min_change
        )

        # 動画からキャリブレーションデータを抽出
        if not calibrator.extract_calibration_data(args.video_path, args.frame_interval):
            sys.exit(2)
        else:
            # 画像サイズを取得（最初のフレームから）
            cap = cv2.VideoCapture(args.video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                height, width = frame.shape[:2]
                image_size = (width, height)

                # キャリブレーションを実行
                result = calibrator.calibrate(image_size)

                if result is None:
                    sys.exit(1)
            else:
                sys.exit(1)

        # 結果出力
        print(f"{result['reprojection_error']} {result['cx']} {result['cy']} {result['fx']} {result['fy']} {result['p1']} {result['p2']} {result['k1']} {result['k2']} {result['k3']}", file=sys.stdout)
        sys.exit(0)

    except Exception:
        sys.exit(2)


if __name__ == "__main__":
    main()

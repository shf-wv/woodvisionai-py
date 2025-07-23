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

import datetime
import logging

logger = logging.getLogger(__name__)


def convert_time_to_seconds(time_value) -> int:
    """TIME型の値を秒数に変換する

    Args:
        time_value: TIME型の値（datetime.time, datetime.timedelta, または文字列など）

    Returns:
        int: 秒数
    """
    # datetime.timeオブジェクトの場合
    if isinstance(time_value, datetime.time):
        return time_value.hour * 3600 + time_value.minute * 60 + time_value.second

    # datetime.timedeltaオブジェクトの場合
    elif isinstance(time_value, datetime.timedelta):
        return int(time_value.total_seconds())

    # 文字列の場合（'HH:MM:SS'または'MM:SS'形式）
    elif isinstance(time_value, str):
        parts = time_value.split(':')
        if len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        else:  # SS
            return int(parts[0])

    # 整数または浮動小数点数の場合
    elif isinstance(time_value, (int, float)):
        return int(time_value)

    # その他の場合（デフォルト値）
    else:
        logger.warning(f"不明なretry_interval形式: {time_value}, デフォルト値(30秒)を使用します")
        return 30  # デフォルト値

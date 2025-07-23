# woodvisionai-py

## 概要

Wood Vision AIのPython部分のスクリプトです。

## 設定

環境変数に、以下の項目を登録してください。

- `DB_HOST`: 使用データベースのホスト名（デフォルト：localhost）
- `DB_PORT`: 使用データベースのポート番号（デフォルト：3306）
- `DB_USER`: データベースに接続するユーザー（デフォルト：root）
- `DB_PASSWORD`: データベース接続用パスワード（デフォルト：パスワードなし）
- `DB_NAME`: データベース名（デフォルト：detection_db）
- `CAMERA_ID`: SORACOM用、カメラのデバイスID
- `API_KEY`: SORACOM用、APIキー
- `API_SECRET`: SORACOM用、APIシークレット
- `WEBSITE_DOMAINS`: 公開先Webサイト、カンマ区切りで複数入力可

**注意:** 本番環境ではセキュリティのため、適切なユーザー権限と強力なパスワードを設定してください。

## ライセンス
Copyright (c) [2025] [SHF Co., Ltd.]

本スクリプトは、GNU Affero General Public License v3.0 (AGPL v3.0) でライセンスされています。

**重要:** このソフトウェアをネットワーク経由で利用する場合（APIとして提供する場合を含む）、AGPL v3.0の条項に従い、利用者にソースコードを提供する必要があります。

本プロジェクトのソースコード一式は、[このリポジトリ](https://github.com/shf-wv/woodvisionai-py)で公開しています。

詳細については、[LICENSE](LICENSE) ファイルをご覧ください。
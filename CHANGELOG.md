# Changelog

このプロジェクトの注目すべき変更点はすべてこのファイルに記録されます。

形式は[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)に基づいており、
このプロジェクトは[Semantic Versioning](https://semver.org/spec/v2.0.0.html)に従っています。

## [2.1.0] - 2026-02-21

### 追加
- BDDC (Balancing Domain Decomposition by Constraints) 前処理行列
  - BilinearForm からの要素ごとの構成
  - NGSolve CouplingType による Wirebasket/interface DOF 分類
  - 粗解法: SparseCholesky (デフォルト)、PARDISO、dense LU
  - メッシュ非依存収束 (すべてのテスト問題で CG 2 反復)
- `docs/` の日本語ドキュメント (アーキテクチャ、アルゴリズム、API リファレンス、チュートリアル)
- すべてのソースファイルに MPL 2.0 ライセンスヘッダ
- CONTRIBUTING.md

### 変更
- ブロック消去 BDDC アプローチを削除 (要素ごとのアプローチに置き換え)
- 未使用の前処理行列を削除 (Identity、Jacobi、ILU)
- MRTRSolver を削除 (SGS-MRTR は自己完結)
- IC 前処理行列の重複する `reorder_matrix()` コードを統合
- 共有 `BuildSparseMatrixView()` を抽出してコード重複を排除
- pybind11 型登録を簡潔化 (ファクトリが公開 API)

### 修正
- DenseMatrix::invert() 置換行列の構成 (P^T → P)
  - 自明でないピボッティングの HCurl 問題で BDDC 失敗を引き起こす
- 複素数内積: 複素対称 FEM 行列の非共役内積
  - 渦電流の発散を修正 (5000 反復 → 58 反復)
- SGS-MRTR 複素数比較: 閾値チェックに `std::real(denom)` を使用

## [2.0.0] - 2026-02-19

### 追加
- NGSolve とは独立したスタンドアロン pybind11 モジュール (`sparsesolv_ngsolve.pyd`)
- 自動ディスパッチ機能を備えたファクトリ関数 (`mat.IsComplex()` による実数/複素数の自動判定)
- 半正定値行列の自動シフト IC 分解 (curl-curl)
- 条件数改善のための対角スケーリング
- ABMC (Algebraic Block Multi-Color) 順序付け (並列三角解法用)
- RCM 帯域幅削減 (オプション、ABMC と組み合わせ)
- 複素数対応 (`std::complex<double>`) を完全サポート
- 発散検出と最良結果の復帰
- 残差履歴記録
- レベルスケジュール三角解法の永続並列領域

### 変更
- ヘッダオンリー C++17 ライブラリとして再構成
- `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` コンパイルフラグによる NGSolve 統合
- 並列化の抽象化: TaskManager / OpenMP / シリアル ディスパッチ

## [1.0.0] - 2026-02-01

[JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv) からの初回フォーク。

### 上流からの機能
- シフトパラメータ付き IC(0) 前処理行列
- SGS 前処理行列
- CG ソルバー
- 分割公式による SGS-MRTR ソルバー

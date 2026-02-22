# SparseSolv

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

[NGSolve](https://ngsolve.org/) 有限要素解析向けのヘッダオンリー C++17 反復法ソルバーライブラリ。
大規模疎行列連立方程式に対して、2つの相補的なアプローチを提供する:

- **BDDC前処理** — 要素単位の領域分割とwirebasket粗空間による前処理。メッシュ非依存の反復回数を実現。NGSolve組込みBDDCと同一のアルゴリズムを実装しており、開発者が独自のBDDCを構築する際のリファレンスとなる。
- **ABMC並列化ICCG** — 代数的ブロックマルチカラー (ABMC) 順序付けによる並列三角解法付きの不完全コレスキー分解CG。BDDCよりセットアップコストが低く、適度な規模の問題に有効。

`double` と `std::complex<double>` の両方をサポートし、NGSolveのソースツリーとは独立したpybind11拡張モジュール (`sparsesolv_ngsolve`) として提供する。

[JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv) のfork。

## なぜSparseSolvか？

NGSolveには直接法ソルバーと組込みBDDCが搭載されている。SparseSolvはNGSolveが標準では提供しない3つの機能を追加する:

1. **透過的で独立したBDDC実装** — NGSolveの組込みBDDCはフレームワーク内部に深く埋め込まれたプロダクションコードである。SparseSolvは同一のアルゴリズムを、読みやすく自己完結したC++ヘッダオンリーライブラリとして再実装した。同一問題での反復回数は[NGSolveのBDDCと完全に一致](docs/03_sparsesolv_vs_ngsolve_bddc.ipynb)し、BDDCを学習・改良・拡張したい開発者のリファレンスとなる。

2. **電磁界 (HCurl) 問題への堅牢性** — curl-curl FEM行列は半正定値であり、標準のIC分解は破綻する。SparseSolvのauto-shift ICは破綻を検出し自動調整する。さらにBDDCは、ソース項が離散的にdiv-freeであるかどうかに関わらず収束する — [ICCGでは必須だが実際には保証しにくい条件](docs/02_performance_comparison.ipynb)を不要にする。

3. **ABMCマルチカラー順序付けによる並列ICCG** — IC前処理の三角解法は本質的に逐次的である。ABMC順序付けがこのボトルネックを解消し、並列前進・後退代入を実現する。BDDCのセットアップコストが割に合わない中規模問題でICCGを競争力のあるものにする。

## ソルバー選択ガイド

| 問題 | 有限要素空間 | 推奨手法 | 理由 |
|------|------------|---------|------|
| Poisson (高次) | H1 (order ≥ 3) | **BDDC+CG** | 2反復、メッシュ非依存 |
| Poisson (低次) | H1 (order 1-2) | **ICCG** | BDDCのセットアップコスト不要 |
| 弾性体 | VectorH1 | **BDDC+CG** | ICCGは細分化で反復数増加 |
| Curl-curl (実数) | HCurl (`nograds=True`) | **BDDC+CG** or **Shifted-ICCG** | BDDCはソース構成に非依存 |
| 渦電流 (複素数) | HCurl (complex) | **BDDC+CG** (`conjugate=False`) | 複素対称行列対応 |
| 小規模 (< 1K DOFs) | 任意 | 直接法 | 反復法のオーバーヘッド大 |

チュートリアルと計算時間の比較は [tutorials](docs/tutorials.md) を参照。

## 性能

3D HCurl curl-curl問題でのベンチマーク (8スレッド, `nograds=True`)。
詳細は [02_performance_comparison.ipynb](docs/02_performance_comparison.ipynb)。

**トロイダルコイル** (148K DOFs, order 2 — div-freeソース):

| ソルバー | 反復数 | 計算時間 | vs ICCG |
|---------|--------|---------|---------|
| ICCG | 513 | 13.1 s | 1.0x |
| ICCG + ABMC (8色) | 444 | 7.4 s | **1.8x** |
| BDDC | 47 | 4.2 s | **3.1x** |

ABMCは三角解法のボトルネックを並列化し、ICCG計算時間をほぼ半減。
BDDCはメッシュ非依存の収束でさらに高速。

**ヘリカルコイル** (565K DOFs, order 2 — ソース構成の堅牢性):

| ソース構成 | ICCG | BDDC |
|-----------|------|------|
| ポテンシャルベース `J*v*dx` (非div-free) | 1000反復, **不収束** | 33反復, 収束 |
| curl-based `T*curl(v)*dx` (div-free) | 161反復, 収束 | 53反復, 収束 |

ICCGはソースが離散的にdiv-freeであることを要求する — 実際には保証しにくい条件。
BDDCはソース構成に関わらず収束する。

## ドキュメント

[docs/](docs/) に詳細ドキュメント (日本語) を整備:
- [アーキテクチャ](docs/architecture.md) — ソースコード構成と設計
- [アルゴリズム](docs/algorithms.md) — アルゴリズム解説 (BDDC, IC, SGS-MRTR, CG, ABMC)
- [BDDC実装ガイド](docs/bddc_implementation_details.md) — 理論、疑似コード、API、ベンチマーク
- [ABMC実装ガイド](docs/abmc_implementation_details.md) — 並列三角解法、性能解析
- [APIリファレンス](docs/api_reference.md) — Python APIリファレンス
- [チュートリアル](docs/tutorials.md) — 実践例 (全ソルバー比較)
- [開発者向け情報](docs/development.md) — ビルド、テスト、開発ノート

### ベンチマーク・ノートブック

| ノートブック | 内容 |
|------------|------|
| [01_shift_parameter.ipynb](docs/01_shift_parameter.ipynb) | 半正定値HCurlでのICシフトパラメータ |
| [02_performance_comparison.ipynb](docs/02_performance_comparison.ipynb) | BDDC vs ICCG vs ICCG+ABMC 性能比較 |
| [03_sparsesolv_vs_ngsolve_bddc.ipynb](docs/03_sparsesolv_vs_ngsolve_bddc.ipynb) | SparseSolv BDDC = NGSolve BDDC 同等性 |

## 機能

### 前処理
- **BDDC** (Balancing Domain Decomposition by Constraints) — 要素単位構築、wirebasket粗空間、メッシュ非依存反復
- **IC** (不完全コレスキー) — shifted IC(0)、半正定値行列向けauto-shift
- **SGS** (対称ガウス・ザイデル) — 分解不要

### 反復法ソルバー
- **CG** (共役勾配法) — SPD系、複素対称 (`conjugate=False`) 対応
- **SGS-MRTR** — split formula内蔵のMRTR

### 統合手法
- **ICCG** — CG + IC前処理 (オプションでABMC並列三角解法)
- **SGSMRTR** — SGS-MRTR (自己完結型)

### 高度な機能
- 半正定値行列 (curl-curl問題) 向けauto-shift IC分解
- 対角スケーリングによる条件数改善
- ABMC (代数的ブロックマルチカラー) 順序付けによる並列三角解法
- 数値的破綻検出と収束認識型回復
- best-result追跡 (未収束時は最良反復解を返却)
- 残差履歴記録

### 並列化

全並列処理は `core/parallel.hpp` によりコンパイル時にディスパッチ:

| ビルド設定 | バックエンド | 用途 |
|-----------|------------|------|
| `SPARSESOLV_USE_NGSOLVE_TASKMANAGER` | NGSolve TaskManager (`ngcore::ParallelFor/ParallelReduce`) | NGSolve統合 |
| `_OPENMP` | OpenMP `#pragma omp parallel for` | スタンドアロン+OpenMP |
| (どちらもなし) | シリアル実行 | スタンドアロン (スレッドなし) |

## インストール

### ビルド済みWheel (推奨)

[Releases](https://github.com/ksugahar/ngsolve-sparsesolv/releases) ページからプラットフォーム対応の `.whl` ファイルをダウンロードしてインストール:

```bash
pip install sparsesolv_ngsolve-2.1.0-cp312-cp312-win_amd64.whl
```

ファイル名はPythonバージョンとOSに合わせて適宜変更。

### ソースからビルド

[Git for Windows](https://gitforwindows.org/) (または同等品)、CMake 3.16+、
C++17コンパイラ、NGSolve (`pip install ngsolve` またはソースビルド) が必要。

```bash
pip install git+https://github.com/ksugahar/ngsolve-sparsesolv.git
```

またはクローンしてローカルビルド:

```bash
git clone https://github.com/ksugahar/ngsolve-sparsesolv.git
cd ngsolve-sparsesolv
pip install .
```

開発ビルド (editable install, 手動CMake) は [docs/development.md](docs/development.md) を参照。

### 動作確認

```bash
python -c "from sparsesolv_ngsolve import SparseSolvSolver; print('OK')"
```

## NGSolveでの使い方

### クイックスタート (ICCG)

```python
from ngsolve import *
from sparsesolv_ngsolve import SparseSolvSolver

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
fes = H1(mesh, order=2, dirichlet="left|right|top|bottom")
u, v = fes.TnT()

a = BilinearForm(fes)
a += grad(u)*grad(v)*dx
a.Assemble()

f = LinearForm(fes)
f += 1*v*dx
f.Assemble()

gfu = GridFunction(fes)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10)
gfu.vec.data = solver * f.vec
```

### BDDC前処理

```python
from sparsesolv_ngsolve import BDDCPreconditioner
from ngsolve.krylovspace import CGSolver

# BDDCはBilinearForm + FESpaceを受け取る (行列だけではない)
pre = BDDCPreconditioner(a, fes, coarse_inverse="sparsecholesky")
inv = CGSolver(a.mat, pre, tol=1e-10)
gfu.vec.data = inv * f.vec
```

### ABMC並列化ICCG

```python
# ABMCでIC前処理の三角解法を並列化
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10,
                          use_abmc=True,
                          abmc_block_size=4,
                          abmc_num_colors=4)
gfu.vec.data = solver * f.vec
```

ABMCアルゴリズムは近傍の行をブロックに集約 (BFS集約) した後、
ブロック隣接グラフを彩色して同色ブロック間の下三角依存関係を排除する。
三角解法では色を逐次処理し、各色内のブロックを並列に実行する。
148K DOF HCurl問題 (8スレッド) で、ABMCはICCG計算時間を13.1秒から7.4秒に短縮 (1.8倍高速化)。

### 前処理 + NGSolve CGSolver

```python
from sparsesolv_ngsolve import ICPreconditioner, SGSPreconditioner
from ngsolve.krylovspace import CGSolver

# IC前処理 + NGSolve CG
pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
inv = CGSolver(a.mat, pre, tol=1e-10, maxiter=2000)
gfu.vec.data = inv * f.vec
```

### 3D curl-curl (半正定値系)

```python
from netgen.occ import Box, Pnt

box = Box(Pnt(0,0,0), Pnt(1,1,1))
for face in box.faces:
    face.name = "outer"
mesh = Mesh(box.GenerateMesh(maxh=0.3))

fes = HCurl(mesh, order=1, dirichlet="outer", nograds=True)
u, v = fes.TnT()

a = BilinearForm(fes)
a += curl(u)*curl(v)*dx
a.Assemble()

f = LinearForm(fes)
f += CF((0,0,1))*v*dx
f.Assemble()

# auto-shift ICで半正定値curl-curl行列に対応
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(),
                          tol=1e-8, maxiter=2000, shift=1.0)
solver.auto_shift = True
solver.diagonal_scaling = True

gfu = GridFunction(fes)
gfu.vec.data = solver * f.vec
```

### 複素対称系 vs エルミート系

SparseSolvは複素対称系とエルミート系の両方を `conjugate` パラメータで切り替える:

```python
# 複素対称 (A^T = A): 渦電流 (curl-curl + iσ mass) など
# デフォルト: conjugate=False (非共役内積 a^T * b)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-8)

# エルミート (A^H = A): 実係数問題を複素FE空間で解く場合など
# conjugate=True (共役内積 a^H * b)
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-8,
                          conjugate=True)
```

NGSolveの `CGSolver(conjugate=True/False)` に対応。

### SparseSolvSolver パラメータ

| パラメータ | 型 | 既定値 | 説明 |
|-----------|------|--------|------|
| `method` | str | `"ICCG"` | ソルバー手法: `ICCG`, `SGSMRTR`, `CG` |
| `tol` | float | `1e-10` | 収束許容値 |
| `maxiter` | int | `1000` | 最大反復回数 |
| `shift` | float | `1.05` | ICシフトパラメータ (安定性) |
| `auto_shift` | bool | `False` | IC破綻時のシフト自動増加 |
| `diagonal_scaling` | bool | `False` | 対角スケーリング |
| `save_best_result` | bool | `True` | 最良反復解の追跡・復元 |
| `save_residual_history` | bool | `False` | 各反復の残差を記録 |
| `conjugate` | bool | `False` | エルミート系用の共役内積 |
| `divergence_check` | bool | `False` | 停滞時の早期終了 |
| `printrates` | bool | `False` | 収束情報のstdout出力 |
| `use_abmc` | bool | `False` | ABMC順序付けによる並列三角解法 |
| `abmc_block_size` | int | `4` | ABMCブロックあたりの行数 |
| `abmc_num_colors` | int | `4` | ABMC目標色数 |

### Pythonクラス一覧

```python
from sparsesolv_ngsolve import (
    # Factory関数 (mat.IsComplex()で実数/複素数を自動判定)
    BDDCPreconditioner,            # BDDC (BilinearForm + FESpace API)
    ICPreconditioner,              # NGSolve CGSolverと併用
    SGSPreconditioner,             # NGSolve CGSolverと併用
    SparseSolvSolver,              # 統合ソルバー (ICCG, SGSMRTR)

    SparseSolvResult,              # ソルブ結果
)
```

Factory関数は行列の型に基づいて適切なテンプレート (`double` or `complex<double>`) に自動ディスパッチ。

## スタンドアロン使用 (C++)

```cpp
#include <sparsesolv/sparsesolv.hpp>

// CSR行列ビューを作成 (ゼロコピー)
sparsesolv::SparseMatrixView<double> A(rows, cols, row_ptr, col_idx, values);

// ICCGで解く
sparsesolv::SolverConfig config;
config.tolerance = 1e-10;
config.max_iterations = 1000;

auto result = sparsesolv::solve_iccg(A, b, x, size, config);
if (result.converged) {
    // 解は x に格納
}
```

## ディレクトリ構成

```
ngsolve-sparsesolv/
├── docs/                        # ドキュメント (日本語)
├── include/sparsesolv/         # ヘッダオンリーライブラリ
│   ├── sparsesolv.hpp          # メインヘッダ (全コンポーネントをインクルード)
│   ├── core/                   # 型、設定、行列ビュー、前処理基底
│   │   ├── dense_matrix.hpp    # 小規模密行列 + LU逆行列 (BDDC用)
│   │   ├── sparse_matrix_coo.hpp # COO疎行列 (組立用)
│   │   ├── sparse_matrix_csr.hpp # CSR疎行列 (格納用)
│   │   ├── parallel.hpp        # 並列化抽象レイヤ (TaskManager/OpenMP/serial)
│   │   ├── level_schedule.hpp  # レベルスケジューリング (三角解法並列化)
│   │   └── abmc_ordering.hpp   # ABMC順序付け (三角解法並列化)
│   ├── preconditioners/        # IC, SGS, BDDC実装
│   ├── solvers/                # CG, SGS-MRTR実装
│   └── ngsolve/                # NGSolve BaseMatrixラッパー + pybind11エクスポート
├── ngsolve/
│   └── python_module.cpp       # pybind11モジュールエントリポイント
├── tests/
│   ├── test_sparsesolv.py      # ソルバー・前処理テスト
│   └── test_bddc.py            # BDDC前処理テスト
├── CMakeLists.txt              # ヘッダオンリーライブラリ + NGSolveモジュールビルド
└── LICENSE
```

## 参考文献

1. C.R. Dohrmann,
   "A preconditioner for substructuring based on constrained energy minimization",
   *SIAM J. Sci. Comput.*, Vol. 25, No. 1, pp. 246–258, 2003.
   — BDDC前処理の原論文。

2. J. Mandel, C.R. Dohrmann, R. Tezaur,
   "An algebraic theory for primal and dual substructuring methods by constraints",
   *Appl. Numer. Math.*, Vol. 54, No. 2, pp. 167–193, 2005.
   — BDDCの代数的理論。

3. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel Multi-Threaded
   Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE 26th International Parallel and Distributed Processing Symposium (IPDPS)*, 2012.
   — ABMCオーダリングアルゴリズム。

4. J.A. Meijerink, H.A. van der Vorst,
   "An iterative solution method for linear systems of which the coefficient matrix
   is a symmetric M-matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148–162, 1977.
   — 不完全コレスキー (IC) 分解。

5. M.R. Hestenes, E. Stiefel,
   "Methods of conjugate gradients for solving linear systems",
   *J. Res. Nat. Bur. Standards*, Vol. 49, No. 6, pp. 409–436, 1952.
   — 共役勾配 (CG) 法。

6. E. Cuthill, J. McKee,
   "Reducing the bandwidth of sparse symmetric matrices",
   *Proc. 24th National Conference of the ACM*, pp. 157–172, 1969.
   — Reverse Cuthill-McKee (RCM) 帯域縮小順序付け。

7. 圓谷吉宏, 三船泰, 岩下武史, 高橋英治,
   "MRTR法に基づく前処理付き反復法の数値実験",
   *電気学会研究会資料*, SA-12-64, 2012.
   — SGS-MRTR法: 国内研究会論文。

8. Y. Tsuburaya, T. Mifune, T. Iwashita, E. Takahashi,
   "Minimum Residual-Like Methods for Solving Ax = b with Shift-and-Invert Enhanced
   Multi-Step MRTR",
   *IEEE Transactions on Magnetics*, Vol. 49, No. 5, pp. 1569–1572, 2013.
   — SGS-MRTRソルバー。

9. 圓谷吉宏,
   "大規模電磁界問題の有限要素解析のための反復法の開発",
   *博士論文*, 京都大学, 2016.
   — 大規模電磁界FEMの反復法に関する包括的参考文献。

10. 比留間伸吾, JP-MARs/SparseSolv,
    https://github.com/JP-MARs/SparseSolv
    — 原実装 (MPL 2.0ライセンス)。

## ライセンス

本プロジェクトは [Mozilla Public License 2.0](LICENSE) の下でライセンスされている。

SparseSolvはNGSolveとは独立したモジュールとして提供される。
インストール済みNGSolveに対してリンクするスタンドアロンpybind11拡張モジュールとしてビルドされる。

# 代数的ブロックマルチカラー (ABMC) オーダリングの実装詳細

このドキュメントでは、SparseSolv で使用されている代数的ブロックマルチカラー (ABMC) オーダリング手法の数学的背景と C++ 実装の詳細について説明します。ABMC は、不完全コレスキー (IC) 前処理における前進代入および後退代入ステップを並列化するための手法です。

**実装ファイル:**
- `core/abmc_ordering.hpp` — ABMCスケジュール構築 (BFS集約 + グラフ彩色)
- `core/level_schedule.hpp` — レベルスケジューリング (比較対象)
- `core/parallel.hpp` — 並列プリミティブ (`parallel_for`, `SpinBarrier`, `get_num_threads`)
- `core/solver_config.hpp` — ABMCパラメータ設定
- `preconditioners/ic_preconditioner.hpp` — 三角解法の実行、apply パス選択

---

## 1. 理論的背景

### 1.1 並列三角行列解法の課題

ICCG のような反復ソルバーでは、前処理の適用に三角連立方程式を解く必要がある:
- 前進代入: $L y = x$ を解く
- 後退代入: $L^T z = y$ を解く

$y_i$ の計算は $L_{ij} \neq 0$ となる $j < i$ のすべての $y_j$ に依存する。このデータ依存性が並列化のボトルネックとなる。

### 1.2 レベルスケジューリング

依存関係を解消する基本手法。各行のレベル（依存深度）を計算し、同じレベルの行を並列に処理する:

```
level[i] = max(level[j] for j in L[i,:] where j < i) + 1
```

**問題点**: FEM行列は帯域幅が広く依存チェーンが深い。レベル数が数百になり、各レベルの行数が少ないため並列効率が低い。

### 1.3 マルチカラーリング

グラフ彩色により依存関係を断ち切るアプローチ。隣接する2つのノードが同じ色を持たないように色付けし、同じ色のノードを並列に処理する。しかし、古典的なポイントベースのマルチカラーリングは、色に基づいてメモリアクセスが飛躍するためキャッシュの局所性が悪い。

### 1.4 ABMC: ブロック + マルチカラー

ABMC [Iwashita, Nakashima, Takahashi 2012] は2レベルの階層を導入して、並列性とキャッシュ局所性を両立する:

1. **ブロッキング**: 近傍の行をブロックにまとめる → キャッシュ局所性を維持
2. **カラーリング**: ブロック間の依存関係を彩色 → 少ない同期回数で並列化

幾何学的情報を必要とせず、CSR形式の行列パターンのみで動作する。

---

## 2. 実装パイプライン

ABMC スケジュールは `ICPreconditioner::setup()` 中に構築される。`ABMCSchedule::build()` が4つのステージを順に実行する (`abmc_ordering.hpp`)。

### ステージ 1: 代数的ブロッキング (BFS集約)

**関数**: `make_algebraic_blocks()`

行列グラフ上のBFS (幅優先探索) により、近傍の行をブロックにまとめる。

```
入力: CSR行列 (row_ptr, col_idx), ブロックサイズ B
出力: raw_block_list[b] = {元の行インデックスの集合}
       block_assign[i] = 行iのブロックID

1. block_assign を -1 (未割当) で初期化
2. 未割当の最初の行を seed として新ブロック開始
3. seed の隣接ノードを BFS キューに追加 (mark -2: キュー内)
4. キューから取り出し、ブロックサイズ B に達するまで追加
5. 残ったキューノードを -1 に戻す (次のブロック用)
6. 全行割当済みまで繰り返し
```

**sentinel値の使い分け:**
| 値 | 意味 |
|----|------|
| `-1` | 未割当 |
| `-2` | BFSキュー内 (重複追加防止) |
| `>= 0` | ブロックID割当済み |

**計算量**: O(nnz) — 各非ゼロ要素を最大2回走査

ブロックサイズ `B` (既定: 4, `SolverConfig::abmc_block_size`) は、ブロック内の行が逐次処理されるため、キャッシュ効率と並列粒度のバランスで決まる。

### ステージ 2: ブロック隣接グラフの構築

**関数**: `build_block_graph()`

ブロック $A$ 内のいずれかの行が、ブロック $B$ 内のいずれかの行に非ゼロ接続を持つ場合、2つのブロックは隣接している。

```
入力: raw_block_list, block_assign, CSR行列
出力: ブロック隣接グラフ (blk_row_ptr, blk_col_idx) — CSR形式

for 各ブロック bi:
    for 各行 row ∈ bi:
        for 各非ゼロ列 j ∈ row:
            bj = block_assign[j]
            if bj != bi and last_seen[bj] != bi:
                neighbors[bi] に bj を追加
                last_seen[bj] = bi  ← 重複防止sentinel
```

**計算量**: O(nnz)。`last_seen` sentinel により、ブロック対あたり1回のみ追加。

### ステージ 3: グリーディ彩色

**関数**: `color_graph()`

ブロック隣接グラフを彩色する。下三角依存関係のあるブロック同士が異なる色を持つようにする。

**2段階のアルゴリズム:**

```
Phase 1: 最小色数の決定
    num_colors = target_colors  (SolverConfig::abmc_num_colors, 既定: 4)
    for 各ブロック i:
        lower_count = |{j : j < i かつ (i,j) が隣接}|
        num_colors = max(num_colors, lower_count + 1)

Phase 2: Forbidden-Color-Set による貪欲彩色
    forbidden[c] = i は「色 c がブロック i の下三角隣接ノードで使用中」を意味
    for 各ブロック i (番号順):
        下三角隣接ノード j < i の色を forbidden にマーク
        最小の未使用色を選択
        全色使用済みの場合: ++num_colors (新色を追加)
```

**重要**: `target_colors` は**下限**であり保証値ではない。実際の色数は:
$$\text{num\_colors} \geq \max(\text{target\_colors},\ 1 + \max_i |\{j < i : (i,j) \in E\}|)$$

Phase 2 でさらに増加する可能性がある（Phase 1 は下三角次数のみで計算するが、彩色順序によっては追加の色が必要になる）。

**計算量**: O(sum of degrees) — 各ブロックについて隣接ノードを1回走査

### ステージ 4: 行レベルの順序構築

**関数**: `build_row_ordering()`

ブロック彩色結果から、行レベルのフラット配列を構築する。

```
出力:
  color_offsets[c] — 色c の開始ブロック位置 (CSR)
  color_blocks[k] — global ブロックID
  block_offsets[b] — ブロックb の開始行位置 (CSR)
  ordering[old] = new — 元→新の置換
  reverse_ordering[new] = old — 新→元の逆置換

走査順: 色0 → 色1 → ... → 色C-1
  各色内: ブロック順
    各ブロック内: 元のインデックスでソート (std::sort)
```

**連番割り当て**: 新しい行インデックスは走査順に連番で割り当てられる。ブロック $b$ 内の行は `block_offsets[b]` から `block_offsets[b+1]-1` まで連続しており、別途の行インデックス配列は不要。三角解法では `for (i = block_offsets[blk]; i < block_offsets[blk+1]; ++i)` と直接ループできる。

ブロック内ソートの目的は、同一ブロック内の行が元の行列で近いインデックスを持つことで、前処理適用時のTLBミスを削減しキャッシュラインの利用率を向上させること。

---

## 3. 並列三角解法のロジック

### 3.1 データ構造

ABMC スケジュール後の三角解法に関わるデータ:

| データ | 空間 | 説明 |
|--------|------|------|
| `L_`, `Lt_`, `inv_diag_` | ABMC空間 | IC分解結果 (下/上三角因子, 逆対角) |
| `abmc_x_perm_` | ABMC空間 | 入力ベクトル置換先 (size n) |
| `abmc_y_perm_` | ABMC空間 | 出力ベクトル置換先 (size n) |
| `work_temp_` | ABMC空間 | 中間結果ベクトル (size n, ABMC/level scheduling共有) |
| `work_temp2_` | ABMC空間 | 第2中間結果 (対角スケーリング時のみ, size n) |
| `composite_perm_` | 元→ABMC | 合成置換 |
| `composite_scaling_` | 元空間 | 対角スケーリング係数 (元空間) |

**ワークベクトルの共有**: `work_temp_` は ABMC パスとレベルスケジューリングパスで共有される。`setup()` 時に1回確保され、`apply()` 時のヒープ割り当てを排除している。

### 3.2 前進代入 ($L y = x$)

`forward_substitution_abmc()` — 3階層ループ:

```cpp
for (色 c = 0; c < num_colors; ++c) {           // 逐次: 色間依存性
    parallel_for(色c内のブロック数, [&](bidx) {    // 並列: 同色ブロックは独立
        blk = color_blocks[blk_begin + bidx];
        row_begin = block_offsets[blk];
        row_end = block_offsets[blk + 1];
        for (i = row_begin; i < row_end; ++i) {  // 逐次: ブロック内依存性
            s = x[i];                             // ABMC空間の行インデックスを直接使用
            for (k : L_の行iの非対角要素)
                s -= L_.values[k] * y[L_.col_idx[k]];
            y[i] = s / L_.values[diag];
        }
    });
    // 暗黙の同期バリア (parallel_for完了 = 次の色に進める)
}
```

`block_offsets` が連続インデックスを定義するため、ブロック内の行は `row_begin` から `row_end` まで直接ループできる。間接参照配列は不要。

### 3.3 後退代入 ($L^T z = D^{-1} y$)

`backward_substitution_abmc()` — 前進代入の逆順:
- 色は C-1 → 0 の逆順
- ブロック内の行も逆順 (`row_end → row_begin`)
- $L^T$ (上三角) を使用

```cpp
for (c = nc; c-- > 0;) {                          // 色を逆順
    parallel_for(色c内のブロック数, [&](bidx) {
        for (i = row_end; i-- > row_begin;) {      // 行を逆順
            s = 0;
            for (k : Lt_の行iの非対角要素)
                s -= Lt_.values[k] * y[Lt_.col_idx[k]];
            y[i] = s * inv_diag_[i] + x[i];       // D^{-1} 適用 + ソース加算
        }
    });
}
```

### 3.4 正確性の根拠

ABMC彩色により、同じ色のブロック間には下三角依存関係が存在しない。したがって:
- 色 c のブロック内の前進代入は、色 < c の結果のみに依存
- 色 c の各ブロックは互いに独立 → 並列実行可能
- ブロック内は元の行列の依存関係を保持 → 逐次実行で正確

---

## 4. apply パスの選択

`ICPreconditioner` は3つの apply パスを提供する。入力/出力ベクトルの「空間」が異なる。

### 4.1 標準パス: `apply_abmc()` (推奨)

CG が元の空間で動作する場合。前処理適用時に置換を行う。

```
x (元空間) → composite_perm_ で置換 → abmc_x_perm_ (ABMC空間)
  → forward_substitution_abmc → work_temp_
  → backward_substitution_abmc → abmc_y_perm_ (ABMC空間)
y (元空間) ← composite_perm_ で逆置換 ← abmc_y_perm_
```

対角スケーリング有効時は、置換と同時にスケーリングを適用:
```cpp
abmc_x_perm_[perm[i]] = x[i] * composite_scaling_[i];
y[i] = abmc_y_perm_[perm[i]] * composite_scaling_[i];
```

`composite_scaling_[i] = scaling_[composite_perm_[i]]` — 事前計算済み。

### 4.2 ABMC全空間パス: `apply_in_reordered_space()`

CG が ABMC 空間で動作する場合（`abmc_reorder_spmv=True` 時）。入出力が既にABMC空間のため置換不要。

### 4.3 RCM+ABMC分離パス: `apply_rcm_abmc()`

CG が RCM 空間で動作する場合。RCM→ABMC の置換のみ必要:
```
x (RCM空間) → composite_perm_rcm_ で置換 → ABMC空間
  → 三角解法
y (RCM空間) ← composite_perm_rcm_ で逆置換 ← ABMC空間
```

### 4.4 パス選択の判断基準

| パス | SpMV 行列 | 前処理 | 設定 |
|------|----------|--------|------|
| 標準 (4.1) | 元の行列 | 合成置換で ABMC 三角解法 | `use_abmc=True` |
| RCM+ABMC (4.3) | RCM行列 | RCM→ABMC 置換 | `use_abmc=True, abmc_use_rcm=True` |
| ABMC全空間 (4.2) | ABMC行列 | 置換なし | `abmc_reorder_spmv=True` |

NGSolve統合 (`sparsesolv_ngsolve` モジュール) では、SpMV は NGSolve の `BaseMatrix::Mult` が担当するため、元の行列空間で動作する**標準パス (4.1)** が使用される。

---

## 5. RCM との組み合わせ

### 5.1 合成置換 (Composite Permutation)

RCM+ABMC を使う場合、2段階の置換が必要:

```
元の行 i → RCM空間: rcm_ordering_[i]
RCM空間 → ABMC空間: abmc_ord[rcm_ordering_[i]]
```

毎回2段階の間接参照を行うのではなく、setup 時に合成置換を事前計算する:

```cpp
// build_composite_permutations()
composite_perm_[i] = abmc_ord[rcm_ordering_[i]];  // 元 → ABMC (1段)
composite_perm_rcm_ = abmc_ord;                    // RCM → ABMC (1段)
```

### 5.2 行列コピーの流れ

RCM+ABMC パスでは行列コピーが2回発生する:

```
元の行列 A
  ↓ reorder_matrix_with_perm(A, rcm_ordering_)
RCM行列 rcm_csr_
  ↓ reorder_matrix(rcm_view)  [ABMC順序で並べ替え]
ABMC行列 reordered_csr_
  ↓ extract_lower_triangular()
L_ (IC分解用)
```

ABMC-only パスでは1回のみ (`A → reordered_csr_ → L_`)。

### 5.3 メモリ管理

factorization完了後、不要な行列コピーを解放:
```cpp
if (!config_.abmc_reorder_spmv)
    reordered_csr_.clear();  // SpMV は元の行列を使用
if (!config_.abmc_use_rcm)
    rcm_csr_.clear();
```

---

## 6. レベルスケジューリングとの比較

### 6.1 並列化手法の違い

SparseSolv は2つの並列三角解法を提供する。ABMCを使わない場合、**persistent parallel region 付きレベルスケジューリング**が使用される。

| 特性 | レベルスケジューリング (persistent) | ABMC |
|------|-----------------------------------|------|
| 並列粒度 | 行単位 | ブロック単位 |
| 同期回数 | レベル数 (数百) | 色数 (4〜数十) |
| 同期方式 | SpinBarrier (1回の parallel_for 内) | parallel_for 呼び出し (色ごと) |
| FEM での並列度 | 低い (深い依存チェーン) | 高い |
| セットアップ | O(nnz) — レベル計算のみ | O(nnz) — BFS + 彩色 + 行列並べ替え |
| 追加メモリ | レベル配列のみ | ワークベクトル 2×n + 行列コピー |
| apply時の割り当て | なし (事前確保済み) | なし (事前確保済み) |

**追加メモリの内訳 (ABMC パス)**:
- `abmc_x_perm_` (n), `abmc_y_perm_` (n) — ABMC固有
- `work_temp_` (n), `work_temp2_` (n) — 両パス共有
- `composite_perm_` (n), `composite_scaling_` (n) — 置換/スケーリング
- `reordered_csr_` — 行列並べ替え (factorization後に解放、`abmc_reorder_spmv=false` の場合)

### 6.2 Persistent Parallel Region

レベルスケジューリングの素朴な実装では、各レベルで `parallel_for` を呼び出す。レベル数が数百の場合、スレッドプール起動のオーバーヘッドが問題となる。

`forward_substitution_persistent()` は、単一の `parallel_for(nthreads, ...)` 内で全レベルを処理し、レベル間は SpinBarrier で同期する:

```cpp
parallel_for(nthreads, [&](index_t thread_id) {
    for (int lev = 0; lev < num_levels; ++lev) {
        // 各スレッドが担当範囲の行を処理
        const index_t my_start = level_size * thread_id / nthreads;
        const index_t my_end = level_size * (thread_id + 1) / nthreads;
        for (idx = my_start; idx < my_end; ++idx) { ... }
        barrier.wait();  // レベル間同期
    }
});
```

スレッド数 > 1 の場合に自動的にこのパスが選択される（`get_num_threads() > 1`）。

### 6.3 SpinBarrier の実装

`SpinBarrier` (`parallel.hpp`) はセンスリバーシングアルゴリズムを使用する:

```cpp
class SpinBarrier {
    alignas(64) std::atomic<int> count_;   // 到着スレッド数
    alignas(64) std::atomic<int> sense_;   // 現在の世代番号
    int num_threads_;
public:
    void wait() {
        int my_sense = sense_.load(acquire);
        if (count_.fetch_add(1, acq_rel) == num_threads_ - 1) {
            // 最後のスレッド: カウンタリセット + センス反転
            count_.store(0, relaxed);
            sense_.fetch_add(1, release);
        } else {
            // スピンウェイト (4096回後に yield)
            while (sense_.load(acquire) == my_sense) { ... }
        }
    }
};
```

**キャッシュライン分離**: `count_` と `sense_` は `alignas(64)` でそれぞれ独立したキャッシュラインに配置し、フォルスシェアリングを防止する。

**スピン→yield戦略**: 4096回のスピン後に `std::this_thread::yield()` を呼び出す。短い同期ではスピンで待ち、長い同期ではCPUを譲る。

---

## 7. 性能特性

### 7.1 ベンチマーク条件

- 8スレッド (NGSolve TaskManager)
- 3D HCurl curl-curl 問題 (order=2, `nograds=True`)
- 単位立方体メッシュ、shifted-ICCG (`shift=1.05, auto_shift=True, diagonal_scaling=True`)
- ベースライン: persistent parallel region 付きレベルスケジューリング
- 3回実行の最小値

### 7.2 スケーリング結果

| DOFs | Level Sched. | ABMC (best) | Speedup | 反復数 |
|------|-------------|-------------|---------|--------|
| 11K | 0.044s | 0.051s | **0.86x** | 31 |
| 27K | 0.158s | 0.154s | **1.02x** | 39 |
| 82K | 0.763s | 0.650s | **1.17x** | 59 |
| 186K | 2.505s | 1.962s | **1.28x** | 79 |

### 7.3 NB02 トロイダルコイルとの比較

[02_performance_comparison.ipynb](02_performance_comparison.ipynb) のトロイダルコイル問題 (148K DOFs) では:

| ソルバー | 反復数 | Wall Time | Speedup |
|---------|--------|-----------|---------|
| ICCG (level sched.) | 513 | 13.1s | 1.0x |
| ICCG + ABMC (8色) | 444 | 7.4s | **1.8x** |

単位立方体 (186K DOFs) では 1.28x、トロイダルコイル (148K DOFs) では 1.8x。差の要因:
- **反復数**: 513 vs 79。反復数が多いほど三角解法の時間比率が大きくなり、ABMC の効果が顕著
- **行列構造**: トロイダルコイルは帯域幅が大きく、レベルスケジューリングのレベル数がより多い → 同期オーバーヘッドが大きい → ABMC の優位性が増す

### 7.4 損益分岐点

ABMCには以下のオーバーヘッドがある:
1. **セットアップ**: BFS集約、ブロックグラフ構築、彩色、行列並べ替え
2. **apply時**: ベクトル置換 (2回の `parallel_for` による scatter/gather)

これらのオーバーヘッドが三角解法の高速化を上回る領域では、ABMCは逆効果になる。

**損益分岐点: 約 25K〜30K DOFs** (8スレッド、単位立方体、HCurl order 2)

- DOFs < 25K: レベルスケジューリングのほうが高速
- DOFs > 30K: ABMC が有利。問題サイズに比例して効果増大

### 7.5 効果を左右する要因

| 要因 | ABMC に有利 | ABMC に不利 |
|------|------------|------------|
| 問題サイズ | 大規模 (>30K DOFs) | 小規模 (<25K DOFs) |
| 反復数 | 多い (>100) | 少ない (<30) |
| 行列帯域幅 | 広い (深い依存チェーン) | 狭い |
| ABMC色数 | 少ない (4〜8) | 多い (>20) |
| スレッド数 | 多い (≥4) | 少ない (1〜2) |

---

## 8. 設定パラメータ

`SolverConfig` で制御される ABMC 関連パラメータ:

| パラメータ | 型 | 既定値 | 説明 |
|-----------|-----|--------|------|
| `use_abmc` | bool | false | ABMC 有効化 |
| `abmc_block_size` | int | 4 | BFS集約のブロックサイズ (行/ブロック) |
| `abmc_num_colors` | int | 4 | 彩色の目標色数 (下限、自動拡張あり) |
| `abmc_reorder_spmv` | bool | false | CG を ABMC 空間で実行 (SpMV も並べ替え) |
| `abmc_use_rcm` | bool | false | ABMC前に RCM 帯域幅縮小を適用 |

**推奨設定**: `use_abmc=True` のみ (他はデフォルト) で十分。

- `abmc_block_size`: 4〜16 で性能差は数%以内
- `abmc_num_colors`: ほぼ常に自動拡張されるため、指定値の影響は小さい
- `abmc_reorder_spmv`: FEMメッシュ順序のキャッシュ局所性を保つため、通常は false
- `abmc_use_rcm`: 追加の行列コピーとセットアップコストがかかるため、通常は false

---

## 9. ABMCSchedule のデータ構造

`ABMCSchedule` 構造体 (`abmc_ordering.hpp`) のメンバ:

```cpp
struct ABMCSchedule {
    // 色 → ブロック (CSR形式)
    std::vector<index_t> color_offsets;    // size: num_colors + 1
    std::vector<index_t> color_blocks;     // 各色のブロックIDリスト

    // ブロック → 行 (CSR形式, 連番のため実質オフセットのみ)
    std::vector<index_t> block_offsets;    // size: num_blocks + 1

    // 行の置換
    std::vector<index_t> ordering;         // ordering[old_row] = new_row
    std::vector<index_t> reverse_ordering; // reverse_ordering[new_row] = old_row
};
```

**三角解法での走査パターン**:
```
for c = 0 to num_colors - 1:
    blk_begin = color_offsets[c]
    blk_end = color_offsets[c + 1]
    parallel_for(blk_begin .. blk_end):
        blk = color_blocks[bidx]
        for i = block_offsets[blk] to block_offsets[blk+1] - 1:
            // ABMC空間の行 i を処理
```

**メモリ効率**: `block_offsets` が連続行範囲を定義するため、行→ブロック対応の配列は不要。ブロック $b$ の行は `[block_offsets[b], block_offsets[b+1])` で完全に決まる。

---

## 10. 既知の制限

### 10.1 色数の非予測性

`target_colors` はグリーディ彩色の下限であり、ブロック間の依存関係が密な場合に自動拡張される。複雑なメッシュ（ヘリカルコイル等）では指定値を大幅に超える可能性がある。

**改善候補**: 実際の色数をログ出力して、ユーザーが確認できるようにする。

### 10.2 ブロックサイズの自動調整

現在 `block_size` (既定: 4) はユーザー指定の固定値。行列構造に応じた自動調整は未実装。

ベンチマーク結果ではブロックサイズ 4〜16 の差は小さい（数%以内）。

### 10.3 RCM事前並べ替えの限定的効果

RCM は帯域幅縮小によりABMC彩色の色数を減らす可能性があるが、追加の行列コピーコストとセットアップ時間を要する。現在のベンチマークでは `abmc_use_rcm=False` と `True` の差は誤差範囲内であり、デフォルトは無効。

---

## 11. 改善履歴

このセクションでは、実装レビューに基づいて実施した改善を記録する。

### 11.1 `block_rows` 配列の除去

**課題**: `build_row_ordering()` で構築していた `block_rows[ridx]` は常に `ridx` に等しかった（新インデックスが連番で割り当てられるため、恒等写像）。三角解法で `i = block_rows[ridx]` とする代わりに `i = ridx` と書ける。

**対応**: `ABMCSchedule` から `block_rows` メンバを削除。三角解法ループを `for (i = row_begin; i < row_end; ++i)` に簡素化。O(n) のメモリ削減。

### 11.2 ワークベクトルの事前確保と共有

**課題**: レベルスケジューリングパスの `apply_level_schedule()` は毎回 `std::vector<Scalar> temp(size)` をヒープ割り当てしていた。ABMCパスは `setup()` 時に事前確保済みだったが、別の変数名 (`abmc_temp_`) で管理されていた。

**対応**: `work_temp_` (と `work_temp2_`) を両パス共有のメンバ変数に統一。`setup()` 時に1回確保し、`apply()` 時のヒープ割り当てを完全に排除。

### 11.3 `build_row_ordering` の未使用パラメータ除去

**課題**: `color_graph()` の出力パラメータと `build_row_ordering()` のパラメータに未使用のものがあった。

**対応**: `color_graph()` から不要な出力パラメータを削除。`build_row_ordering()` のシグネチャを簡素化。

---

## 参考文献

1. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel
   Multi-Threaded Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE IPDPS*, 2012.

2. E. Cuthill, J. McKee,
   "Reducing the Bandwidth of Sparse Symmetric Matrices",
   *Proc. 24th Nat. Conf. ACM*, pp. 157–172, 1969.

3. J. A. Meijerink, H. A. van der Vorst,
   "An Iterative Solution Method for Linear Systems of Which the
   Coefficient Matrix is a Symmetric M-Matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148–162, 1977.

# アルゴリズム解説

## 1. BDDC (Balancing Domain Decomposition by Constraints)

### 1.1 概要

BDDCは有限要素法の大規模問題に対する**領域分割前処理**である。
メッシュを要素ごとに「サブドメイン」と見なし、
サブドメイン間の結合を**粗空間 (coarse space)** で処理する。

**特徴**:
- 反復回数がメッシュサイズ *h* にほぼ依存しない (スケーラブル)
- 要素行列から直接構築できる (組立済み行列不要)
- NGSolveの組込みBDDCと同等の性能

**実装ファイル**: `preconditioners/bddc_preconditioner.hpp`

### 1.2 DOF分類

有限要素空間の自由度 (DOF) を2種類に分類する:

| 種類 | 英語名 | 役割 | NGSolve CouplingType |
|------|--------|------|---------------------|
| **Wirebasket** | 粗空間DOF | 頂点・辺のDOF。グローバルに連成 | `WIREBASKET_DOF` |
| **Interface** | 局所DOF | 面・内部のDOF。サブドメイン内で消去 | それ以外 |

高次有限要素 (order ≥ 2) では、wirebasket DOFは少数の低次DOFであり、
interface DOFが大多数を占める。BDDCの効率はこの階層性に依存する。

### 1.3 要素レベルのSchur補体

各要素 *e* の剛性行列を wirebasket (w) と interface (i) に分割する:

```
K^(e) = | K_ww  K_wi |
        | K_iw  K_ii |
```

**Schur補体** (wirebasket上の等価行列):

```
S^(e) = K_ww - K_wi * K_ii^{-1} * K_iw
```

**Harmonic extension** (wirebasketの値からinterface値を復元):

```
he^(e) = -K_ii^{-1} * K_iw
```

実装箇所: `BDDCPreconditioner::process_element()` (L169-273)

```cpp
// Schur complement
DenseMatrix<Scalar> K_ii_inv = K_ii;
K_ii_inv.invert();  // PA=LU 部分ピボット

DenseMatrix<Scalar> harm_ext = DenseMatrix<Scalar>::multiply(K_ii_inv, K_iw);
harm_ext.negate();  // he = -K_ii^{-1} * K_iw

// schur += K_wi * harm_ext = K_ww - K_wi * K_ii^{-1} * K_iw
DenseMatrix<Scalar>::multiply_add(K_wi, harm_ext, schur);
```

### 1.4 重み付け

共有DOF (複数要素に属するDOF) の寄与を正しく分配するため、
**重み付け (averaging)** を行う。

各要素のinterface DOF *k* に対する重み:

```
w_k^(e) = |K_ii^(e)(k,k)|   (K_iiの対角成分の絶対値)
```

重みの正規化: 全要素からの寄与を合計し、逆数をとる:

```
W_k = 1 / Σ_e w_k^(e)
```

この正規化後の重みで harmonic extension, inner solve, harmonic extension transpose
の行列要素をスケーリングする。

実装箇所: `BDDCPreconditioner::finalize_weights()` (L276-304)

### 1.5 粗空間ソルバー

全要素のSchur補体を組み立てた **wirebasket行列** S_global を直接法で解く。

| ソルバー | 設定値 | 用途 |
|---------|--------|------|
| SparseCholesky | `"sparsecholesky"` (既定) | 汎用、NGSolve組込み |
| PARDISO | `"pardiso"` | Intel MKL使用時 |
| 密LU | `"dense"` | 小規模問題、デバッグ用 |

SparseCholesky/PARDISOの場合、wirebasket CSR行列をNGSolveの
`SparseMatrix::InverseMatrix()` に渡す:

```cpp
// sparsesolv_precond.hpp: build_ngsolve_coarse_inverse()
sp_mat->SetInverseType(coarse_inverse_type_);  // "sparsecholesky" or "pardiso"
coarse_inv_ = sp_mat->InverseMatrix();
```

### 1.6 Applyアルゴリズム (5ステップ)

BDDCの前処理適用は以下の5ステップで行う。
入力 *r* は残差ベクトル、出力 *y* は前処理済みベクトル。

```
Step 1: y = r                              (コピー)
Step 2: y += H^T * r                       (harmonic extension transpose)
Step 3: y_wb = S^{-1} * y_wb              (wirebasket成分の粗空間ソルブ)
Step 4: y += K_inner * r                   (inner solve: interface成分)
Step 5: y = y + H * y                      (harmonic extension)
```

ここで:
- H: harmonic extension行列 (全要素の he^(e) を重み付けで組み立てたもの)
- H^T: その転置
- K_inner: 全要素の K_ii^{-1} を重み付けで組み立てたもの

実装箇所: `BDDCPreconditioner::apply_element_bddc()` (L316-366)

### 1.7 NGSolve BDDCとの比較

SparseSolvのBDDCはNGSolveの組込みBDDCと**数学的に同一のアルゴリズム**を実装しており、
反復回数は完全一致する。唯一の差異は要素行列の取得タイミング（Assembly中 vs 後）であり、
セットアップ時間に約1.3倍の差がある。ソルブ時間は同等。

トーラスコイル HCurl curl-curl 問題での比較:

| DOFs | SparseSolv setup | NGSolve setup | SparseSolv solve | NGSolve solve | 反復数 |
|------|-----------------|--------------|-----------------|--------------|--------|
| 38K (p=2) | 0.29s | 0.24s | 0.15s | 0.14s | 23 |
| 102K (p=3) | 1.09s | 0.74s | 0.65s | 0.67s | 40 |
| 309K (p=2) | 4.11s | 3.70s | 5.18s | 5.43s | 38 |
| 837K (p=3) | 10.82s | 7.96s | 14.89s | 15.01s | 61 |

詳細は [bddc_implementation_details.md](bddc_implementation_details.md) を参照。

---

## 2. IC分解 (不完全コレスキー分解)

### 2.1 概要

対称正定値行列 *A* に対して近似分解を求める:

```
A ≈ L * D * L^T
```

- *L*: 下三角行列 (Aと同じスパースパターン)
- *D*: 対角行列

前処理の適用: y = (LDL^T)^{-1} * x を前進代入・対角スケーリング・後退代入の3段階で解く。

**実装ファイル**: `preconditioners/ic_preconditioner.hpp`

### 2.2 シフトパラメータ

IC分解の対角に**シフト**を加えて安定化する:

```
d_i = α * a_ii - Σ_{k<i} l_ik^2 * d_k^{-1}
```

ここで α はシフトパラメータ (既定: 1.05)。

- α = 1.0: 標準IC(0)。不定値行列では破綻する場合がある
- α = 1.05: デフォルト。やや安定
- α = 1.1〜1.2: 困難な問題向け

### 2.3 Auto-shift

半正定値行列 (curl-curl問題) では、対角が小さくなりすぎて
IC分解が破綻することがある。Auto-shiftはこれを自動検出し、
シフトを増加させて分解を再試行する:

```cpp
// ic_preconditioner.hpp: compute_ic_factorization()
if (abs_s < config_.min_diagonal_threshold && abs_orig > 0.0) {
    shift += config_.shift_increment;  // シフト増加
    restart = true;                     // 分解を再開
}
```

### 2.4 対角スケーリング

行列の条件数を改善するため、対角スケーリングを適用できる:

```
scaling[i] = 1 / sqrt(|A[i,i]|)
A_scaled[i,j] = scaling[i] * A[i,j] * scaling[j]
```

実装箇所: `ICPreconditioner::compute_scaling_factors()`, `apply_scaling_to_L()`

### 2.5 三角解法の並列化

IC分解の適用 (前進代入・後退代入) は本質的に逐次的であるが、
データ依存性解析により部分的に並列化できる。

2つの方式を提供する:

| 方式 | ファイル | 並列粒度 | 特徴 |
|------|---------|---------|------|
| レベルスケジューリング | `level_schedule.hpp` | 行単位 | 単純、低オーバーヘッド |
| ABMC順序付け | `abmc_ordering.hpp` | ブロック単位 | 高並列性 |

詳細はセクション4 (ABMCオーダリング) を参照。

---

## 3. SGS-MRTR

### 3.1 概要

SGS-MRTR は**対称ガウス・ザイデル (SGS)** 前処理を**MRTR反復法**に
組み込んだソルバーである。分割公式 (split formula) により、
前処理の前進部 *L* と後退部 *L^T* を別々に適用する。

**特徴**:
- IC分解が不要 (分解コストなし)
- DAD変換 (対角スケーリング) が内蔵されている
- 複素対称行列にも対応 (ただし精度に制限あり)

**実装ファイル**: `solvers/sgs_mrtr_solver.hpp`

**参考文献**:
- 圓谷吉宏, 三船泰, 岩下武史, 高橋英治,
  "MRTR法に基づく前処理付き反復法の数値実験",
  *電気学会研究会資料*, SA-12-64, 2012.
- 圓谷吉宏, 三船泰, 岩下武史, 高橋英治,
  "Minimum Residual-Like Methods for Solving Ax = b with
  Shift-and-Invert Enhanced Multi-Step MRTR",
  *IEEE Trans. Magnetics*, Vol. 49, No. 5, pp. 1569-1572, 2013.

### 3.2 DAD変換

行列 *A* を対角スケーリングする:

```
D = diag(|A|)^{-1/2}
A' = D * A * D
b' = D * b
x  = D * x'
```

スケーリング後の行列 A' の対角は全て1に近くなり、
条件数が改善される。

### 3.3 split formula

SGSの前進部 L と後退部 L^T を個別に適用し、
MRTR反復の各ステップで:

```
rd = L^{-1} * r          (前進ソルブ)
u  = L^{-T} * rd         (後退ソルブ)
ARd = u + L^{-1}(rd - u) (近似 M^{-1}Ard)
```

### 3.4 MRTR反復

MRTRは最小残差型の反復法で、2項の再帰で解を更新する:

```
p_{k+1} = u_k + (η_k ζ_{k-1} / ζ_k) p_k
x_{k+1} = x_k + ζ_k p_k
```

ここで ζ_k, η_k は最適化パラメータ。

### 3.5 複素数対応の注意

SGS-MRTRの ζ_k 計算で除算が出現する際、複素数の比較に `std::real()` を使用:

```cpp
// sgs_mrtr_solver.hpp
if (std::abs(denom) < constants::DENOMINATOR_BREAKDOWN) {
    denom = (std::real(denom) >= 0) ? ... : ...;
}
```

複素対称行列 (A^T = A) の渦電流問題では、DAD変換が最適でない可能性があり、
収束精度が5%程度にとどまる場合がある。こうした場合はBDDC+CGを推奨する。

---

## 4. ABMC順序付け (Algebraic Block Multi-Color Ordering)

### 4.1 概要

三角解法 (前進代入・後退代入) の**並列化**のための行順序付けアルゴリズム。
レベルスケジューリングの問題点 (FEM行列での並列度不足) を解消する。

**実装ファイル**: `core/abmc_ordering.hpp`

**参考文献**:
- 岩下武史, 中島浩, 高橋康人,
  "Algebraic Block Multi-Color Ordering Method for Parallel
  Multi-Threaded Sparse Triangular Solver in ICCG Method",
  *Proc. IEEE IPDPS*, 2012.

### 4.2 アルゴリズム

ABMCは2段階で構成される:

#### Stage 1: 代数的ブロッキング (BFS集約)

行列のグラフ上でBFS (幅優先探索) により、近傍の行をブロックにまとめる。

```
入力: 行列グラフ G, ブロックサイズ B
出力: ブロック割当て block[i] (行i → ブロックID)

1. 未割当ての行 seed から BFS開始
2. BFS探索中に B 行を集約してブロック形成
3. 全行が割り当てられるまで繰り返し
```

ブロックサイズ `B` (既定: 4) は、ブロック内の行が逐次処理されるため、
キャッシュ効率とのバランスで決まる。

#### Stage 2: マルチカラーリング (グリーディ彩色)

ブロック隣接グラフ (ブロック間に下三角依存関係があるとき辺を張る) を
グリーディ彩色する。

```
入力: ブロック隣接グラフ, 目標色数 C
出力: ブロック色 color[b] (ブロックb → 色ID)

1. ブロック依存関係の隣接グラフ構築
2. グリーディ彩色: 各ブロックに未使用の最小色を割当て
```

目標色数 `C` (既定: 4) は並列粒度を制御する。

#### 彩色アルゴリズムの詳細

彩色は2段階で行われる:

1. **最小色数の決定**: `num_colors = max(target_colors, 1 + max_lower_triangular_degree)`
2. **Forbidden-Color-Set 法**: 各ブロックの下三角隣接色を forbidden 配列にマークし、最小の未使用色を割り当て。全色が衝突した場合、新しい色を追加。

`target_colors` は**下限**であり保証値ではない。実際の色数はブロックグラフの構造に依存する。

#### 既知の制限

- 複雑なメッシュ（ヘリカルコイル等）では色数が `target_colors` を大幅に超える可能性がある
- BFS ブロッキングは行列グラフの局所性に依存。帯域幅の大きい行列ではブロック間依存が密になり彩色数が増加

### 4.3 三角解法の並列実行

ABMC順序付け後の三角解法は3階層で実行される:

```
for 色 c = 0 to C-1:            ← 逐次 (色間依存性)
    parallel_for ブロック b ∈ 色c:  ← 並列 (同色ブロックは独立)
        for 行 i ∈ ブロックb:       ← 逐次 (ブロック内依存性)
            前進代入の1行処理
```

**レベルスケジューリングとの比較**:

| 特性 | レベルスケジューリング | ABMC |
|------|---------------------|------|
| 並列粒度 | 行単位 | ブロック単位 |
| 同期回数 | レベル数 (数百) | 色数 (4程度) |
| FEMでの並列度 | 低い (深い依存チェーン) | 高い |
| オーバーヘッド | 低い | 行列並べ替えコスト |

### 4.4 Persistent parallel region (レベルスケジューリング高速化)

標準のレベルスケジューリングでは各レベルで `parallel_for` を呼び出すため、
スレッドプール起動のオーバーヘッドが数百回発生する。

`forward_substitution_persistent()` / `backward_substitution_persistent()` は、
単一の `parallel_for(nthreads, ...)` 内で全レベルを処理し、
レベル間はスピンバリアで同期する:

```cpp
// ic_preconditioner.hpp
parallel_for(nthreads, [&](index_t thread_id) {
    for (int lev = 0; lev < num_levels; ++lev) {
        // 各スレッドが担当行を処理
        barrier.wait();  // レベル間同期
    }
});
```

### 4.5 RCM順序付けとの組み合わせ

RCM (Reverse Cuthill-McKee) 帯域縮小順序付けをABMCの前に適用することで、
行列の帯域幅を削減し、SpMV (行列ベクトル積) のキャッシュ効率を改善できる。

3つの実行パスがある:

| パス | SpMV行列 | 前処理 | 設定 |
|------|---------|--------|------|
| 標準 | 元の行列 | ABMC順序で三角解法 | `use_abmc=True` |
| RCM+ABMC | RCM行列 | RCM→ABMCの合成順序 | `use_abmc=True, abmc_use_rcm=True` |
| ABMC全空間 | ABMC行列 | 順序なし (直接) | `abmc_reorder_spmv=True` |

### 4.6 性能特性

ABMCの比較対象は**persistent parallel region付きレベルスケジューリング**である
（逐次レベルスケジューリングではない）。

3D HCurl curl-curl (order=2, 8スレッド) での実測:

| DOFs | Level Sched. | ABMC (best) | Speedup |
|------|-------------|-------------|---------|
| 11K | 0.044s | 0.051s | 0.86x |
| 27K | 0.158s | 0.154s | 1.02x |
| 82K | 0.763s | 0.650s | 1.17x |
| 186K | 2.505s | 1.962s | 1.28x |

**損益分岐点: 約 25K〜30K DOFs** (8スレッド)。小規模問題ではABMCの
セットアップコストとベクトル置換オーバーヘッドが三角解法の高速化を上回る。

問題構造にも依存し、トロイダルコイル (148K DOFs, 帯域幅大) では1.8xの高速化を
達成している ([02_performance_comparison.ipynb](02_performance_comparison.ipynb))。

詳細は [abmc_implementation_details.md](abmc_implementation_details.md) のセクション7を参照。

---

## 5. CG法 (共役勾配法)

### 5.1 概要

対称正定値 (SPD) 行列に対する**前処理付き共役勾配法 (PCG)**。

**実装ファイル**: `solvers/cg_solver.hpp`

### 5.2 アルゴリズム

```
r_0 = b - A*x_0
z_0 = M^{-1} * r_0
p_0 = z_0

for k = 0, 1, 2, ...:
    α_k = (r_k, z_k) / (p_k, A*p_k)
    x_{k+1} = x_k + α_k * p_k
    r_{k+1} = r_k - α_k * A*p_k
    z_{k+1} = M^{-1} * r_{k+1}
    β_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
    p_{k+1} = z_{k+1} + β_k * p_k
```

### 5.3 複素対称行列への対応

FEM行列は複素対称 (A^T = A) であり、**エルミートではない** (A^H ≠ A)。
例: 渦電流方程式 curl-curl + iσ mass。

この場合、内積は**非共役** (unconjugated) にする必要がある:

```
(a, b) = Σ a_i * b_i       (非共役: complex-symmetric)
(a, b) = Σ conj(a_i) * b_i  (共役: Hermitian)
```

実装での切替え:

```cpp
// iterative_solver.hpp: dot_product()
if (config_.conjugate)
    sum += std::conj(a[i]) * b[i];  // Hermitian
else
    sum += a[i] * b[i];  // complex-symmetric (default)
```

NGSolveの `CGSolver(conjugate=False)` に対応する。

### 5.4 収束判定

相対残差ノルムで判定:

```
||r_k|| / ||r_0|| < tol
```

追加機能:
- **best-result追跡**: 最良の解を保持し、未収束時はそれを返す
- **発散検出**: 残差が最良値の閾値倍を超えると早期終了
- **残差履歴**: 各反復の残差を記録

---

## 6. DenseMatrix LU逆行列

### 6.1 概要

BDDC内部で使用される小規模密行列の LU分解と逆行列計算。
PA = LU 形式の部分ピボット付きLU分解を行う。

**実装ファイル**: `core/dense_matrix.hpp`

### 6.2 置換行列の構築

部分ピボット付きLU分解では、行交換を記録する置換ベクトル `piv[]` が得られる。
置換行列 P は次のように構築する:

```
P[k, piv[k]] = 1   (正しい: PA = LU)
```

**注意**: `P[piv[k], k] = 1` は P^T (転置) であり、間違い。
この区別は対角優位な行列 (H1) では影響しにくいが、
非自明なピボット交換が発生する行列 (HCurl) では致命的になる。

```cpp
// dense_matrix.hpp: invert()
DenseMatrix inv(n, n);
for (index_t k = 0; k < n; ++k) {
    inv(k, piv[k]) = Scalar(1);  // P[k, piv[k]] = 1
}
```

### 6.3 逆行列計算の3段階

```
1. LU分解: PA = LU (部分ピボット)
2. 前進代入: L * Y = P (P は置換行列)
3. 後退代入: U * X = Y → X = A^{-1}
```

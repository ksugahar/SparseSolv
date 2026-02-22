# SparseSolv BDDC実装 — 詳細アーキテクチャ・実装ガイド

本ドキュメントでは、SparseSolv が提供する BDDC (Balancing Domain Decomposition by Constraints) 前処理の理論と実装詳細を、**他の開発者が自分のソルバーに BDDC を組み込む**際のリファレンスとなるレベルで解説します。

本実装は NGSolve 組み込み BDDC と**数学的に同一のアルゴリズム** [Dohrmann 2003] を、独立した pybind11 拡張モジュールとして再実装したものです。代数的定式化は [Mandel, Dohrmann, Tezaur 2005] に基づきます。
要素行列の抽出を Assembly 後に `CalcElementMatrix` で再計算する点のみが異なり、それ以外のセットアップ処理・Apply 操作・並列性能・反復回数・ソルブ時間は NGSolve 組み込み BDDC と同等です。

---

## 1. 理論的背景と概念モデル

BDDCは非オーバーラップ領域分割法 (Domain Decomposition Method) の一種であり、対象の計算領域（メッシュ）を要素単位に分離し、要素内部 (Interface) と要素境界の拘束部分 (Wirebasket: 頂点・辺など) に問題を分割することで、条件数を大幅に改善します。

### 1.1 DOF (自由度) の分類
系全体、および各要素の自由度ベクトルの成分を以下の2つに分割します。
*   **$I$ (Interface / Internal)**: 要素内部に閉じている、局所的に消去可能な自由度（面・内部DOF）。
*   **$W$ (Wirebasket / Coarse)**: 複数の要素間で共有され、全体の大域的な繋がり（粗格子問題）を担う自由度（頂点・辺DOF）。

各要素 $e$ の局所剛性行列 $K^{(e)}$ は、この $I$ と $W$ の順序で並べ替えることで以下のようにブロック化できます。

$$
K^{(e)} = \begin{pmatrix}
K_{II}^{(e)} & K_{IW}^{(e)} \\
K_{WI}^{(e)} & K_{WW}^{(e)}
\end{pmatrix}
$$

### 1.2 Schur補体と粗空間問題
BDDCの中核は、各要素の $I$ 自由度を「局所的」に静的消去 (Static Condensation) して $W$ 自由度のみからなる大域的な「粗空間問題 (Coarse Problem)」を構築することです。

1要素あたりの局所Schur補体 $S_{WW}^{(e)}$ は次のように定義されます：
$$ S_{WW}^{(e)} = K_{WW}^{(e)} - K_{WI}^{(e)} (K_{II}^{(e)})^{-1} K_{IW}^{(e)} $$

これらを全要素にわたって足し合わせた（アセンブルした）ものが、大域的なWirebasket粗行列 $S_{WW}$ となります。
$$ S_{WW} = \sum_{e} R_W^{(e)T} S_{WW}^{(e)} R_W^{(e)} $$
ここで $R_W^{(e)}$ は要素内の $W$ 自由度から大域的 $W$ 自由度への抽出マトリクスです。

---

## 2. 実装アルゴリズム：セットアップフェーズ

### Step 2.1: FEMフレームワークからの情報抽出

BDDC の構築に必要な入力データは以下の4点のみです。FEMフレームワーク依存の部分はここだけであり、これらを用意できれば任意のソルバーに組み込めます。

#### 必要な入力データ一覧

| # | データ | 型 | サイズ | 説明 |
|---|--------|------|--------|------|
| 1 | **要素DOFリスト** | `vector<vector<int>>` | 要素数 × 各要素のDOF数 | 各要素に属するDOFの全体番号 |
| 2 | **要素行列** | `vector<DenseMatrix>` | 要素数 × (n_el × n_el) | 各要素の局所剛性行列 |
| 3 | **DOF分類** | `vector<DOFType>` | 全DOF数 | 各DOFが Wirebasket か Interface か |
| 4 | **自由度マスク** | `vector<bool>` | 全DOF数 | Dirichlet拘束DOFの識別（任意） |

#### NGSolve からの具体的な抽出方法

以下は NGSolve 固有のコードですが、他のFEMフレームワークでも同等の情報を取得できます。

**要素行列の再計算** — 要素行列は組み立て済みの疎行列からは逆算できないため、積分子を使って再計算します：

```cpp
// 体積積分子のみフィルタリング
Array<shared_ptr<BilinearFormIntegrator>> vol_integrators;
for (auto& integ : bfa->Integrators())
    if (integ->VB() == VOL)
        vol_integrators.Append(integ);

// 要素並列ループ（IterateElements は OpenMP/TaskManager で並列化）
LocalHeap lh(10000000, "bddc_setup", true);  // true = スレッド数分確保

IterateElements(*fes, VOL, lh,
    [&](FESpace::Element el, LocalHeap& lh_thread) {
        size_t elnr = el.Nr();

        // 1. 要素DOFリストの取得
        auto dnums = el.GetDofs();  // Array<DofId>

        // 2. 正値DOFのフィルタリング（後述「注意点」参照）
        vector<int> valid_local;    // 要素内局所番号
        vector<int> valid_global;   // 全体DOF番号
        for (int i = 0; i < dnums.Size(); ++i) {
            if (IsRegularDof(dnums[i])) {  // 負値DOFを除外
                valid_local.push_back(i);
                valid_global.push_back(dnums[i]);
            }
        }
        element_dofs[elnr] = valid_global;

        // 3. 要素行列の計算
        auto& fe = el.GetFE();      // FiniteElement（形状関数）
        auto& trafo = el.GetTrafo(); // ElementTransformation（Jacobian等）

        int ndof_el = dnums.Size();
        FlatMatrix<SCAL> elmat(ndof_el, ndof_el, lh_thread);
        elmat = SCAL(0);

        for (auto& integrator : vol_integrators) {
            FlatMatrix<SCAL> contrib(ndof_el, ndof_el, lh_thread);
            contrib = SCAL(0);
            integrator->CalcElementMatrix(fe, trafo, contrib, lh_thread);
            elmat += contrib;
        }

        // 4. 正値DOFに対応する部分行列の抽出
        int nvalid = valid_local.size();
        DenseMatrix<SCAL> dm(nvalid, nvalid);
        for (int i = 0; i < nvalid; ++i)
            for (int j = 0; j < nvalid; ++j)
                dm(i, j) = elmat(valid_local[i], valid_local[j]);
        element_matrices[elnr] = dm;
    });
```

`CalcElementMatrix` が内部で必要とする情報:

| 情報 | 取得元 | 内容 |
|------|--------|------|
| `FiniteElement` (fe) | `el.GetFE()` | 要素の形状関数（次数・型） |
| `ElementTransformation` (trafo) | `el.GetTrafo()` | 参照要素 → 物理座標の写像（Jacobian等） |
| `BilinearFormIntegrator` | `bfa->Integrators()` | 弱形式の定義（例: `curl(u)*curl(v)*dx`） |
| `LocalHeap` (lh) | スレッドローカル確保 | 一時メモリ（要素行列の作業領域） |

**DOF分類の取得**:

```cpp
vector<DOFType> dof_types(ndof);
for (size_t d = 0; d < ndof; ++d) {
    auto ct = fes->GetDofCouplingType(d);
    dof_types[d] = (ct == WIREBASKET_DOF)
        ? DOFType::Wirebasket
        : DOFType::Interface;
}
```

| FESpace | order=1 | order=2 | order=3+ |
|---------|---------|---------|----------|
| H1 | 全頂点=WB | 頂点=WB, 辺=IF | 頂点=WB, 辺/面/内部=IF |
| HCurl | 辺=WB | 辺=WB, 面=IF | 辺=WB, 面/内部(LOCAL)=IF |

**自由度マスクの取得**:

```cpp
auto freedofs = fes->GetFreeDofs(false);  // false = LOCAL_DOFを含める
```

> **重要**: `GetFreeDofs(false)` の `false` は LOCAL_DOF を含めるために必須です。order≥3 では LOCAL_DOF（要素内部に閉じた自由度）が存在し、`GetFreeDofs(true)` で除外すると BDDC の Interface ブロック構造が破綻します。

#### NGSolve 組み込み BDDC との差異

| 項目 | SparseSolv BDDC | NGSolve BDDC |
|------|----------------|-------------|
| 要素行列取得 | Assembly **後**に `CalcElementMatrix` で再計算 | Assembly **中**にフックで取得（追加コストゼロ） |
| API | `BDDCPreconditioner(a, fes)` は Assemble **後**に呼ぶ | `Preconditioner(a, "bddc")` は Assemble **前**に登録 |

これがセットアップ時間が約 **1.3 倍**になる唯一の原因です。

---

### Step 2.2: DOFマッピングの構築

全DOFを Wirebasket と Interface に分類し、コンパクトな番号付けを行います。

```
入力: dof_types[0..N-1], free_dofs[0..N-1]
出力: wb_dofs[]  — コンパクトWB番号 → 全体DOF番号
      if_dofs[]  — コンパクトIF番号 → 全体DOF番号
      wb_map[]   — 全体DOF番号 → コンパクトWB番号（WBでなければ -1）

for d = 0 to N-1:
    if free_dofs[d] == false:  continue  // Dirichlet拘束DOFはスキップ
    if dof_types[d] == Interface:
        if_dofs.append(d)
    else:
        wb_map[d] = len(wb_dofs)
        wb_dofs.append(d)

n_wb = len(wb_dofs)
n_if = len(if_dofs)
```

このマッピングは3箇所で使用されます:
1. 要素ループ内の DOF分類（local_wb / local_if の振り分け）
2. Wirebasket Schur 補体の COO 蓄積（全体番号 → コンパクトWB番号への変換）
3. Apply 時の WB ベクトル抽出・散布

---

### Step 2.3: 要素ごとの局所演算と重み付き蓄積

各要素は互いに完全に独立しており、並列化可能です。以下に1要素の処理の完全な擬似コードを示します。

```
入力: el_dofs[0..nel-1]  — この要素のDOF全体番号
      elmat[nel × nel]   — 要素剛性行列
      dof_types[]        — 全体DOF分類
      free_dofs[]        — 自由度マスク
      wb_map[]           — 全体→コンパクトWB番号

出力: COO行列 (he_coo, het_coo, is_coo, wb_coo) への寄与
      weight[]           — 重み累積配列への加算

// --- ステップ1: 要素DOFをWBとIFに分類 ---
local_wb = []  // 要素内局所インデックス（WB）
local_if = []  // 要素内局所インデックス（IF）
for k = 0 to nel-1:
    d = el_dofs[k]
    if d < 0: continue           // 負値DOFはスキップ（注意点参照）
    if not free_dofs[d]: continue // Dirichlet拘束
    if dof_types[d] == Interface:
        local_if.append(k)
    else:
        local_wb.append(k)

nw = len(local_wb)
ni = len(local_if)
if nw == 0: return  // WB DOFがない要素はスキップ

// --- ステップ2: ブロック行列の抽出 ---
K_ww[nw × nw]:  K_ww[i,j] = elmat[local_wb[i], local_wb[j]]
K_wi[nw × ni]:  K_wi[i,j] = elmat[local_wb[i], local_if[j]]
K_iw[ni × nw]:  K_iw[i,j] = elmat[local_if[i], local_wb[j]]
K_ii[ni × ni]:  K_ii[i,j] = elmat[local_if[i], local_if[j]]

// --- ステップ3: 要素レベル重みの計算 ---
elem_weight[k] = |K_ii[k,k]|   (k = 0..ni-1)
// ※ K_ii の対角成分の絶対値。共有DOFの寄与を測る指標
// ※ 全体の重み配列に累積加算:
for k = 0 to ni-1:
    weight[el_dofs[local_if[k]]] += elem_weight[k]

// --- ステップ4: Schur補体の計算 ---
K_ii_inv = inverse(K_ii)                          // 密LU分解
H = -K_ii_inv * K_iw                              // Harmonic extension (ni × nw)
S_ww = K_ww + K_wi * H                            // Schur補体 (nw × nw)
  // 展開すると S_ww = K_ww - K_wi * K_ii^{-1} * K_iw

H_T = -K_wi * K_ii_inv                            // H^T に相当 (nw × ni)

// --- ステップ5: 要素レベル重みの適用 ---
// ここが BDDC の multiplicityスケーリングの核心部分。
// 重みは「このDOFを共有する全要素の寄与の和で割る」ための準備段階。
// 最終的な重みは finalize_weights() で w[i] = 1/weight[i] として確定する。

for k = 0 to ni-1:
    H[k, :] *= elem_weight[k]       // 行スケーリング
for l = 0 to ni-1:
    H_T[:, l] *= elem_weight[l]     // 列スケーリング
for k = 0 to ni-1:
    K_ii_inv[k, :] *= elem_weight[k]  // 行スケーリング
for l = 0 to ni-1:
    K_ii_inv[:, l] *= elem_weight[l]  // 列スケーリング

// --- ステップ6: COO行列への蓄積 ---
// 全体DOF番号で COO に追加（重複エントリは to_csr() で合算される）
g_if[k] = el_dofs[local_if[k]]   (k = 0..ni-1)
g_wb[k] = el_dofs[local_wb[k]]   (k = 0..nw-1)
c_wb[k] = wb_map[g_wb[k]]        (k = 0..nw-1)  // コンパクトWB番号

he_coo.add_submatrix(rows=g_if, cols=g_wb, matrix=H)         // ni × nw
het_coo.add_submatrix(rows=g_wb, cols=g_if, matrix=H_T)       // nw × ni
is_coo.add_submatrix(rows=g_if, cols=g_if, matrix=K_ii_inv)   // ni × ni
wb_coo.add_submatrix(rows=c_wb, cols=c_wb, matrix=S_ww)       // nw × nw
```

> **ポイント**: `he_coo`, `het_coo`, `is_coo` は全空間 ($N \times N$) の行列、`wb_coo` のみコンパクト番号付け ($n_{wb} \times n_{wb}$) の行列です。同じ (row, col) に複数の要素から寄与がある場合、COO → CSR 変換時に自動的に合算されます。

---

### Step 2.4: 重みの最終化と CSR 行列へのスケーリング適用

全要素の処理が終わった後、累積された重みを反転し、構築済みの CSR 行列に適用します。

```
// --- 重みの反転 ---
for i = 0 to N-1:
    if weight[i] > 0:
        weight[i] = 1.0 / weight[i]
    // weight[i] は「DOF i を含む全要素の |K_ii(i,i)| の和」の逆数
    // これが multiplicity スケーリング w_i に相当

// --- CSR 行列へのスケーリング ---
// Inner solve: is[i,j] *= w[i] * w[j]
for each row i in is_csr:
    wi = weight[i]
    for each entry (i, j, val) in row i:
        val *= wi * weight[j]

// Harmonic extension: he[i,:] *= w[i]  (行スケーリング)
for each row i in he_csr:
    wi = weight[i]
    for each entry in row i:
        val *= wi

// Harmonic extension transpose: het[:,j] *= w[j]  (列スケーリング)
for each row i in het_csr:
    for each entry (i, j, val) in row i:
        val *= weight[j]
```

> **なぜ2段階のスケーリングか？**: 要素ループ内で `elem_weight[k]` を乗じ、最終化で `1/weight[i]` を乗じることで、全体として $w_i = |K_{II,ii}^{(e)}| / \sum_{e'} |K_{II,ii}^{(e')}|$ という「自分の要素の寄与 ÷ 全要素の寄与合計」のスケーリングが実現されます。これは共有DOFの multiplicity（多重度）を自然に重み付ける標準的な手法です。

---

### Step 2.5: COO → CSR 変換

COO 蓄積データを CSR 形式に変換します。変換アルゴリズム:

```
入力: entries[] = [(row, col, value), ...]
出力: CSR {row_ptr, col_idx, values}

// 1. (row, col) でソート
sort entries by (row, col)

// 2. 重複エントリの合算
merged = []
for each entry in sorted_entries:
    if entry.row == merged.back().row and entry.col == merged.back().col:
        merged.back().value += entry.value
    else:
        merged.append(entry)

// 3. row_ptr の構築
row_ptr[0..rows] = 0
for each (row, col, val) in merged:
    row_ptr[row + 1]++
for i = 1 to rows:
    row_ptr[i] += row_ptr[i-1]

col_idx = [e.col for e in merged]
values  = [e.value for e in merged]
```

---

### Step 2.6: 粗ソルバー (Coarse Solver) の初期化

Wirebasket Schur 補体 `wb_csr_` ($n_{wb} \times n_{wb}$) に対する直接法ソルバーを構築します。

| `coarse_inverse` | 方式 | 用途 |
|-------------------|------|------|
| `"sparsecholesky"` | NGSolve SparseCholesky | 標準（既定） |
| `"pardiso"` | Intel PARDISO | 大規模 wirebasket 向け |
| `"dense"` | SparseSolv 密 LU | テスト用 |

粗ソルバーは **コールバック関数** `void(const Scalar* rhs, Scalar* sol)` として BDDC コアに渡されます。コールバックのインターフェース:

- **入力**: `rhs` — コンパクト WB 番号付けの $n_{wb}$ 次元ベクトル
- **出力**: `sol` — コンパクト WB 番号付けの $n_{wb}$ 次元ベクトル
- **処理**: $\text{sol} = S_{WW}^{-1} \cdot \text{rhs}$

NGSolve の `InverseMatrix` を利用する場合の具体的な手順:

```cpp
// 1. CSR → NGSolve SparseMatrix 変換
Array<int> elsperrow(n_wb);
for (int i = 0; i < n_wb; ++i)
    elsperrow[i] = wb_csr.row_ptr[i+1] - wb_csr.row_ptr[i];
auto sp_mat = make_shared<SparseMatrix<SCAL>>(elsperrow, n_wb);
for (int i = 0; i < n_wb; ++i) {
    auto cols = sp_mat->GetRowIndices(i);
    auto vals = sp_mat->GetRowValues(i);
    int off = wb_csr.row_ptr[i];
    for (int k = 0; k < elsperrow[i]; ++k) {
        cols[k] = wb_csr.col_idx[off + k];
        vals[k] = wb_csr.values[off + k];
    }
}

// 2. 直接法ソルバーの構築
sp_mat->SetInverseType("sparsecholesky");  // or "pardiso"
auto coarse_inv = sp_mat->InverseMatrix();

// 3. コールバック関数として登録
// VVector はデータコピーを介して rhs/sol をやりとり
auto coarse_rhs = make_shared<VVector<SCAL>>(n_wb);
auto coarse_sol = make_shared<VVector<SCAL>>(n_wb);

bddc_precond->set_coarse_solver(
    [=](const SCAL* rhs, SCAL* sol) {
        // コンパクトWBベクトル → NGSolve BaseVector にコピー
        auto rhs_fv = coarse_rhs->FV<SCAL>();
        for (int i = 0; i < n_wb; ++i)
            rhs_fv[i] = rhs[i];
        // 直接法ソルブ
        coarse_inv->Mult(*coarse_rhs, *coarse_sol);
        // NGSolve BaseVector → コンパクトWBベクトルにコピー
        auto sol_fv = coarse_sol->FV<SCAL>();
        for (int i = 0; i < n_wb; ++i)
            sol[i] = sol_fv[i];
    });
```

> **メモリ寿命の注意**: `sp_mat`（NGSolve SparseMatrix）は `coarse_inv` が参照しているため、`coarse_inv` より長く生存させる必要があります。`shared_ptr` で保持してください。

---

## 3. 実装アルゴリズム：前処理適用 (Apply) フェーズ

前処理 $y = M_{BDDC}^{-1} x$ は、セットアップ時に構築した CSR 疎行列による SpMV (疎行列ベクトル積) で実行されます。**要素ループは不要です**。

### 完全な擬似コード

```
入力: x[0..N-1]     — 残差ベクトル（全空間）
出力: y[0..N-1]     — 前処理後のベクトル（全空間）
作業: work1[0..N-1], work2[0..N-1]         — 全空間作業ベクトル
      wb_work1[0..n_wb-1], wb_work2[0..n_wb-1]  — WB空間作業ベクトル

// Step 1: y = x (コピー)
y = copy(x)

// Step 2: y += H^T * x (Harmonic extension 転置)
work1 = het_csr * x        // SpMV: het_csr_ (N×N, 非零は WB行・IF列のみ)
y += work1

// Step 3: 粗空間ソルブ (Wirebasket inverse)
// y から WB 成分を抽出 → コンパクト WB ベクトルで直接法ソルブ
for k = 0 to n_wb-1:
    wb_work1[k] = y[wb_dofs[k]]

coarse_solve(wb_work1, wb_work2)   // wb_work2 = S_WW^{-1} * wb_work1

// WB ソルブ結果を全空間に散布、IF 成分は 0
work1 = zeros(N)
for k = 0 to n_wb-1:
    work1[wb_dofs[k]] = wb_work2[k]

// Step 4: Inner solve 加算
work2 = is_csr * x         // SpMV: is_csr_ (N×N, 非零は IF行・IF列のみ)
work1 += work2

// Step 5: Harmonic extension 適用
work2 = he_csr * work1     // SpMV: he_csr_ (N×N, 非零は IF行・WB列のみ)
y = work1 + work2
```

### 計算コスト

| 操作 | FLOPs | 備考 |
|------|-------|------|
| SpMV × 3 (het, is, he) | $O(\text{nnz})$ | 疎行列ベクトル積 |
| 粗ソルブ | $O(n_{wb})$ | 前処理済み直接法の前進・後退代入 |
| ベクトル加算・コピー × 4 | $O(N)$ | |

$H$, $H^T$, $IS$ の各疎行列には、セットアップ時に multiplicity 重みが適用済みのため、Apply 時の追加スケーリングは不要です。

CSR SpMV は `parallel_for` で行並列化されており、NGSolve 組み込み BDDC と同等の並列性能を持ちます:

```cpp
void multiply(const Scalar* x, Scalar* y) const {
    parallel_for(rows, [&](index_t i) {
        Scalar sum = Scalar(0);
        for (index_t k = row_ptr[i]; k < row_ptr[i+1]; ++k)
            sum += values[k] * x[col_idx[k]];
        y[i] = sum;
    });
}
```

---

## 4. 実装上の注意点

### 4.1 負値DOF ID

NGSolve では `el.GetDofs()` が返す DOF 番号に**負値**が含まれる場合があります。負値は interface-only DOF（そのDOFが他の要素にのみ属することを示す）であり、要素行列中の対応する行・列は使用しません。`IsRegularDof(d)` で正値 DOF のみをフィルタリングしてください。

```cpp
// dnums[i] が正値かチェック（NGSolve固有）
if (IsRegularDof(dnums[i])) {
    valid_local.push_back(i);
    valid_global.push_back(dnums[i]);
}
```

### 4.2 LOCAL_DOF と order≥3

order≥3 の有限要素空間では `LOCAL_DOF`（要素内部に完全に閉じたDOF）が存在します。これらは:
- `WIREBASKET_DOF` でないため Interface に分類される
- `GetFreeDofs(true)` では**除外される**（LOCAL は他要素と共有しないため通常の自由度マスクに含まれない）
- `GetFreeDofs(false)` では**含まれる**

BDDC では LOCAL_DOF も Interface ブロックの一部として正しく消去する必要があるため、**必ず `GetFreeDofs(false)` を使用**してください。

### 4.3 複素数対応

テンプレートパラメータ `Scalar` を `std::complex<double>` にすることで複素行列に対応できます。注意点:

- **重み計算**: `elem_weight[k] = std::abs(K_ii(k,k))` — 絶対値を使用
- **内積**: FEM行列が複素対称 ($A^T = A$) の場合は非共役内積 $\langle u, v \rangle = u^T v$ を使用。CG ソルバー側で `conjugate=False` を設定
- **CSR SpMV**: `sum += values[k] * x[col_idx[k]]` — 共役は不要

### 4.4 セットアップ後のメモリ管理

セットアップ完了後、要素行列は不要になります。メモリを節約するため明示的に解放してください:

```cpp
element_matrices_.clear();
element_matrices_.shrink_to_fit();  // メモリを実際に解放
```

Apply フェーズで使用する作業ベクトルはセットアップ時に事前確保しておくことで、Apply のたびの動的確保を避けます。

### 4.5 Interface DOF が無い要素

低次要素 (order=1) では、一部の要素が全て Wirebasket DOF のみで構成される場合があります。この場合 $K_{II}$ ブロックが空になるため、Schur 補体の計算は不要で、$K_{WW}$ をそのまま `wb_coo` に追加します:

```
if ni == 0:
    wb_coo.add_submatrix(rows=c_wb, cols=c_wb, matrix=K_ww)
    return  // H, H_T, IS の寄与なし
```

---

## 5. NGSolve 組み込み BDDC との比較

本実装は NGSolve 組み込み BDDC と同一のアルゴリズム（要素単位 Schur 補体 + wirebasket 粗空間）を用いています。
唯一の差異は**要素行列の取得タイミング**です。

### セットアップの差異

| 処理 | SparseSolv BDDC | NGSolve BDDC |
|------|----------------|-------------|
| 要素行列取得 | Assembly **後**に `CalcElementMatrix` で再計算 | Assembly **中**にフックで取得（追加コストゼロ） |
| Schur 補体計算 | 同一（密LU分解） | 同一 |
| 粗行列構築 | COO → CSR 変換 | NGSolve 内部フォーマット |
| 粗ソルバー | NGSolve `InverseMatrix` 利用 | 同一 |

要素行列の再計算がセットアップ時間の差の主因であり、**セットアップ時間は約 1.3 倍**です。

### Apply の同等性

Apply フェーズは両実装とも CSR SpMV ベースであり、計算量・並列性能は同等です。

### ベンチマーク結果

トーラスコイル HCurl curl-curl 問題での比較:

| 項目 | SparseSolv BDDC | NGSolve BDDC |
|------|----------------|-------------|
| 反復回数 | **完全一致** (全スケールで確認) | 同左 |
| ソルブ時間 | 同等 | 同等 |
| セットアップ時間 | **約 1.3 倍** | (基準) |

| DOFs | SparseSolv setup | NGSolve setup | SparseSolv solve | NGSolve solve | 反復数 |
|------|-----------------|--------------|-----------------|--------------|--------|
| 38K (p=2) | 0.29s | 0.24s | 0.15s | 0.14s | 23 |
| 102K (p=3) | 1.09s | 0.74s | 0.65s | 0.67s | 40 |
| 309K (p=2) | 4.11s | 3.70s | 5.18s | 5.43s | 38 |
| 837K (p=3) | 10.82s | 7.96s | 14.89s | 15.01s | 61 |

**結論**: 反復数は完全一致。ソルブ時間は同等。セットアップのみ要素行列再計算分の差がある。

---

## 6. ソースコード構成

| ファイル | 内容 | 依存 |
|---------|------|------|
| `core/sparse_matrix_coo.hpp` | COO 蓄積・CSR 変換 | なし |
| `core/sparse_matrix_csr.hpp` | CSR 格納・SpMV | `core/parallel.hpp` |
| `core/dense_matrix.hpp` | 密行列演算（LU逆行列・乗算） | なし |
| `preconditioners/bddc_preconditioner.hpp` | **BDDC コア**（FEM非依存） | 上記3ファイル |
| `ngsolve/sparsesolv_precond.hpp` | NGSolve BaseMatrix ラッパー | BDDC コア + NGSolve |
| `ngsolve/sparsesolv_python_export.hpp` | pybind11 バインディング + 要素抽出 | ラッパー + pybind11 |

BDDC コア (`bddc_preconditioner.hpp`) は NGSolve に一切依存しません。入力データ（要素DOFリスト・要素行列・DOF分類・自由度マスク）を用意できれば、任意の FEM フレームワークで利用可能です。

---

## 7. この実装の特性

1.  **NGSolve BDDC と数学的に同一**: 反復数・精度が完全一致することを確認済み。
2.  **FEM非依存コア**: BDDC コアは NGSolve に依存せず、入力データのみで動作。
3.  **要素完全独立並列**: セットアップの要素ループは `IterateElements` で並列化。Apply は CSR SpMV で行並列化。
4.  **Shifted-BDDC**: 構築行列と系行列を分離可能（ただし単連結領域のみ。後述）。
5.  **複素数対応**: テンプレートにより `double` / `complex<double>` の両方に対応。
6.  **API の違い**: `BDDCPreconditioner(a, fes)` は `a.Assemble()` 後に呼ぶ。NGSolve の `Preconditioner(a, "bddc")` は `a.Assemble()` 前に登録する必要がある。

---

## 8. Python API

```python
pre = BDDCPreconditioner(a, fes, coarse_inverse="sparsecholesky")
inv = CGSolver(mat=a.mat, pre=pre, maxiter=500, tol=1e-8)
gfu.vec.data = inv * f.vec
```

| 引数 | 型 | 既定値 | 説明 |
|------|------|--------|------|
| `a` | `BilinearForm` | 必須 | **組立済み**の双線形形式 |
| `fes` | `FESpace` | 必須 | 有限要素空間 |
| `coarse_inverse` | `str` | `"sparsecholesky"` | 粗空間ソルバー |

| プロパティ | 型 | 説明 |
|-----------|------|------|
| `num_wirebasket_dofs` | `int` | Wirebasket DOF 数 |
| `num_interface_dofs` | `int` | Interface DOF 数 |

### 対応する FESpace と行列型

| FESpace | 実数 | 複素数 | 備考 |
|---------|------|--------|------|
| H1 | OK | OK | |
| HCurl | OK | OK | `nograds=True` 推奨 |
| HDiv | OK | OK | |

| 行列型 | 対応 | CG の `conjugate` |
|--------|------|------------------|
| 実対称 | OK | `False`（既定） |
| 複素対称 ($A^T=A$) | OK | `False` |
| 複素エルミート ($A^H=A$) | OK | `True` |

---

## 9. Shifted-BDDC（前処理と系行列の分離）— 単連結領域のみ

BDDC の構築行列と CG の系行列を分離する「Shifted-BDDC」:

```python
# ε付き行列で BDDC を構築（正定値化）
a_shift = BilinearForm(curl(u)*curl(v)*dx + eps*u*v*dx)
a_shift.Assemble()
pre = BDDCPreconditioner(a_shift, fes)

# ε=0 の行列で CG を実行（正確な解）
a_pure = BilinearForm(curl(u)*curl(v)*dx)
a_pure.Assemble()
inv = CGSolver(mat=a_pure.mat, pre=pre, maxiter=500, tol=1e-8)
```

### 単連結領域では有効

`nograds=True` を使用すると、勾配空間（curl のカーネルの自明な部分）が FE 空間から除去される。
単連結領域では、これにより curl-curl 行列は正定値となり、Shifted-BDDC は正常に動作する。

| メッシュ (Unit Cube) | DOFs (p=2) | Shifted-BDDC | Standard BDDC |
|----------------------|-----------|--------------|---------------|
| maxh=0.15 | 13K | 30 反復 OK | — |
| maxh=0.10 | 28K | 31 反復 OK | — |
| maxh=0.07 | 86K | 30 反復 OK | — |
| maxh=0.05 | 185K | 34 反復 OK | — |

反復数はメッシュ非依存で、ε の影響がない正確な解が得られる。

### 多重連結領域では不収束

コイル問題のように領域が多重連結（第一ベッチ数 β₁ ≥ 1）の場合、
`nograds=True` で勾配空間を除去しても **調和形式**（cohomological generators）が
curl-curl 行列のカーネルに残る。調和形式は curl = 0 だが勾配ではない場であり、
多重連結領域にのみ存在する。

このカーネルと BDDC 粗空間のミスマッチにより、メッシュ細分化で収束しなくなる:

| メッシュ (トーラスコイル) | DOFs (p=2) | Shifted-BDDC | Standard BDDC |
|--------------------------|-----------|--------------|---------------|
| maxh=0.04 | 38K | 23 反復 OK | 23 OK |
| maxh=0.03 | 82K | **214** 反復 | 26 OK |
| maxh=0.025 | 111K | **不収束** (500反復) | 31 OK |

ε の値を変えても (1e-6 〜 1.0) 改善しない。CG 行列にカーネルがあることが本質的原因であり、
BDDC 構築行列の正則化の強さは無関係。

### 適用可能性

| 領域のトポロジー | Shifted-BDDC | 理由 |
|------------------|-------------|------|
| 単連結 (box, 球等) | **OK** | `nograds=True` でカーネル完全除去 |
| 多重連結 (コイル問題等) | **NG** | 調和形式がカーネルに残存 |

**推奨**: 多重連結領域の HCurl curl-curl 問題には `eps*u*v*dx` 付きの Standard BDDC を使用する。
ε は十分小さく（例: 1e-6）、解への影響は無視できる。

---

## 参考文献

1. C. R. Dohrmann,
   "A Preconditioner for Substructuring Based on Constrained Energy Minimization",
   *SIAM J. Sci. Comput.*, Vol. 25, No. 1, pp. 246–258, 2003.

2. J. Mandel, C. R. Dohrmann, R. Tezaur,
   "An Algebraic Theory for Primal and Dual Substructuring Methods
   by Constraints",
   *Appl. Numer. Math.*, Vol. 54, No. 2, pp. 167–193, 2005.

3. J. Li, O. B. Widlund,
   "FETI-DP, BDDC, and Block Cholesky Methods",
   *Int. J. Numer. Methods Eng.*, Vol. 66, No. 2, pp. 250–271, 2006.

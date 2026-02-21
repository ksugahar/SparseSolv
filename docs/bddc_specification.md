# SparseSolv BDDC 前処理 — 仕様書

## 概要

SparseSolv BDDC は、NGSolve の `BilinearForm` と `FESpace` から必要な情報を抽出し、
独立した BDDC (Balancing Domain Decomposition by Constraints) 前処理を構築する。

NGSolve 組み込み BDDC と数学的に同等の結果を生成する（反復数・精度が一致）。

## Python API

```python
pre = BDDCPreconditioner(a, fes, coarse_inverse="sparsecholesky")
inv = CGSolver(mat=a.mat, pre=pre, maxiter=500, tol=1e-8)
gfu.vec.data = inv * f.vec
```

| 引数 | 型 | 既定値 | 説明 |
|------|------|--------|------|
| `a` | `BilinearForm` | 必須 | **組立済み**の双線形形式 |
| `fes` | `FESpace` | 必須 | 有限要素空間 |
| `coarse_inverse` | `str` | `"sparsecholesky"` | 粗空間ソルバー（後述） |

| プロパティ | 型 | 説明 |
|-----------|------|------|
| `num_wirebasket_dofs` | `int` | Wirebasket DOF 数 |
| `num_interface_dofs` | `int` | Interface DOF 数 |

---

## NGSolve から抽出する情報

`BDDCPreconditioner(a, fes)` の呼び出し時に、以下の情報を NGSolve から抽出する。

### 1. 組立済み行列

```
bfa->GetMatrixPtr()  →  SparseMatrix<SCAL>
```

- BDDC の apply 時に直接使用するわけではない（CG 側で使用）
- **型判定**に使用: `IsComplex()` で `double` / `Complex` をディスパッチ

### 2. 自由度情報 (FreeDofs)

```
fes->GetFreeDofs(false)  →  BitArray
```

- Dirichlet 境界条件で拘束された DOF を識別
- `false` を指定して **LOCAL_DOF を含める**（order≥3 対応に必須）
- 拘束 DOF は前処理で恒等行として扱われる

### 3. DOF 分類 (CouplingType)

```
fes->GetDofCouplingType(d)  →  WIREBASKET_DOF / その他
```

全 DOF を 2 種類に分類:

| 分類 | NGSolve CouplingType | 意味 |
|------|---------------------|------|
| **Wirebasket** | `WIREBASKET_DOF` | 頂点・辺 DOF（粗空間） |
| **Interface** | それ以外 | 面・内部 DOF（局所消去対象） |

この分類は FESpace の種類と次数に依存:

| FESpace | order=1 | order=2 | order=3+ |
|---------|---------|---------|----------|
| H1 | 全頂点 | 頂点=WB, 辺=IF | 頂点=WB, 辺/面/内部=IF |
| HCurl | 辺 | 辺=WB, 面=IF | 辺=WB, 面/内部(LOCAL)=IF |

### 4. 要素 DOF リスト

```
el.GetDofs()  →  Array<DofId>  (各要素について)
```

- 各体積要素に属する DOF の全体番号リスト
- **正値 DOF のみ**使用（負値 = interface-only、無視）
- `IsRegularDof(d)` でフィルタリング

### 5. 要素行列

```
integrator->CalcElementMatrix(fe, trafo, elmat, lh)
```

- Assembly 後に **再計算** する（これが NGSolve BDDC との速度差の原因）
- **体積 (VOL) 積分子のみ** 使用（境界積分子は除外）
- 複数の積分子がある場合は加算
- 正値 DOF に対応する部分行列を抽出

### 6. メッシュ情報

```
fes->GetMeshAccess()->GetNE(VOL)  →  要素数
```

- 体積要素の数（ループ範囲の決定に使用）

---

## BDDC 内部処理

### セットアップ

抽出した情報から、要素ごとに以下を計算:

```
要素行列 K_e を wirebasket (w) と interface (i) に分割:

K_e = | K_ww  K_wi |
      | K_iw  K_ii |

1. Interface ブロック逆:  K_ii^{-1}
2. Schur 補体:           S_ww = K_ww - K_wi * K_ii^{-1} * K_iw
3. Harmonic extension:   H = -K_ii^{-1} * K_iw
4. Inner solve:          IS = K_ii^{-1}
```

全要素の寄与を重み付きで集約し、以下の疎行列を構築:
- `wb_csr_`: Wirebasket Schur 補体（n_wb × n_wb）
- `he_csr_`: Harmonic extension（全空間）
- `het_csr_`: Harmonic extension 転置
- `is_csr_`: Inner solve（interface → interface）

### 粗空間ソルバー

Wirebasket Schur 補体 `S_wb` の逆を以下のいずれかで計算:

| `coarse_inverse` | 方式 | 用途 |
|-------------------|------|------|
| `"sparsecholesky"` | NGSolve SparseCholesky | 標準（既定） |
| `"pardiso"` | Intel PARDISO | 大規模 wirebasket 向け |
| `"dense"` | SparseSolv 密 LU | テスト用 |

`"sparsecholesky"` と `"pardiso"` は wirebasket CSR 行列を NGSolve の `SparseMatrix` に変換し、
NGSolve の `InverseMatrix` を利用する（コールバック方式）。

### Apply 操作

前処理 `y = M^{-1} x` は以下のステップで計算:

```
1. y = x                              (コピー)
2. y += H^T * x                       (harmonic extension 転置)
3. z_wb = S_wb^{-1} * y_wb            (粗空間ソルブ)
4. z_if = 0                           (interface 部はゼロ)
5. z += IS * x                        (inner solve)
6. y = z + H * z                      (harmonic extension)
```

---

## Shifted-BDDC（前処理と系行列の分離）

BDDC の構築行列と CG の系行列を分離できる:

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

これにより、半正定値の curl-curl 行列を正則化アーティファクトなしに解ける。
NGSolve 組み込み BDDC では ε=0 で BDDC を構築すると収束しない。

---

## 対応する FESpace と行列型

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

## NGSolve BDDC との比較

| 項目 | SparseSolv BDDC | NGSolve BDDC |
|------|----------------|-------------|
| 反復数 | 同等 | 同等 |
| ソルブ時間 | 同等 | 同等 |
| セットアップ時間 | **1.3x 遅い** | Assembly に統合 |
| Shifted-BDDC | **可能** | 不可 |
| 独立モジュール | pybind11 拡張 | NGSolve 内部 |
| API | `BDDCPreconditioner(a, fes)` | `Preconditioner(a, "bddc")` ※Assemble前 |

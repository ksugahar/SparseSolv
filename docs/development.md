# 開発者向け情報

## ビルド手順

### 前提条件

- NGSolve (ソースビルドまたはpipインストール) + CMake設定ファイル
- CMake 3.16以上
- C++17対応コンパイラ (MSVC 2022, GCC 10+, Clang 10+)
- pybind11 (CMakeが自動取得)

### ビルドコマンド

```bash
git clone https://github.com/ksugahar/ngsolve-sparsesolv.git
cd ngsolve-sparsesolv
mkdir build && cd build

cmake .. -DSPARSESOLV_BUILD_NGSOLVE=ON \
         -DNGSOLVE_INSTALL_DIR=/path/to/ngsolve/install/cmake

cmake --build . --config Release
```

**Intel MKL使用時** (PARDISO粗空間ソルバー対応):

```bash
cmake .. -DSPARSESOLV_BUILD_NGSOLVE=ON \
         -DNGSOLVE_INSTALL_DIR=/path/to/ngsolve/install/cmake \
         -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
```

### インストール

```bash
# NGSolve site-packages にコピー
SITE_PACKAGES=$(python -c "import ngsolve, pathlib; print(pathlib.Path(ngsolve.__file__).parent.parent)")
mkdir -p "$SITE_PACKAGES/sparsesolv_ngsolve"
cp build/Release/sparsesolv_ngsolve*.pyd "$SITE_PACKAGES/sparsesolv_ngsolve/"
echo "from .sparsesolv_ngsolve import *" > "$SITE_PACKAGES/sparsesolv_ngsolve/__init__.py"
```

---

## テスト実行

```bash
python -m pytest tests/test_sparsesolv.py tests/test_bddc.py -v --tb=short
```

テスト構成:
- `test_sparsesolv.py`: ソルバー・前処理テスト (46件)
  - ICCG, SGSMRTR, CG, IC, SGS の各種問題
  - 2D/3D, H1/VectorH1/HCurl, 実数/複素数
  - ABMC順序付け, 対角スケーリング, auto-shift
- `test_bddc.py`: BDDCテスト (7件)
  - H1 Poisson, NGSolve比較, メッシュ非依存性
  - HCurl curl-curl, 渦電流 (複素数)
  - 密逆行列精度

---

## 過去のバグと教訓

### 1. DenseMatrix P^T バグ

**問題**: BDDC内部の密LU逆行列で、置換行列を P^T として構築していた。

```cpp
// 間違い (P^T):
inv(piv[j], j) = Scalar(1);

// 正しい (P):
inv(k, piv[k]) = Scalar(1);
```

**症状**: H1問題では影響なし (対角優位でピボット交換が少ない)。
HCurl問題では BDDC が 500反復で発散 (非自明なピボット交換が必要)。

**教訓**: PA = LU の P は「行 k を piv[k] 行目に移す」行列。
P[k, piv[k]] = 1 が正しい構築法。

### 2. 複素内積 (非共役 vs 共役)

**問題**: CG法の内積で `std::conj(a[i]) * b[i]` (共役内積) を使用していた。

FEM行列は**複素対称** (A^T = A) であり、エルミート (A^H = A) ではない。
例: 渦電流方程式 curl-curl + iσ mass。

```cpp
// 間違い (Hermitian用):
sum += std::conj(a[i]) * b[i];

// 正しい (complex-symmetric用):
sum += a[i] * b[i];
```

**症状**: 渦電流問題で 5000反復でも発散。
修正後: 58反復で収束。

**教訓**: FEMの複素数問題では常に非共役内積を使用。
NGSolveの `CGSolver(conjugate=False)` に対応。

### 3. SGS-MRTR 複素比較

**問題**: SGS-MRTRの ζ 計算で `denom >= 0` を使用していたが、
`std::complex<double>` は `>=` 演算子を持たない。

```cpp
// 修正後:
denom = (std::real(denom) >= 0) ? ... : ...;
```

---

## NGSolve API注意点

### AutoVector

NGSolveの `BaseVector::CreateVector()` は `AutoVector` を返す。
これは shared_ptr ライクなセマンティクスを持つが、
独自のムーブ/コピー規約がある。

### CalcElementMatrix

要素行列の計算には `LocalHeap` が必要:

```cpp
LocalHeap lh(10000000, "name", true);  // mult_by_threads=true

IterateElements(*fes, VOL, lh,
    [&](FESpace::Element el, LocalHeap& lh_thread) {
        auto& fe = el.GetFE();
        auto& trafo = el.GetTrafo();
        FlatMatrix<SCAL> elmat(ndof, ndof, lh_thread);
        integrator->CalcElementMatrix(fe, trafo, elmat, lh_thread);
    });
```

### IsRegularDof

要素のDOFリストには無効DOF (< 0) が含まれる場合がある。
`IsRegularDof(dnum)` でフィルタリングする:

```cpp
auto dnums = el.GetDofs();
for (int i = 0; i < dnums.Size(); ++i) {
    if (IsRegularDof(dnums[i])) {
        // 有効DOF
    }
}
```

### FreeDofs (coupling=true)

BDDCでは `fes->GetFreeDofs(true)` (coupling=true) を使用。
これにより interface DOF が「自由」に含まれる。

---

## 参考文献

1. C. R. Dohrmann,
   "A Preconditioner for Substructuring Based on Constrained Energy Minimization",
   *SIAM J. Sci. Comput.*, Vol. 25, No. 1, pp. 246–258, 2003.

2. J. Mandel, C. R. Dohrmann, R. Tezaur,
   "An Algebraic Theory for Primal and Dual Substructuring Methods
   by Constraints",
   *Appl. Numer. Math.*, Vol. 54, No. 2, pp. 167–193, 2005.

3. J. A. Meijerink, H. A. van der Vorst,
   "An Iterative Solution Method for Linear Systems of Which the
   Coefficient Matrix is a Symmetric M-Matrix",
   *Math. Comp.*, Vol. 31, No. 137, pp. 148–162, 1977.

4. M. R. Hestenes, E. Stiefel,
   "Methods of Conjugate Gradients for Solving Linear Systems",
   *J. Research of the National Bureau of Standards*,
   Vol. 49, No. 6, pp. 409–436, 1952.

5. T. Iwashita, H. Nakashima, Y. Takahashi,
   "Algebraic Block Multi-Color Ordering Method for Parallel Multi-Threaded
   Sparse Triangular Solver in ICCG Method",
   *Proc. IEEE IPDPS*, 2012.

6. E. Cuthill, J. McKee,
   "Reducing the Bandwidth of Sparse Symmetric Matrices",
   *Proc. 24th Nat. Conf. ACM*, pp. 157–172, 1969.

7. Y. Tsuburaya, T. Mifune, T. Iwashita, E. Takahashi,
   "MRTR法に基づく前処理付き反復法の数値実験",
   *電気学会研究会資料*, SA-12-64, 2012.

8. Y. Tsuburaya, T. Mifune, T. Iwashita, E. Takahashi,
   "Minimum Residual-Like Methods for Solving Ax = b with Shift-and-Invert
   Enhanced Multi-Step MRTR",
   *IEEE Trans. Magnetics*, Vol. 49, No. 5, pp. 1569–1572, 2013.

9. Y. Tsuburaya,
   "大規模電磁界問題の有限要素解析のための反復法の開発",
   博士論文, 宇都宮大学, 2016.

10. JP-MARs/SparseSolv,
    https://github.com/JP-MARs/SparseSolv

# ngsolve-sparsesolv への貢献

貢献に興味を持っていただきありがとうございます。

## はじめに

[docs/development.md](docs/development.md) を参照してください：
- ビルド手順 (CMake, MKL)
- テスト実行方法
- 既知の問題と過去のバグ

## バグ報告

[GitHub Issue](https://github.com/ksugahar/ngsolve-sparsesolv/issues) を以下の情報を含めて作成してください：
- NGSolve のバージョンと OS
- 最小限の再現例
- 期待される動作と実際の動作

## プルリクエスト

1. リポジトリをフォークしてフィーチャーブランチを作成します
2. 変更を実施します
3. 全テストスイートを実行します：
   ```bash
   python -m pytest tests/test_sparsesolv.py tests/test_bddc.py -v --tb=short
   ```
4. 明確な説明を付けてプルリクエストを提出します

### コードスタイル

- C++17、header-only テンプレート (`<Scalar>` は double/complex 用)
- 既存の命名規則に従う (関数は snake_case、クラスは CamelCase)
- NGSolve 固有のコードは `include/sparsesolv/ngsolve/` に分離します

## ライセンス

貢献することで、あなたの貢献が [MPL 2.0](LICENSE) の下でライセンスされることに同意します。

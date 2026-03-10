# 量子計算とテンソルネットワーク

- 講義資料:
  - [tnqc-1.pdf](tnqc-1.pdf): 次元の呪いと圧縮
  - [tnqc-2.pdf](tnqc-2.pdf): 量子回路をテンソルで書く
  - [tnqc-3.pdf](tnqc-3.pdf): MPSの時間発展
  - [tnqc-4.pdf](tnqc-4.pdf): 大規模シミュレーション戦略
  - tnqc-5.pdf: 応用＋アドバンス話題

- 実習コード
  - 言語
    - [python](python): Python実習コード
    - [julia](julia): Julia実習コード
    - [rust](rust): Rust実習コード
  - Hands-onコード
    - 11_svd
    - 12_image-compression
    - 13_tensor
    - 14_contraction
    - 21_one-qubit
    - 22_two-qubit
    - 23_statevector
    - 24_statevector-tn
    - 25_mps2statevector
    - 26_teleportation
    - 27_gate-teleportation
    - 31_statevector2mps
    - 32_tebd
    - 41_clifford (python only)

- [data](data): 実習で使用するデータファイル

- Python
  - 以下、`python`ディレクトリ内での実行を想定
  - Python の仮想環境準備
    - `python3 -m venv .venv`
    - `source .venv/bin/activate`
  - 必要なパッケージのインストール
    - `pip3 install -r requirements.txt`
  - 実行例
    - `python3 11_svd.py`

- Julia
  - 以下、`julia`ディレクトリ内での実行を想定
  - 必要なパッケージのインストール
    - `julia setup.jl`
  - 実行例
    - `julia --project 11_svd.jl`
  - キャッシュを使った実行
    - `repl`を立ち上げる
      - `julia --project`
    - `repl`内での実行例
      - `include("11_svd.jl")`
    - プログラムのコンパイル結果がキャッシュされるため、2回目以降の実行が高速になる

- Rust
  - 以下、`rust`ディレクトリ内での実行を想定
  - コンパイル
    - `cargo build --release`
  - 実行例
    - `cargo run --release --bin 11_svd`
  - ソースコードは`src/bin`の下にある

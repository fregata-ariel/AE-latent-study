現時点での理解は、次のようにまとめられる。

- **Step 2** は orbit gluing の確立点である
  - AE でも invariance loss を入れれば `X/G` 的な圧縮はかなり強く作れる
- **Step 3 + topology** は、fundamental-domain VAE+invariance が `k=2` までは比較的安定な 2D quotient chart を持つことの主証拠である
- **Step 4** は quotient/gauge を分けて議論する scaffold として成功した
- **Step 5** は、その scaffold 上で現行の chart regularizer 形式が不適切であることを示した

このため現在の研究判断は、次の 2 点に収束している。

- factorization という方向性は正しい
- ただし quotient spread を押す loss は、今の local-distance matching のままではだめ

したがって current recommendation は `A2` であり、「factorized scaffold を保ちながら chart-preserving regularizer の設計を作り直す」ことが次の本線になる。

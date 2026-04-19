現在の分岐は、roadmap と対応させると次のように整理できる。

## A2. Chart-Preserving Regularizer 再設計

本線。Step 5 が示したように、現行 regularizer は scale-invariant すぎて quotient collapse を止められていない。

次に設計し直すべき論点:

- quotient spread を直接保つ loss をどう定義するか
- local geometry と global spread をどう両立させるか
- partner rank / `j` / gauge consistency を壊さない条件をどう組み込むか

## A1. Previous Validated Branch

履歴上の重要到達点として残す。

- Phase B までは `A1` が自然だった
- Step 4 で factorized latent を explicit に実装することには成功した
- ただし Step 5 を踏まえると、そのまま full equivariant latent に進む前に quotient regularization を詰める必要がある

## A3. Sampling Redesign

現時点では parked に近い active branch である。

- wide sampling は coverage probe としては有益
- ただし現時点では本線の successor model ではなく、A2 で fundamental-domain 側を詰めた後に再評価するのが妥当

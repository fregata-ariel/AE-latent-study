# オートエンコーダにおける多様体構造の崩壊要因と幾何学・トポロジー制約を用いた潜在空間学習の包括的分析

## 1. 序論

深層生成モデルおよび非構造化データの表現学習において、オートエンコーダ（AE）および変分オートエンコーダ（VAE）は、高次元の観測データから本質的な低次元の潜在表現（Latent representation）を抽出するための基盤技術として広く活用されている。これらのアーキテクチャの根底には、高次元データが実際には低次元の多様体（Manifold）上に分布しているという「多様体仮説（Manifold Hypothesis）」が存在する。理想的な条件下では、エンコーダは観測空間から潜在空間への写像を、デコーダはその逆写像を学習し、データ多様体の内在的な構造を正確に捉えることが期待される。
しかしながら、実運用の環境や複雑なデータセットにおいて、「局所的な再構成精度は高いものの、大域的には元のパラメータ空間の構造（幾何学・トポロジー）を正確に表現できていない」という重大な欠陥が頻繁に観察される [1]。この症状は、潜在空間上でのデータのクラスタリング、異常検知、あるいは潜在変数の補間（Interpolation）による新規データの生成時において、生成サンプルが多様体から逸脱し、物理的または意味的に破綻した出力を生む根本的な原因となる。近年、この「局所的適合と大域的破綻」というジレンマを解消するため、データの近傍構造（Neighborhood Structure）や大域トポロジー（Global Topology）をモデルの損失関数やアーキテクチャに明示的に組み込む研究領域が急速に発展している [3]。
本報告書では、標準的なAEおよびVAEがなぜデータ多様体の近傍構造や大域トポロジーを破壊するのか、その数学的・情報理論的なメカニズムを網羅的に分析する。次に、k近傍グラフ（kNN）、測地線（Geodesic）、および計算トポロジー（Persistent Homology）の概念をいかにして微分可能な損失関数に昇華させ、この構造的崩壊を防いでいるかについて、最先端の既存研究を解剖する。最後に、画像の局所的な整合性に注目する「Local patch VAE」のアーキテクチャを評価・検証する上で、理論的妥当性を証明するために採用すべきベースラインモデル群の体系的な選定基準を提示する。

## 2. 通常のAE/VAEがデータ多様体の近傍構造・大域トポロジーを壊すメカニズム

標準的なAEおよびVAEが元空間の幾何学的・位相幾何学的構造を潜在空間上で保存できない背景には、次元削減アルゴリズムに内在する数学的制約、目的関数の局所性、確率的推論における事前分布のミスマッチ、および多様体間の写像におけるトポロジー的障害という複数の要因が複雑に絡み合っている。

### 2.1. 次元削減の不可避的代償と等長写像の欠如

オートエンコーダは、入力をそのまま出力にマッピングする自明な恒等写像（Identity mapping）の学習を防止し、データの意味的な特徴を抽出するために、中間層に低次元のボトルネックを設ける [6]。しかし、数学的な証明によれば、n次元のユークリッド空間 \mathbb{E}^n からより低次元のユークリッド空間 \mathbb{E}^m (m < n) への変換において、空間全体の距離を完全に保存する等長写像（Isometry）は存在しない [6]。
この等長写像の欠如により、次元削減を伴うAEは本質的に距離の歪み（Distortion）を引き起こす運命にある [6]。さらに、標準的なバニラAE（Vanilla AE）の損失関数は、入力データ x とデコーダによる再構成データ \hat{x} の間の点単位のユークリッド距離（平均二乗誤差等）を最小化することにのみ焦点を当てている [7]。この目的関数は、個々のデータポイントの絶対的な再構成を促すものの、データポイント間の相対的な距離関係や、データが構成する多様体の局所的な接続性（Connectivity）に対する制約を一切持たない。その結果、エンコーダは訓練データに含まれるノイズに過剰適合（Overfit）しやすく、滑らかであるべき多様体を折り畳んだり交差させたりしてしまい、局所的な接続性と幾何学を誤って学習する [1]。

### 2.2. VAEにおけるKLダイバージェンスと等方性事前分布のミスマッチ

VAEの場合、潜在空間の大域的構造の破壊は、変分下限（ELBO: Evidence Lower Bound）の数学的定式化そのものに起因する側面が強い。VAEの目的関数は、再構成誤差（Distortion）と、エンコーダが推論する潜在分布 q_\phi(z|x) を設定された事前分布 p(z) に近づけるためのKullback-Leibler（KL）ダイバージェンス項（Rate）の和として定義される [8]。
通常、VAEの事前分布には、数学的・計算的な利便性から等方性の球面ガウス分布（Isotropic Spherical Gaussian） \mathcal{N}(0, I) が仮定される。しかし、実世界の複雑なデータ分布に対して、この事前分布は過度に単純化されている（Overly simplistic prior distributions） [9]。データが元空間において複数のクラスタに分かれていたり、ループや穴（Hole）を持つ複雑なトポロジーを形成している場合であっても、KLダイバージェンスによる過剰正則化（Over-regularization）は、すべてのデータを原点付近の単一の球状領域に強制的に押し込めようとする [9]。
この過程で、本来離れているべきクラスタ同士が潜在空間上で不自然に接近・融合したり、連続している多様体が強制的に「引き裂かれる（Tearing）」現象が発生する [9]。例えば、元空間で完全に分離した2つの円状の多様体を学習させた場合、標準的なVAEはこれらを結合させ、閉じていない歪んだ形状を生成してしまうことが確認されている [11]。このようなKLダイバージェンスの性質が、局所的なデータ再構成を維持しつつも、大域的なトポロジーを致命的に破壊する主要な要因となっている。

### 2.3. トポロジー的障害（Topological Obstruction）と特異点の発生

多様体学習の構造崩壊を微分位相幾何学の観点から説明する理論として、「トポロジー的障害（Topological Obstruction）」が存在する [12]。トポロジー的障害とは、異なる位相幾何学的特徴（例えばベッチ数やオイラー標数）を持つ2つの空間の間に、滑らかな同相写像（Homeomorphism：連続かつ逆も連続な全単射）を構築することが不可能であるという数学的定理に基づく [12]。
例えば、データが球面（S^2）やトーラスのような閉じた多様体上に分布しているとする。これを標準的なAEやVAEの平坦なユークリッド潜在空間 \mathbb{R}^m にマッピングしようとすると、エンコーダは物理的に多様体を「切断（Cut）」するか、あるいは無限に引き伸ばして特異点（Singularities）を生成せざるを得ない [13]。このような幾何学的制約（Geometric inductive biases）を持つ空間間で学習を強制すると、勾配降下法は局所的最適解（Local optima）に陥りやすく、トポロジー的次数（Topological Degree）の観点からも、モデルの大域的な表現能力が破綻することが証明されている [13]。

### 2.4. 単一チャート（Single Chart）表現の限界と被覆の不均一性

多様体上のパラメータ化に関するもう一つの深刻な問題は、単一のグローバルな潜在空間（Single-charted latent space）を用いることの限界である [5]。微分幾何学において、多様体は局所的にユークリッド空間と同相である空間と定義され、全体は複数の「チャート（Chart）」が重なり合う「アトラス（Atlas）」として記述される [5]。
地球の表面を1枚の平面地図で正確に表現しようとすると、極地方などで必ず巨大な距離と面積の歪みが生じるのと同様に、複雑なデータ多様体全体を1つのニューラルネットワークで表現することは極めて困難である [5]。平坦な領域は非常に大きなチャートで覆うことができるが、曲率が高い領域や複雑な構造を持つ領域は、歪みを抑えるために複数の小さなチャートを必要とする [5]。このため、標準的なAEの単一チャート構造では、潜在空間上で一様にサンプリングを行ったとしても、それがデータ多様体上での一様サンプリングには決して対応せず、生成結果に偏りや歪みをもたらす結果となる [5]。

| 多様体構造の崩壊要因 | 数学的・理論的背景 | 潜在空間および生成モデルへの影響 |
|---|---|---|
| 等長写像の欠如 | 高次元から低次元への写像において距離を完全保存するIsometryは存在しない。 | 次元削減時に必然的な歪みが発生し、点単位のMSE損失では局所的な近傍関係（グラフ構造）が維持されない。 |
| 事前分布のミスマッチ | VAEのKLダイバージェンスが複雑なデータを等方性ガウス分布 \mathcal{N}(0, I) に強制適合させる。 | 多体体の引き裂き（Tearing）や、独立したクラスタの不自然な融合が発生し、大域的トポロジーが破壊される。 |
| トポロジー的障害 | 異なるトポロジーを持つ空間（例：球面と平面）間に同相写像は構築不可能である。 | 写像関数に特異点（Singularities）が生じ、局所最適解へのトラップや極端なパラメータの歪みが引き起こされる。 |
| 単一チャートの限界 | 複雑な多様体を単一の座標系（Single Chart）で覆うと、高曲率領域で表現が破綻する。 | 潜在空間上の均一な移動が、多様体上での不規則・非連続な移動に変換され、補間（Interpolation）が失敗する。 |

## 3. 既存研究における近傍構造および大域トポロジーを保存する損失関数の定式化

前述のトポロジー崩壊と局所幾何学の歪みを克服するため、最新の研究ではk近傍グラフ、微分幾何学的な測地線、パーシステントホモロジーといった数学的ツールを深層学習の目的関数に組み込むアプローチが多数提案されている。これらの手法は、単なる点単位の再構成を超え、データ間の「関係性」や「大域的な形」をモデルに学習させることを目的としている。

### 3.1. 近傍グラフと局所二次近似による接続性の再構成（NRAE）

バニラAEが訓練データのノイズに過剰適合し局所的な幾何学を誤る問題に対し、Yonghyeon Leeらによって提案された Neighborhood Reconstructing Autoencoder (NRAE) は、k近傍グラフの接続情報をデコーダの局所近似と統合する革新的な手法を採用している [1]。
既存のグラフベースの手法（例えばグラフVAE）が主にエンコーダ側の潜在空間分布を正則化することに注力するのに対し、NRAEはデコーダの挙動そのものを幾何学的に制約する [1]。具体的には、エンコーダ g_\phi とデコーダ f_\theta に対して、NRAEは観測データ x_i とその近傍点集合 \mathcal{N}(x_i) を用いて、以下の「近傍再構成損失（Neighborhood Reconstruction Loss）」を最小化する [1]。
ここで、\tilde{f}_\theta(\cdot; g_\phi(x_i)) は、デコーダ関数 f_\theta の潜在表現 g_\phi(x_i) 周りでの局所的な二次近似（Local Quadratic Approximation）または一次近似（テイラー展開）を意味する [1]。この定式化の最大の利点は、近傍点 x' の再構成が、単なる非線形デコーダの出力に依存するのではなく、基準点 x_i を中心とした滑らかな局所多様体表現に拘束されることにある [1]。
計算コストの観点からもこの手法は優れている。デコーダ全体のヘッシアン（Hessian）やヤコビアン（Jacobian）行列を直接計算・保持するのは非常に高コストであるが、NRAEではヤコビアン・ベクトル積およびベクトル・ヤコビアン積を用いることで、次元数の二乗に比例する計算量（Quadratic scaling）を回避し、スケーラブルかつ効率的な学習を実現している [1]。この正則化により、多様体の折り畳みを防ぎ、データの真の近傍構造と滑らかさを兼ね備えた表現が獲得される [20]。
またグラフ構造を扱う他のアプローチとして、Neighborhood Wasserstein Reconstructionを用いたグラフVAE（NWR-GAE）が挙げられる。この手法では、直接のリンクだけでなく近傍全体の経験的分布とデコードされた分布の間の2-Wasserstein距離を最適化し、トポロジー構造をより広範に捉えようと試みている [21]。

### 3.2. 連続k近傍グラフ（CkNN）を用いた局所距離保存（LPL）

Nutan Chenらによる「Local Distance Preserving Auto-encoders」は、あらゆるスケールのトポロジー的特徴を一つのグラフ構造で捉える Continuous k-Nearest Neighbours (CkNN) グラフを活用している [22]。通常のkNNグラフは近傍の数 k や距離閾値に対して非常に敏感であるが、CkNNはデータ密度の変動を正規化し、位相幾何学的に一貫した重みなしグラフを構築できる利点を持つ [24]。
このアーキテクチャでは、データ空間におけるCkNNグラフから隣接行列 S_{ij} とグラフラプラシアン L を導出し、潜在空間 y \in \mathbb{R}^p において近傍同士が近づくように働く局所距離保存損失（Locality Preserving Loss: LPL）を導入する [25]。
学習を安定させるため、再構成精度を制約（Constraint）とし、局所距離の保存を主目的関数とする制約付き最適化問題（Constraint optimisation problem）として定式化される [22]。これにより、多様体上の滑らかなマッピングが強制され、階層的VAEなどの生成モデルにおいても幾何学的に一貫した潜在空間の構築が可能となる [22]。

### 3.3. 計算トポロジーと構成的ホモロジーを用いた大域トポロジーの同期（TAE）

近傍グラフは局所的な接続性を保証するが、多様体の「穴（Holes）」「ループ（Loops）」「連結成分（Connected components）」といった大域的なトポロジー特徴を保証するには不十分である。この問題を解決するため、Michael Moorらが提案した Topological Autoencoder (TAE) は、トポロジカル・データ解析（TDA）の中核技術である「構成的ホモロジー（Persistent Homology）」を損失関数に組み込んでいる [3]。
TAEは、入力データ空間 X と潜在空間 Z のそれぞれにおいて距離行列を計算し、Vietoris-Rips複体（Vietoris-Rips complex）を構築する [29]。距離の閾値 \epsilon を徐々に増加させるフィルトレーション（Filtration）の過程で、多様体上のトポロジー的特徴がどのスケールで生成（Birth）し、どのスケールで消滅（Death）するかを追跡し、パーシステンス図（Persistence diagram）を生成する [29]。
トポロジー的に重要な役割を果たすデータポイントのペア（例えば0次元ホモロジーにおける最小全域木の構成エッジ）を抽出し、入力空間のペア集合 \mathcal{P}_X と潜在空間のペア集合 \mathcal{P}_Z を特定する [30]。そして、以下の微分可能なトポロジー損失 \mathcal{L}_{topo} を定義する [31]。
この損失関数は、入力空間でトポロジー的に重要なペア間の距離が、潜在空間でも同等のスケールを持つようにペナルティを与える [31]。ホモロジーの計算自体は離散的であるが、ペアを構成する頂点間の距離行列の要素に直接勾配を伝播させることで、微弱な理論的仮定の下でバックプロパゲーションを可能にしている [3]。この手法により、標準AEでは完全に引き裂かれてしまう入れ子状の球面構造などを、潜在空間内で見事に保存することが実証されている [29]。ただし、この損失関数は \mathcal{L}_{topo} = 0 がトポロジー的等価性の必要条件であっても十分条件ではなく、点の微小な摂動に対して最小全域木が変化することで不連続性が生じるという理論的限界も有している [31]。

### 3.4. 統計的多様体と引き戻し計量に基づく測地線（Geodesic）の保存

潜在空間全体を単一のユークリッド空間として扱うVAEの限界に対し、リーマン幾何学を用いて潜在空間に計量（Metric）を与え、直線補間を多様体上の測地線（Geodesic）に合致させるアプローチが存在する [32]。
Geodesic Latent Space Regularization (GLSR-VAE) などの研究では、デコーダ関数のヤコビアンを用いて観測空間の計量を潜在空間に引き戻し、さらに「引き戻し計量（Pull-back metric）」やフィッシャー情報計量（Fisher Information Metric: FIM）を定義する [33]。多様体上の二点 x, y 間の測地線距離は、両者を結ぶ区分的に滑らかな曲線 \gamma の長さの最小値 d_M(x,y) = \min_\gamma L(\gamma) として定義される [32]。
最適輸送距離（Optimal Transport distance）や引き戻し計量を用いた正則化により、潜在空間上の直線移動が、生成空間における連続的で物理的に妥当な属性変化（例えば画像の回転や構造の滑らかな変形）に正確に対応するようになる [34]。近年では FlatVI のように、標準的なVAEボトルネックで見られる曲線的な測地線（Curvature）を抑制し、直線的なパスを達成するアーキテクチャも考案されている [35]。

### 3.5. 複数チャート（Multi-Chart）の枠組みによるトポロジー的障害の回避

前述のトポロジー的障害を完全に回避するためのより根本的な構造変更として、単一のグローバルな潜在空間（Single-charted latent space）を放棄し、多様体のアトラス（Atlas）を複数の局所チャート（Charts）で表現する Chart Auto-Encoder (CAE) がある [5]。
CAEは、データ多様体を複数の重なり合うチャートの集合としてパラメータ化し、各チャートに対応する独立したデコーダ D_i \circ E_i(x) のセットを用意する [5]。学習過程において、入力データはその局所的な幾何学構造に最も適したチャートに割り当てられ、局所的な潜在表現（Local latent representations）が構築される [5]。この手法の最大の利点は、球やダブルトーラスといった自明ではないトポロジーを持つデータに対して、多様体を切断したり歪めたりすることなく、低次元空間で極めて正確な近似と再構成を行えることである [5]。理論的な最悪ケースの境界解析（Worst-case scenario bound）からも、限られた数のチャートで任意の多様体を忠実に表現できることが示されている [39]。

| 手法・モデル名 | 導入される概念 / 計量 | 損失関数・アーキテクチャのメカニズム |
|---|---|---|
| NRAE | k近傍グラフ, 局所近似 | デコーダの局所二次近似（テイラー展開）を用いた近傍再構成損失。 |
| CkNN-AE | CkNN, グラフラプラシアン | 連続k近傍グラフから導出されるラプラシアンを用いた局所距離保存制約。 |
| TAE | 構成的ホモロジー | パーシステンス図から抽出した重要ペア間の距離を合致させる微分可能トポロジー損失。 |
| GLSR-VAE | リーマン計量, 測地線距離 | デコーダのヤコビアンによる引き戻し計量等を用い、潜在空間の直線を測地線に対応させる。 |
| CAE | アトラス, 複数チャート | 複数の局所エンコーダ・デコーダを用意し、トポロジー的障害を回避した局所潜在表現を構成。 |

## 4. Local patch VAEのベースラインとして採用すべきモデルと評価戦略

ユーザーの調査問いである「Local patch VAEのbaselineとして何を採用すべきか」に対して、これまでの理論的考察を踏まえて体系的な選定基準を提示する。
Local patch VAEは、データ全体（例えば画像）を一括して単一のグローバルな潜在変数に落とし込むのではなく、局所的なパッチごとに表現を学習し、それらを統合することで大域的整合性を再構成しようとするアプローチであると解釈される [41]。これは、思想的に「複数チャートによるアトラス構築（CAE）」や「局所近傍構造の統合（NRAE, CkNN）」に酷似している。
したがって、Local patch VAEの有効性（特に「局所的には良いが大域的に破綻する」症状の克服）を証明するためには、純粋なパッチ特化アーキテクチャの先駆的モデル群と、最新の大域トポロジー・局所幾何学保存モデル群の双方から、以下のようにベースラインを選定し、比較実験を設計する必要がある。

### 4.1. パッチ・局所表現学習に特化した直接的アーキテクチャの比較

Local patch VAEのアーキテクチャ上の直接の競合として、既存のパッチベースモデルを採用することは不可欠である。
 * **PatchVAE (Gupta et al., 2020)** : 画像認識や表現学習において、中レベルのスタイル表現（Mid-level style representations）を学習するために、ボトルネックをパッチレベルで定式化したモデルである [41]。純粋なバニラVAEが全体をぼやけた形で再構成するのに対し、PatchVAEは局所的な潜在コード（Local latent codes）を推論することで、画像内の繰り返しパターンや一貫した特徴を捉え、ダウンストリームの認識タスクで大幅な性能向上を示している [41]。このモデルは、局所パッチベース手法のベースラインとして絶対的に含めるべきである。
 * **PatchVAE-GAN / Hierarchical Patch VAE-GAN** : 単一サンプルからの多様な映像生成や、生成結果のリアリズム向上を目指した拡張モデル [41]。敵対的生成ネットワーク（GAN）の識別器を用いることで、再構成パッチの視覚的品質を高めている。Local patch VAEが生成タスクも視野に入れている場合、SVFIDスコア等の比較対象として有用である [43]。
 
### 4.2. 多様体構造の保存能力を実証するための最先端ベースライン

「大域的な元パラメータ空間の表現」が改善されていることを示すためには、前章で分析した幾何学・トポロジー特化型の最新モデル群との定量的な性能比較が必須となる。

| 推奨されるベースラインモデル | 選定の意図および期待される比較効果 | 参考文献 |
|---|---|---|
| Vanilla VAE / \beta-VAE | KLダイバージェンスによる過剰正則化とトポロジー破壊の最悪ケースを示す基準点。これに対する大幅な性能向上が必須。 | [4] |
| Neighborhood Reconstructing AE (NRAE) | 最先端の局所幾何学保存モデル。Local patch VAEが画像パッチを用いた空間的局所性だけで、NRAEのグラフ的アプローチと同等以上の滑らかな近傍接続を達成できるかを比較する。 | [1] |
| Topological Autoencoder (TAE) | 大域トポロジー保存の指標。パッチという局所的な切り出しを行うLocal patch VAEが、TAEのような明示的なトポロジー計算（ホモロジー）なしに大域構造の引き裂きを防げるか検証する強力な対抗馬。 | [3] |
| Chart Auto-Encoder (CAE) | トポロジー的障害を回避する複数チャートモデル。パッチを局所チャートと見なせる場合、CAEの理論的限界（必要チャート数など）に対してどのような優位性を持つかを議論するための比較対象。 | [4] |
| Geometry Regularized AE (GRAE) | 多様体学習アルゴリズム（UMAP等）の表現とAEの表現を明示的に一致させるモデル。NRAEと同等のスケーラビリティと幾何学保存能力を持つため比較に有用。 | [4] |

### 4.3. 評価指標（Metrics）と検証フレームワークの設計

適切なベースラインを選定した後は、定性的な可視化（例：2D潜在空間のプロットによるトポロジー構造の確認）だけでなく、定量的な指標を用いてモデルの構造保存能力を評価しなければならない。
 * **局所幾何学の評価指標** : パッチ表現が元空間の局所構造をどの程度維持しているかを評価するために、多様体学習分野で標準的な Trustworthiness（信頼性） と Continuity（連続性） を採用する [25]。また、Mean Relative Rank Error (MRRE) を用いて、近傍のランク関係が潜在空間でも保存されているかを測定する [25]。
 * **大域トポロジーの評価指標** : 「大域的な元パラメータ空間」が正しく表現されているかを定量化するため、入力データ空間と潜在空間のそれぞれからパーシステンス図（Persistence diagrams）を計算し、両図間の Wasserstein距離 や Bottleneck距離 を測定する。これにより、トポロジカルな特徴の保存度合いを数値化できる。
 * **ダウンストリームタスクの性能評価** : パッチから得られた表現の質を評価するため、ImageNetやCIFAR-100といったデータセットを用いたkNN分類精度やクラスタリングタスクを適用し、PatchVAEやNRAEなどの結果と比較する [41]。
もしLocal patch VAEが、直感的なパッチ分割という手法を用いつつ、NRAEの「近傍の滑らかな接続」やTAEの「大域的な穴・クラスタの維持」と同等、あるいはそれ以上のトポロジー保存能力を示し、かつ計算コスト（明示的なホモロジー計算のオーバーヘッド回避など）で優位に立つことが証明できれば、表現学習分野において極めて説得力の高い貢献となる。したがって、「局所特化型の既存モデル（PatchVAE, NRAE）」と「大域特化型の既存モデル（TAE, CAE）」の両極端をベースラインとして設定し、その間で自手法の位置づけと優位性を明確にする検証フレームワークが最適である。
## 5. 結論
深層生成モデルおよび表現学習における「局所的には良好な再構成を示すが、大域的なパラメータ空間の構造を破壊する」という課題は、次元削減に伴う等長写像の欠如、KLダイバージェンスによる過剰正則化、そして単一チャートによる写像のトポロジー的障害という数学的・幾何学的な必然に根ざしている。
本分析を通じて、最先端の研究領域がこの問題にいかに対処しているかが明らかとなった。Neighborhood Reconstructing Autoencoder (NRAE) や CkNNを用いたアプローチは、局所二次近似やグラフラプラシアンを用いてデータの近傍接続性を精緻に再構成する。一方で Topological Autoencoder (TAE) は、構成的ホモロジーを微分可能な損失に変換することで大域的なトポロジー特徴を同期させ、GLSR-VAEなどのリーマン幾何学ベースのモデルは、引き戻し計量を用いて測地線の滑らかさを担保している。また、Chart Auto-Encoder (CAE) は多様体を複数の局所チャートで被覆し、トポロジー的障害を構造的に回避している。
新たに「Local patch VAE」の有効性を検証するにあたっては、この多角的な視点を評価プロセスに組み込むことが不可欠である。画像パッチ特化のパイオニアであるPatchVAEを基礎的な比較対象としつつ、局所幾何学の最高峰であるNRAEと、大域トポロジー保存の指標であるTAEおよびCAEを厳格なベースラインとして設定すべきである。この包括的な比較フレームワークを通じて、局所パッチの組み合わせがいかにして特異点やトポロジーの引き裂きを回避し、大域的に一貫した多様体アトラスを構築しうるかを示すことができれば、多様体学習と生成モデルの融合領域における極めて重要な理論的・実践的ブレイクスルーとなるだろう。

## 引用文献

[1]: Neighborhood Reconstructing Autoencoders - Advances in Neural Information Processing Systems, 5月 25, 2026にアクセス、 https://proceedings.neurips.cc/paper_files/paper/2021/file/05311655a15b75fab86956663e1819cd-Paper.pdf

[2]: Autoencoding Dynamics: Topological Limitations and Capabilities - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2511.04807v2

[3]: Topological Autoencoders - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 http://proceedings.mlr.press/v119/moor20a/moor20a.pdf

[4]: Neighborhood Reconstructing Autoencoders | OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=_kaH2bAI3O

[5]: Chart Auto-Encoders for Manifold Structured Data | OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=rJeBJJBYDB

[6]: Do Autoencoders preserve distances? - Cross Validated - Stats StackExchange, 5月 25, 2026にアクセス、 https://stats.stackexchange.com/questions/351212/do-autoencoders-preserve-distances

[7]: Autoencoder with Distribution Preservation - OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=L4cQ2Btscv

[8]: Learning Useful Representations with Variational Autoencoders - UvA-DARE (Digital Academic Repository), 5月 25, 2026にアクセス、 https://pure.uva.nl/ws/files/163641968/Thesis.pdf

[9]: Reining in the Deep Generative Models - Universität Tübingen, 5月 25, 2026にアクセス、 https://publikationen.uni-tuebingen.de/xmlui/bitstream/handle/10900/141548/Reining%20in%20the%20Deep%20Generative%20Models.pdf?sequence=1&isAllowed=y

[10]: What's anomalous in LHC jets? Abstract Contents - SciPost, 5月 25, 2026にアクセス、 https://scipost.org/SciPostPhys.15.4.168/pdf

[11]: Manifold Learning by Mixture Models of VAEs for Inverse Problems, 5月 25, 2026にアクセス、 https://jmlr.org/papers/volume25/23-0396/23-0396.pdf

[12]: Deep Learning Methods for 2D Material Electronic Structures - ChemRxiv, 5月 25, 2026にアクセス、 https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6809000de561f77ed472d7d4/original/deep-learning-methods-for-2d-material-electronic-structures.pdf

[13]: Topological Obstructions and How to Avoid Them - OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=1tviRBNxI9&noteId=C949McUOje

[14]: [PDF] Topological Obstructions and How to Avoid Them | Semantic, 5月 25, 2026にアクセス、 https://www.semanticscholar.org/paper/Topological-Obstructions-and-How-to-Avoid-Them-Esmaeili-Walters/ba3a342180aaec6e0f624b93b5441e00f2bab3a7

[15]: Topological degree as a discrete diagnostic for disentanglement, with applications to the \DeltaVAE - CEUR-WS.org, 5月 25, 2026にアクセス、 https://ceur-ws.org/Vol-3928/paper_145.pdf

[16]: (PDF) Topological degree as a discrete diagnostic for disentanglement, with applications to the \DeltaVAE - ResearchGate, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/383701990_Topological_degree_as_a_discrete_diagnostic_for_disentanglement_with_applications_to_the_DeltaVAE

[17]: Neighborhood Reconstructing Autoencoders, 5月 25, 2026にアクセス、 https://neurips.cc/media/neurips-2021/Slides/27723.pdf

[18]: Bi-Lipschitz Autoencoder With Injectivity Guarantee - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2604.06701v1

[19]: On Explicit Curvature Regularization in Deep Generative Models - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 https://proceedings.mlr.press/v221/lee23a/lee23a.pdf

[20]: Recovering manifold representations via unsupervised meta-learning - Frontiers, 5月 25, 2026にアクセス、 https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1255517/full

[21]: Graph Auto-Encoder via Neighborhood Wasserstein Reconstruction | OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=ATUh28lnSuW

[22]: Local Distance Preserving Auto-encoders using Continuous k ..., 5月 25, 2026にアクセス、 https://openreview.net/forum?id=MpwWSMOlkc

[23]: [2206.05909] Local Distance Preserving Auto-encoders using Continuous k-Nearest Neighbours Graphs - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/abs/2206.05909

[24]: LOCAL DISTANCE PRESERVING AUTO-ENCODERS USING CONTINUOUS KNN GRAPHS - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 https://proceedings.mlr.press/v196/chen22b/chen22b.pdf

[25]: Locality Preserving Loss in Representation Learning - Emergent Mind, 5月 25, 2026にアクセス、 https://www.emergentmind.com/topics/locality-preserving-loss-lpl

[26]: Local distance preserving auto-encoders using Continuous k-Nearest Neighbours graphs, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/361274002_Local_distance_preserving_auto-encoders_using_Continuous_k-Nearest_Neighbours_graphs

[27]: Local Distance Preserving Auto-encoders using Continuous kNN Graphs - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 https://proceedings.mlr.press/v196/chen22b.html

[28]: Topological Autoencoders - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 https://proceedings.mlr.press/v119/moor20a.html

[29]: Topological Autoencoders. - Michael Moor, 5月 25, 2026にアクセス、 https://michaelmoor.me/blog/topoae/main/

[30]: Topological Autoencoders - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/pdf/1906.00722

[31]: Manifold-Matching Autoencoders - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2603.16568v1

[32]: Distance preserving Fermat VAE - Diva-Portal.org, 5月 25, 2026にアクセス、 https://www.diva-portal.org/smash/get/diva2:1711712/FULLTEXT01.pdf

[33]: Enforcing Latent Euclidean Geometry in VAEs for Statistical Manifold Interpolation, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=a72vorQK8v

[34]: GLSR-VAE: Geodesic Latent Space Regularization for Variational AutoEncoder Architectures - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/pdf/1707.04588

[35]: Enforcing Latent Euclidean Geometry in Single-Cell VAEs for Manifold Interpolation - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2507.11789v1

[36]: MARBLE: interpretable representations of neural population dynamics using geometric deep learning - PMC, 5月 25, 2026にアクセス、 https://pmc.ncbi.nlm.nih.gov/articles/PMC11903309/

[37]: Interpretable statistical representations of neural population dynamics and geometry - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2304.03376v4

[38]: CHART AUTO-ENCODERS FOR MANIFOLD STRUCTURED DATA | SciSpace, 5月 25, 2026にアクセス、 https://scispace.com/pdf/chart-auto-encoders-for-manifold-structured-data-29db6o9ncf.pdf

[39]: Semi-Supervised Manifold Learning with Complexity Decoupled Chart Autoencoders - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2208.10570v2

[40]: Semi-Supervised Manifold Learning with Complexity Decoupled Chart Autoencoders, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/362886993_Semi-Supervised_Manifold_Learning_with_Complexity_Decoupled_Chart_Autoencoders

[41]: PatchVAE: Learning Local Latent Codes for Recognition - CVF Open Access, 5月 25, 2026にアクセス、 https://openaccess.thecvf.com/content_CVPR_2020/papers/Gupta_PatchVAE_Learning_Local_Latent_Codes_for_Recognition_CVPR_2020_paper.pdf

[42]: [2004.03623] PatchVAE: Learning Local Latent Codes for Recognition - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/abs/2004.03623

[43]: (PDF) Hierarchical Patch VAE-GAN: Generating Diverse Videos from a Single Sample, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/342377915_Hierarchical_Patch_VAE-GAN_Generating_Diverse_Videos_from_a_Single_Sample

[44]: Hierarchical Patch VAE-GAN: Generating Diverse Videos from a Single Sample, 5月 25, 2026にアクセス、 https://proceedings.neurips.cc/paper_files/paper/2020/file/c2f32522a84d5e6357e6abac087f1b0b-Paper.pdf

# 高次元データ表現における位相幾何学的構造の保存と評価：Topology-aware Metricsの包括的分析

## 序論：表現学習と多様体仮説における位相幾何学的課題

現代の深層学習および表現学習（Representation Learning）の根底には、高次元空間に存在する現実世界のデータ（自然画像、音声、テキストなど）が、実際にははるかに低次元の多様体（Manifold）上に分布しているという「多様体仮説（Manifold Hypothesis）」が存在する [1]。この仮説に基づき、オートエンコーダ（Autoencoder）や主成分分析（PCA）、t-SNE、UMAPといった次元削減手法は、データの複雑な構造を低次元の潜在空間（Latent Space）へとマッピングすることを目的として発展してきた。しかし、伝統的な手法の多くはデータ点間の局所的な近傍関係（Local Connectivity）の維持に最適化されており、データセット全体を規定する大域的な構造（Global Structure）や、複数のスケールにまたがる位相幾何学的（Topological）な連結性の保存には根本的な限界を抱えている [2]。
この問題が顕著に現れる典型例として、高次元空間において1つの大きな球面が複数の小さな球面を内包しているような「Synthetic Spheres（合成球面）」データセットの次元削減が挙げられる [4]。標準的なオートエンコーダやUMAPを用いてこのデータを2次元の潜在空間にマッピングすると、各球面の局所的な連続性は保たれるものの、球面同士が単に空間上で引き離され、元のデータが持っていた「入れ子構造（包含関係）」という大域的な位相構造が完全に破壊されることが観測されている [4]。このような位相の崩壊は、下流タスク（分類、生成、クラスタリング）における重大な性能低下や、解釈可能性の喪失を引き起こす。
こうした背景から、位相的データ解析（Topological Data Analysis; TDA）の中心的なツールである「パーシステントホモロジー（Persistent Homology; PH）」を機械学習のパイプラインに統合し、位相構造を明示的に保存・評価しようとするTopology-awareなアプローチが急速に台頭している [7]。本報告書では、潜在空間が元空間の位相を保存しているかをいかにして測定するかという評価指標の設計、パーシステントホモロジーを訓練損失（Training Loss）と評価指標（Evaluation Metric）の両方に用いる際の理論的妥当性とリスク、とベッチ数（Betti Number）、パーシステンス図（Persistence Diagram）、Wasserstein距離といった位相的計量の具体的な数理と応用方法について、網羅的かつ深層的な分析を提供する。

## 潜在空間における位相保存の測定手法

潜在表現が元の高次元空間の位相を保存しているかどうかを定量的に測定することは、次元削減手法の品質を担保する上で不可欠である。この測定アプローチは、伝統的な局所的・幾何学的な評価指標と、近年発展している大域的・位相的な評価指標（Topology-aware metrics）の二つに大別される。

### 局所的構造の保存を測る伝統的指標とその限界

データ空間から潜在空間への写像において、各データ点の近傍構造がどの程度維持されているかを評価するために、以下のような指標が広く用いられてきた。

| 評価指標 | 測定する性質とメカニズム | 位相構造評価における限界 |
|---|---|---|
| **Trustworthiness (信頼性)** | 潜在空間において新たに近傍に現れた「偽の近傍点（False Neighbors）」の割合にペナルティを与える [8]。 | 局所的な密集度を測るのみであり、データ全体の穴や空洞といった大域的なトポロジーの破壊を検知できない [13]。 |
| **Continuity (連続性)** | 元空間で近傍であったにもかかわらず、潜在空間で遠くに配置されてしまった「欠落した近傍点（Missing Neighbors）」にペナルティを与える [8]。 | 同様に局所スケールでの連続性しか保証せず、離れたクラスタ間の相対的な位置関係を無視する [13]。 |
| **MRRE (Mean Relative Rank Error)** | 距離の順位（ランク）が元空間から潜在空間へと変換される際に生じる相対的な順位誤差の平均を測定する [3]。 | ペアワイズ距離の単調な変換には強いが、多様体の折り畳みや位相的欠陥（Topological Defects）の発生には鈍感である [13]。 |
| **dRMSE (Distance Matrix RMSE)** | 元空間と潜在空間におけるペアワイズ距離行列間の二乗平均平方根誤差を計算し、距離スケールの保存性を評価する [3]。 | ユークリッド距離の絶対値に依存するため、多様体の本質的な形状（測地線距離や位相不変量）を捉えきれない [3]。 |

これらの伝統的な評価尺度は、t-SNEやUMAPのような局所的近傍に基づく手法を高く評価する傾向がある一方で、Synthetic Spheresのようにデータセットの半分を占める包含関係が失われている状態であっても、高いスコアを算出してしまうという致命的な欠点を持っている [4]。したがって、潜在空間が元空間の位相を真に保存しているかを測るためには、より大域的な不変量を直接的に捉える新たな尺度が求められる。

### 位相幾何学的な構造保存の測定（Topology-aware metrics）

潜在表現の位相的保存性を直接的に測定するためには、入力空間 X と潜在空間 Z の間で、スケールや次元を超えて位相的特徴がどの程度一致しているかを比較するアプローチが必要となる [10]。これを実現するために、データ点群からパーシステントホモロジーを計算し、抽出されたトポロジカル・シグネチャを比較する手法が開発されている。
具体的な測定アプローチとして、データのマルチスケールな要約であるパーシステンス図（Persistence Diagram）やバーコード（Barcode）を入力空間と潜在空間のそれぞれで構築し、その間のBottleneck距離やWasserstein距離を計算することで、トポロジーの忠実度（Topological Fidelity）を測定する手法が主流となっている [10]。さらに、二つの異なる表現空間の位相的非類似性を直接的に測るための高度な指標として、Representation Topology Divergence (RTD) や、元データと生成データの分布の位相を比較するGeometry Score (Geometric Alignment Score) が提案されている [10]。これらの指標の具体的なアルゴリズムについては後段のセクションで詳述するが、共通する哲学は「単一の距離スケールに基づく比較から、すべてのスケールを通じた位相不変量のライフサイクル（誕生と消滅）の比較への移行」である [8]。

## パーシステントホモロジーの二面性：訓練損失と評価指標

「パーシステントホモロジーを訓練損失（Training Loss）と評価指標（Evaluation Metric）の両方に使用できるか？」という問いは、位相的機械学習における最も深遠な論点の一つである。結論から言えば、技術的には両方に使用可能であるが、実践的な評価論理においては「循環論法（Circular Reasoning）」という重大なリスクを伴うため、両者の役割を厳密に分離し、独立した手法で評価を行うことが強く推奨される。

### 微分可能な位相的損失関数（Topological Loss）の構築メカニズム

従来、パーシステントホモロジーを深層学習の最適化制約として直接用いることは極めて困難であった。これは、ホモロジーの計算が本質的に離散的（データ点からの単体的複体の構築や、境界行列のランク計算など）であり、勾配降下法（Gradient Descent）に必要な誤差逆伝播法（Backpropagation）を適用できないためである [4]。しかし、Topological Autoencoder (TopoAE) を提唱したMoorらの研究により、弱い理論的仮定の下でこの位相的損失を微分可能な形で構築する画期的な手法が確立された [7]。
微分可能なパーシステントホモロジーのパイプラインは以下のメカニズムで機能する。まず、ミニバッチ内の入力データ空間および潜在空間のそれぞれにおいて、ペアワイズ距離行列を計算する [5]。次に、距離の閾値 \epsilon を徐々に増大させながら、距離が \epsilon 以下であるデータ点間にエッジを張り、三角形や四面体といった単体（Simplex）を形成していく「Vietoris-Rips (VR) フィルトレーション」を実行する [5]。この過程で、特定の次元における位相的特徴（連結成分、穴、空洞など）が生成（Birth）され、別の閾値で消滅（Death）する。この誕生と消滅のスケールを記録したものがパーシステンス図である [4]。
ここで微分可能性を担保する鍵となるのが、「パーシステンス図上の任意の座標 (b, d) は、VR複体においてその位相的特徴を誕生または消滅させた特定のクリティカルな単体（すなわち特定の2つのデータ点間のエッジ）の長さに厳密に対応する」という事実である [22]。このエッジの長さは、ニューラルネットワークが出力した2点間の距離そのものであるため、連鎖律（Chain Rule）を適用することで、パーシステンス図の座標から特定のデータ点の出力、さらにはネットワークの重みへと勾配を逆伝播させることができる [8]。現在では torchph や TopologyLayer といったPyTorch拡張ライブラリが提供されており、ユーザーは複雑な実装を意識することなく、数行のコードで微分可能なパーシステントホモロジーを損失関数として組み込むことが可能となっている [23]。

### 訓練と評価の同一化がもたらす循環論法の危険性

TopoAEのようなモデルは、入力空間と潜在空間のパーシステンス図間の差異（位相的損失）を最小化するように訓練される [6]。この最適化プロセスにおいて、もし評価指標として「入力空間と潜在空間の距離行列間のRMSE」や「訓練に用いたのと全く同じパーシステンス図間のマッチング距離」を報告した場合、それは真の位相保存性を証明したことにはならない。
研究コミュニティにおいて指摘されている通り、最適化の対象となった損失関数と同じ尺度をパフォーマンス指標として報告することは「循環論法（Circular Reasoning）」を構成する [3]。これは経済学におけるGoodhartの法則（測定が目標になると、それは良い測定ではなくなる）と同様の現象であり、ネットワークがデータ多様体の真の大域等位相を学習したのではなく、単に特定のミニバッチ上でのトポロジカル・シグネチャの差異を数値的に最小化する「近道」を見つけたに過ぎない可能性を排除できないためである [3]。
したがって、位相的損失を適用したモデルの汎化性能と構造保存性を正当に評価するためには、訓練プロセスから独立した評価指標を併用することが必須となる。具体的には、訓練にはVRフィルトレーションに基づくTopological Lossを用い、評価には後述するRepresentation Topology Divergence (RTD) のような独立した位相的ダイバージェンスや、クラスタリング品質（ARI、Silhouette）、さらには下流の合成画像生成タスクやゼロショット学習における性能（Zero-shot stitchingのMSEなど）といった多角的なアプローチを組み合わせることが、堅牢な研究手法として確立されている [3]。また、計算コストの観点から、評価時のみオイラー標数曲線（Euler Characteristic Curve; ECC）やDirectional Sign Loss (DSL) などの代替的な位相指標を用いて計算効率を高めつつ構造を評価する手法も提案されている [10]。

## 表現位相の計量：ベッチ数、パーシステンス図、Wasserstein距離の数理と適用

潜在空間の位相を測定・最適化する上で、位相的特徴をどのように定量化し比較するかが手法の成否を分ける。ここでは、ベッチ数（Betti Number）、パーシステンス図距離、およびWasserstein距離という三つの核心的な概念が、どのように理論化され、実際の機械学習パイプラインで使用されているかを詳解する。

### ベッチ数 (Betti Number) と相対生存時間 (Relative Living Times; RLT)

ベッチ数 \beta_k は、空間における k 次元の独立した位相的特徴（穴）の数を表す代数トポロジーの基本的な不変量である。具体的に、 \beta_0 は連結成分の数、 \beta_1 は1次元のループ（円）の数、 \beta_2 は2次元の空洞の数を示す [6]。
しかし、機械学習においてベッチ数を直接的な評価指標や損失関数として使用することには大きな障壁がある。VRフィルトレーションの過程において、特定の距離閾値 \alpha に依存するベッチ数 \beta_k(\alpha) は離散的な整数値を取るステップ関数となるため、微分が不可能であり、またデータにわずかなノイズが混入しただけで無数の小さな「穴」が生成され、ベッチ数が激しく変動するという不安定性（Instability）を抱えている [32]。
この限界を克服し、ベッチ数の概念を生成モデル（GANなど）の位相的評価に応用するために、KhrulkovとOseledetsによって「Geometry Score（Geometric Alignment Score）」という画期的な手法が提案された [1]。この手法の核心は、「相対生存時間（Relative Living Times; RLT）」という連続的な確率分布の概念を導入したことにある [17]。
RLTは、フィルトレーションの全パラメータ範囲 [0, \alpha_{\max}] において、特定のベッチ数 i （例えば、1次元の穴がちょうど1つ存在する状態）が観測された区間の長さの合計を求め、それを最大スケール \alpha_{\max} で正規化した値として定義される [33]。
ここで、 \mu はルベーグ測度 [48]、 W は計算効率を高めるために一部のランドマークポイント L のみを用いて構築されるWitness複体（Witness Complex）である [33]。RLTは、特定のトポロジーが存在した「時間の割合」を測定するため、ノイズによって生じた短命な穴の影響を自然に減衰させ、その位相的特徴の「信頼度（Confidence）」を連続的な値として表現することができる [17]。Geometry Scoreは、真のデータ分布と生成されたデータ分布のそれぞれから期待値としてのMean RLT (MRLT) を計算し、その間のWasserstein距離に基づく不類似性行列を評価することで、モード崩壊や生成品質を極めて高精度に検出する [32]。

### パーシステンス図距離におけるBottleneck距離とWasserstein距離の差異

パーシステンス図（Persistence Diagram）は、データから抽出されたすべての位相的特徴の誕生 b_i と消滅 d_i のタイミングを、2次元平面上の座標群として要約したマルチセットである [22]。対角線 \Delta = \{(x, x) \mid x \in \mathbb{R}\} に近い点はノイズとみなされ、遠い点は真の構造的シグナルと解釈される [6]。二つの異なるパーシステンス図 Dgm_1 と Dgm_2 （例えば入力空間と潜在空間のパーシステンス図）の間の距離を測定することは、位相的保存性を評価する上で最も直接的なアプローチである [57]。
この距離を計算するためには、二つの図の点群間で最適輸送計画（Optimal Transport Plan）を立案し、バイジェクション（全単射） \gamma を見つける必要がある [58]。図の要素数が異なる場合でも、余った点を対角線 \Delta （各点にとってコストが最小となる自身の射影）にマッチングさせることで全単射を構成する [37]。このマッチングコストの評価方法として、Bottleneck距離（ d_\infty ）とWasserstein距離（ d_{W,p} ）が存在し、それぞれ数学的特性と機械学習における役割が明確に異なる。

| 比較属性 | Bottleneck Distance (d_\infty) | Wasserstein Distance (d_{W,p}) |
|---|---|---|
| **数学的定義** | \inf_{\gamma} \sup_{x \in Dgm_1} \|x - \gamma(x)\|_\inf [41] | \left( \inf_{\gamma} \sum_{x \in Dgm_1} \|x - \gamma(x)\|_\infty^p \right)^{1/ [37] |
| **評価対象** | マッチングされたペアの中で **最大の誤差（最悪の移動距離）のみ** をコストとする [40]。 | マッチングされた すべてのペアの誤差の **p 乗和** をコストとする（通常 p=2 ） [37]。 |
| **計算アルゴリズム** | Hopcroft-Karpアルゴリズム（最小コストの完全マッチング） [43] | Auctionアルゴリズム（競売アルゴリズムによる最適輸送） [43] |
| **TDAにおける役割** | 入力データの微小な変動に対するパーシステンス図の「安定性（Stability）」を数学的に証明するための理論的基盤 [38]。 | 損失関数や評価指標として、全体のトポロジカルな歪みを定量化する実用的な計量 [38]。 |
| **最適化（勾配）の性質** | 誤差が最大の単一の点にしか勾配が流れないため、極めて**疎（Sparse）な勾配**となる。学習が進まない原因となる [38]。 | すべてのマッチング点に対して**密（Dense）な勾配**を提供するため、ニューラルネットワークの最適化に極めて適している [22]。 |

このように、Bottleneck距離は理論的な堅牢性を保証するための分析ツールとして優れているが、深層学習におけるLoss関数として使用した場合、ネットワーク内の大半の重みに対して勾配がゼロとなるため、最適化が失敗する。一方、Wasserstein距離はすべての位相的特徴のずれを合算するため、ネットワーク全体の位相的な歪みを効果的に補正する「密な勾配（Dense Gradients）」を供給できる [38]。このため、Topological Autoencoderの損失関数や、評価指標としてのパーシステンス図間の比較には、Wasserstein距離（特に2次Wasserstein距離）を使用することが実践的な標準となっている [10]。さらに、計算コストを低減するために、空間分割（Octree）を用いたヒストグラムの粗視化やスライスド・ワッサースタイン距離（Sliced Wasserstein Distance）といった近似アルゴリズムも積極的に導入されている [42]。

## 表現位相ダイバージェンス（RTD）と最新の評価フレームワーク

Topological Autoencoderの文脈において、近年最も注目を集めている包括的な評価指標が「Representation Topology Divergence (RTD)」である [10]。前述のGeometry Scoreが、GANなどの「二つのデータ分布」が生成するトポロジーを比較するためのドメイン非依存な指標であるのに対し、RTDは、エンコーダ等によって生成された「一対一の対応関係（Bijection）を持つ二つの異なる表現空間（例：元の高次元空間と潜在空間）」の位相的非類似性を直接比較することに特化している点が決定的に異なる [18]。

### RTDのアルゴリズム：Cross-Barcodeとスケール正規化

RTDアルゴリズムの中心的な革新は、異なる次元を持つ二つの空間（例：100次元の入力空間と2次元の潜在空間）を同一の土俵で比較可能にする「Cross-Barcode（クロスバーコード）」の概念である [48]。
アルゴリズムは以下の手順で進行する：
 1. **スケールの正規化**：入力空間 X と潜在空間 Z のそれぞれでデータ点間のペアワイズ距離行列を計算する。この際、二つの空間の距離スケールを揃えるため、それぞれの空間のペアワイズ距離の「0.9分位点（0.9 Quantile）」を用いて距離行列を正規化する [21]。これにより、RTDは等方的なスケーリング（Isotropic Scaling）に対して不変性（Scale Invariance）を持ち、異なるアーキテクチャやドメイン間でも比較可能な妥当なスコアを提供する [21]。
 2. **Vietoris-Rips複体の結合**：正規化された二つの距離行列を統合し、確率的なアルゴリズムを用いて距離閾値 \epsilon に対するグラフを構築し、Vietoris-Rips複体に基づく1次元のバーコードを計算する [21]。
 3. **位相的ダイバージェンスの算出**：クロスバーコードとして抽出された、二つの空間間でミスマッチを起こしている位相的特徴（保存されなかったループやクラスタ）の区間の長さの合計を測定する。この合計値がRTDスコアとして出力される [10]。スコアが低いほど、二つの表現が細かい位相構造を共有していることを意味する。
### RTDの派生と応用
RTDはその汎用性の高さから、単なる評価指標にとどまらず、損失関数やモデル分析ツールとして応用範囲を拡大している。
 * **RTD-AE**：Autoencoderの損失関数そのものにRTDの微分を組み込んだ「RTD-AE」は、大域的なデータ多様体のトポロジーを強力に保存することが示されており、線形相関やトリプレット距離のランキング精度、さらにはパーシステンスバーコード間のWasserstein距離において最先端の性能を達成している [3]。
 * **Symmetric RTD**：オリジナルのRTDが持つ非対称性を理論的に補完し、正規化されたスケールを持つ「Symmetric Representation Topology Divergence (SRTD)」およびその軽量版である「SRTD-lite」が提案されており、異なるニューラルネットワークモデル間の直接的な解釈可能性と横断的な比較機能がさらに強化されている [53]。
## 先端応用領域におけるTopology-aware手法の展開
位相幾何学的な損失関数や評価指標は、オートエンコーダの次元削減にとどまらず、様々な最先端の機械学習ドメインにおいて不可欠な技術となりつつある。これらの応用事例は、パーシステントホモロジーが局所的なピクセル単位・ベクトル単位の最適化では到達できない、大域的な事前知識（Prior）をモデルに付与できることを証明している。
### 画像セグメンテーションにおける位相的正則化
医療画像解析、とりわけニューロンの膜や心臓MRI（ACDC short axis CMR）のセグメンテーションにおいて、ピクセル単位の精度（Cross-Entropy等）は高いものの、わずかな予測の欠落によって細胞膜が途切れたり、心筋のドーナツ状（Toroidal）のトポロジーが破壊されたりする問題が存在した [22]。これに対し、ニューラルネットワークが出力する連続的な尤度関数（Likelihood Function）に対してすべての閾値でパーシステントホモロジーを計算し、予測されたパーシステンス図とグラウンドトゥルースのパーシステンス図間のWasserstein距離をTopological Lossとして最小化する手法が確立された [22]。このアプローチは、オーバーラップ性能（Dice係数等）を犠牲にすることなく、すべての位相的エラー（穴の塞がりや線の途切れ）を解決し、解剖学的に正確な単一の連結成分としての構造を保証することに成功している [22]。
### 大規模視覚言語モデル（VLMs）とFew-shot学習
自然言語処理（NLP）および視覚言語モデル（CLIPやBLIPなど）の表現アラインメントにおいても、トポロジーの概念が導入されている [56]。大規模モデルのFew-shot学習において、事前に学習された知識を保持しつつタスク固有の適応を行う際、画像表現とテキスト表現の潜在的な多様体構造を整列させるためにRTDが活用されている [57]。軽量なTask Residual（TR）パラメータのみを最適化し、画像とテキストの潜在表現間のRTDをクロスエントロピー損失と組み合わせて最小化することで（Topology-aware tuning）、複数のベンチマークにおいてベースラインを上回るFew-shot性能の向上が実証されている [57]。
### 顔認識モデルと物理情報ニューラルネットワーク
顔認識（Face Recognition）の分野では、Topological Autoencoderの概念を応用した「TopoFR」モデルが提案されている [58]。これは、PTSA（Persistent Topology Structure Alignment）という戦略を用いて入力空間と潜在空間のトポロジーを整列させ、さらにSDE（Structure Damage Score）によって位相構造に悪影響を与えるハードサンプルを自動的に識別・軽減することで、IJB-Cベンチマーク等で最先端の汎化性能を達成している [58]。
また、物理科学の領域においても、磁区形成（Magnetic domain formation）の時系列画像データからパーシステンス図を抽出し、それをオートエンコーダで潜在空間に埋め込んだ上でハミルトニアン・ニューラルネットワークに類似した「Neural Reduced Potential」を学習させる枠組みが提案されており、物理的なダイナミクスをトポロジカルな観点から滑らかにモデリングすることに成功している [35]。

## 結論

高次元データの複雑な構造を低次元の潜在空間で維持・評価するためのTopology-aware metricsとTopological Autoencoderの研究は、表現学習におけるパラダイムシフトを引き起こしている。本報告書の包括的な分析により、以下の重要な結論が導出される。
 1. 潜在空間の位相保存性の測定においては、TrustworthinessやMRREといった局所的な近傍関係に依存する指標だけでは不十分であり、データの大域的な多様体構造を捉えるためには、Representation Topology Divergence (RTD) やGeometry Score、パーシステンス図間のWasserstein距離といったTopology-awareな評価指標が不可欠である。これらの指標は、スケールや次元を超えて位相不変量のライフサイクルを追跡することで、表現の質をより深く定量化する。
 2. パーシステントホモロジーは、Vietoris-Ripsフィルトレーションとクリティカル単体への逆伝播マッピングを用いることで、強力な微分可能な損失関数（Training Loss）として機能する。しかし、この最適化対象そのものを単独の評価指標（Evaluation Metric）として使用することは、循環論法（Goodhartの法則）を招く危険性がある。真の位相保存性を実証するためには、訓練に使用した損失関数とは独立したトポロジー評価指標（RTDなど）や、下流タスクでのパフォーマンステストを用いた交差評価が必須である。
 3. 位相的特徴の計量において、Bottleneck距離は理論的安定性の証明には有用であるが、その最悪値（ L_\infty ノルム）のみを評価する性質上、勾配が極めて疎となり最適化には適さない。一方、Wasserstein距離はすべての位相的ミスマッチを合算する（ L_p ノルム）ため、ニューラルネットワークの訓練に必要な密な勾配（Dense Gradients）を提供でき、損失関数として圧倒的な優位性を持つ。また、離散的なベッチ数を評価指標として用いる場合は、Relative Living Times (RLT) を介して連続的な確率分布へと変換するアプローチが極めて有効である。
パーシステントホモロジーと深層学習の融合は、単なる次元削減にとどまらず、画像セグメンテーションにおける解剖学的正確性の保証や、大規模視覚言語モデルの表現アラインメントに至るまで、広範な領域でブレイクスルーをもたらしている。計算コストの削減（ECCの利用や空間分割法）といった課題は残されているものの、トポロジーを事前知識としてモデルに組み込むこれらの一連のアプローチは、次世代の堅牢かつ解釈可能なAIシステムを構築するための最も有望な基盤技術の一つであると結論付けられる。

## 引用文献

[1]: Geometry Score: A Method For Comparing Generative Adversarial Networks - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 http://proceedings.mlr.press/v80/khrulkov18a/khrulkov18a.pdf

[2]: 序論：表現学習と多様体仮説における位相幾何学的課題

[3]: Manifold-Matching Autoencoders - OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=Iq6o8A8NVA

[4]: Topological Autoencoders - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/pdf/1906.00722

[5]: Topological Autoencoders - Semantic Scholar, 5月 25, 2026にアクセス、 https://pdfs.semanticscholar.org/a074/d406a6a67a494d48c9dd9e918623c644e0f4.pdf

[6]: Topological Autoencoders. - Michael Moor, 5月 25, 2026にアクセス、 https://michaelmoor.me/blog/topoae/main/

[7]: ICML Poster Topological Autoencoders, 5月 25, 2026にアクセス、 https://icml.cc/virtual/2020/poster/5851

[8]: Topological Autoencoders - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 http://proceedings.mlr.press/v119/moor20a/moor20a.pdf

[10]: Topology-Preserving Latent Organization - Emergent Mind, 5月 25, 2026にアクセス、 https://www.emergentmind.com/topics/topology-preserving-latent-organization

[13]: Classes are not Clusters: Improving Label-based Evaluation of Dimensionality Reduction - IEEE Xplore, 5月 25, 2026にアクセス、 https://ieeexplore.ieee.org/iel7/2945/10373160/10308618.pdf

[17]: Geometric Alignment Score (GAS) - Emergent Mind, 5月 25, 2026にアクセス、 https://www.emergentmind.com/topics/geometric-alignment-score-gas

[18]: Representation Topology Divergence: A Method for Comparing Neural... - OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=ljnUrvex8d

[21]: Representation Topology Divergence: a Method for Comparing Neural Network Representations - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 https://proceedings.mlr.press/v162/barannikov22a/barannikov22a.pdf

[22]: Topology-Preserving Deep Image Segmentation, 5月 25, 2026にアクセス、 http://papers.neurips.cc/paper/8803-topology-preserving-deep-image-segmentation.pdf

[23]: A Topology Layer for Machine Learning | SAIL Blog - Stanford AI Lab, 5月 25, 2026にアクセス、 https://ai.stanford.edu/blog/topologylayer/

[32]: Geometrical Methods in Machine Learning and Tensor Analysis, 5月 25, 2026にアクセス、 https://back.skoltech.ru/storage/app/media/defenses/2021/valentin-hrulkov/theme/thesis9.pdf

[33]: Learning metrics for persistence-based summaries and applications for graph classification - NIPS, 5月 25, 2026にアクセス、 http://papers.neurips.cc/paper/9178-learning-metrics-for-persistence-based-summaries-and-applications-for-graph-classification.pdf

[35]: Neural Reduced Potential via Persistent Homology - Machine Learning and the Physical Sciences, 5月 25, 2026にアクセス、 https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_334.pdf

[37]: Chapter 6 Distances and Stability, 5月 25, 2026にアクセス、 https://ti.inf.ethz.ch/ew/courses/TDA24/Chapter6.pdf

[38]: Sliced Wasserstein Kernel for Persistence Diagrams - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 http://proceedings.mlr.press/v70/carriere17a/carriere17a.pdf

[40]: computing wasserstein distance vs. bottleneck distance between persistence diagrams - Math Stack Exchange, 5月 25, 2026にアクセス、 https://math.stackexchange.com/questions/2952977/computing-wasserstein-distance-vs-bottleneck-distance-between-persistence-diagr

[41]: 5月 25, 2026にアクセス、 https://math.stackexchange.com/questions/2952977/computing-wasserstein-distance-vs-bottleneck-distance-between-persistence-diagr#:~:text=bottleneck%20distance%20between%20persistence%20diagrams,-Ask%20Question&text=According%20to%20the%20software%20Hera,computed%20by%20the%20auction%20algorithm.

[42]: n° 2017-82 Sliced Wasserstein Kernel for Persistence Diagrams M. CARRIÈRE1 M. CUTURI2 S. OUDOT3 - CREST, 5月 25, 2026にアクセス、 https://crest.science/RePEc/wpstorage/2017-82.pdf

[43]: Representation Topology Divergence: A Method for Comparing Neural Network Representations. - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 https://proceedings.mlr.press/v162/barannikov22a.html

[48]: Diagnosing Neural Convergence with Topological Alignment Spectra - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2411.08687v2

[53]: FROM DIVERGENCE TO NORMALIZED SIMILARITY: A SYMMETRIC AND SCALABLE TOPOLOGICAL TOOLKIT FOR REPRESENTATION ANALYSIS - OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/pdf/0b52789ca6fa104497a0ed2bd07042fd9eedebdf.pdf

[56]: STITCH: Surface reconstrucTion using Implicit neural representations with Topology Constraints and persistent Homology - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2412.18696v1

[57]: Topology-Aware CLIP Few-shot Learning - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2505.01694v1

[58]: TopoFR: A Closer Look at Topology Alignment on Face Recognition - OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=R4gqcDRJ9l

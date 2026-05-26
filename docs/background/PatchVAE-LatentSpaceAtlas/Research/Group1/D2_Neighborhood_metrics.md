# 潜在空間の幾何学および位相構造における評価指標の包括的分析：近傍メトリクスと表現の安定性

## 1. 序論

深層学習、多様体学習（Manifold Learning）、および次元削減（Dimensionality Reduction）の分野において、高次元データを低次元の潜在空間へマッピングする際の表現学習の評価は、長らく平均二乗誤差（MSE）に基づく再構成誤差や、単純なユークリッド距離の維持といった原始的な指標に依存してきた。しかしながら、画像、音声、単一細胞RNAシーケンス（scRNA-seq）などの複雑な自然データは、高次元空間全体に一様に分布しているわけではなく、本質的にははるかに低次元の非線形多様体（Non-linear Manifold）の表面または近傍に集中しているとする「多様体仮説（Manifold Hypothesis）」が広く支持されている [1]。この仮説を前提とした場合、単純な直線距離や再構成の良し悪しに基づく評価は、データの背後にある真の幾何学的・位相的（トポロジカル）構造の保存度を正確に反映しないという根本的な問題を抱えている [3]。
データ表現の質をより精密かつ数学的に厳密に評価するためには、局所的な近傍構造がどの程度維持されているか（Local Neighborhood Preservation）と、多様体全体の大域的な位相構造がどの程度破綻せずに維持されているか（Global Topology Preservation）を分離し、それぞれを定量化する評価指標（Topology Metrics）が不可欠である。特に、近傍メトリクス（Neighborhood metrics）と呼ばれる指標群は、パーシステント・ホモロジーのような純粋な代数トポロジー的計算に比べて計算コストが低く、実装が容易でありながら、局所多様体の性質を強力に捉えることができる [5]。
本報告書では、Neighborhood metricsに含まれる主要な指標群（kNN overlap、trustworthiness、continuity、distance-rank Spearman、graph geodesic correlation、local intrinsic dimension、Jacobian spectrum）について、その数理的定式化、幾何学的解釈、および適用限界を網羅的に詳解する。その上で、特定の研究課題である以下の3つの問いに対し、最新の研究動向と数学的裏付けを伴った解を提示する。第一に、局所近傍保存と大域Topology保存を分けて評価するための最適戦略は何か。第二に、Patch VAEとAtlas VAEというアーキテクチャの比較に最も適したトポロジー指標は何か。そして第三に、ニューラルネットワークの初期化シード（Seed）間で発生する表現のばらつき（Variability）を、表面的なアライメントの誤謬を避けていかに数値化するか、である。

## 2. 局所近傍保存の数理と評価指標

局所近傍保存を評価する指標は、主にデータポイントの周辺の局所的なトポロジー、すなわちk近傍グラフの構造、局所多様体の次元、および写像の接空間における微小な歪みが、変換前後でどの程度一致しているかを測定する。これらの指標は、モデルが局所的な特徴空間を破綻させることなく潜在空間にマッピングできているかを監査する役割を担う。

### 2.1 kNN Overlap (k-Nearest Neighbor Overlap)

kNN Overlapは、最も直感的かつ広く用いられる局所近傍メトリクスの一つである。この指標は、高次元の元データ空間と低次元の潜在空間（または別の表現空間）との間で、各データポイントの近傍集合がどの程度共通しているかを直接的に測定する [7]。
数学的には、高次元の元空間における点 x_i の k 近傍集合を N_k(x_i) とし、低次元空間における対応する点 y_i の k 近傍集合を M_k(y_i) としたとき、両空間における近傍の重なり合い（積集合のサイズ）を以下のように評価する。
この指標の最大の利点は、その計算論的単純さと解釈のしやすさにある。例えば、単一細胞RNAシーケンスデータの解析において、ダウンサンプリングや次元削減アルゴリズム（scLENSなど）が、元の複雑な局所構造をどの程度正確に保持しているかを実証するために頻繁に使用される [8]。また、kNNベースの近似探索（ANN）やオーバーラップグラフの構築といった分野においても、次元削減が検索精度に与える影響の指標として機能する [9]。しかしながら、kNN Overlapは近傍集合内の「距離の順位」の変動を一切考慮せず、指定された k の枠内に存在するか否かの二値的な評価に終始するため、近傍内での微細な局所的歪みや配置の逆転を捉える能力には限界がある。

### 2.2 Trustworthiness と Continuity

kNN Overlapの解像度不足を補完し、次元削減において発生する位相的エラーをより詳細にペナルティ化するために導入されるのが、順位ベースの非対称な評価指標であるTrustworthiness（信頼性）とContinuity（連続性）である [3]。Kaskiらによって提唱されたこれらの指標は、次元削減の過程で生じるエラーを「Hard Intrusions（重度な侵入）」と「Hard Extrusions（重度な押し出し）」という二つの相反する幾何学的現象に分類して評価する [3]。
**Trustworthiness（信頼性）**は、元の空間では遠く離れていたデータポイントが、次元削減の過程で誤って低次元空間の近傍に配置されてしまう現象（Intrusions）をペナルティ化する指標である [6]。数式は以下のように定式化される。
ここで、r_{i,j} は元の空間における点 i から見た点 j の距離の順位である。元の空間での順位 r_{i,j} が k より大きいほど（すなわち、本来は遠く離れた点であるにもかかわらず近傍に押し込まれた度合いが大きいほど）、ペナルティが線形に重くなる構造を持っている [6]。これにより、偽の近傍関係（False neighborhoods）が構築されることを厳しく監査する。
一方、**Continuity（連続性）**は、元の空間では近傍にあったデータポイントが、低次元空間において遠くに引き離され、引き裂かれてしまう現象（Extrusions）をペナルティ化する [3]。
ここで、\hat{r}_{i,j} は低次元空間における距離順位である [6]。正規化係数 \frac{2}{nk(2n - 3k - 1)} は、スコアが0から1の範囲に収まるように設計されており、値が1に近いほど元の位相が完璧に保存されていることを示す [3]。UMAPやt-SNEのようなアルゴリズムは、局所構造の保持を優先するためTrustworthinessが高くなる傾向があるが、パラメータ設定によってはContinuityが犠牲になり、連続した多様体が不自然に分断される現象が観察される。これらの指標をペアで使用することで、モデルが引き起こす局所歪みの「方向性」を正確に特定することが可能となる [3]。

### 2.3 Local Intrinsic Dimension (LID)

データの複雑さと局所構造を評価するためのより高度で幾何学的な指標が、局所内在次元（Local Intrinsic Dimension; LID）である [1]。LIDは、特定のクエリ点の周辺においてデータが効果的に展開される自由度の数、すなわち局所的なスケールに依存した多様体の実効次元数を推定する [1]。これは、データが埋め込まれている空間の絶対次元（例：画像のピクセル数）とは異なり、データ生成プロセスの真の自由度を表す。
LIDの推定手法としては、極値理論に基づくものや、ポアソン過程に基づく最尤推定量（MLE）などが標準的である [5]。点 x の周囲のデータ分布が、ある半径 R の微小な球の内部において均質なポアソン過程に従うと仮定した場合、局所次元は以下のように推定される。
ここで、T_j(x) は点 x から j 番目の最近傍点までの距離である [5]。また、Farahmand–Szepesvári–Audibert (FSA) メソッドと呼ばれる別の推定手法では、2つの入れ子になった近傍半径（ k 近傍と k/2 近傍）の比率に基づいて局所次元を定義する [5]。
LIDは、正規化流（Normalizing Flows）や拡散モデル（Diffusion Models）といった生成モデルにおいて、「モデルが局所的な密度と次元を維持できているか」を検証する強力な診断ツールとなる [1]。多様体の次元を暗黙的に仮定する古典的な推定器とは異なり、LIDはデータ駆動型で次元を検証するため、モデルが多様体の曲率や厚みによってバイアスを受けている領域を特定することが可能である [1]。LIDの崩壊は、潜在空間においてデータの意味的構造が不自然に潰されていることを示唆する。

### 2.4 Jacobian Spectrum

深層生成モデルやオートエンコーダの表現の質を、幾何学的な微分構造（接空間の性質）の観点から直接的に評価する指標がヤコビアン・スペクトル（Jacobian Spectrum）である [18]。入力 x から潜在変数 z へのマッピング f : \mathbb{R}^D \rightarrow \mathbb{R}^d の局所的な歪みは、ヤコビ行列 J_f(x) = \frac{\partial f(x)}{\partial x} の特異値分解（SVD）によって得られる特異値の分布（スペクトル）として捉えられる [18]。
局所等長写像（Local Isometry）が完全に成立している理想的な状態では、すべての特異値が1付近に集中し、全方向に対してスケーリングが等しく維持される（これをDynamical Isometryと呼ぶこともある） [21]。しかしながら、実際の潜在拡散モデル（LDMs）などの局所ヤコビアンスペクトルは、極めてヘビーテールな分布（Heavy-tailed distribution）を示すことが実証されている [18]。例えば、ある実証研究では、第一主固有値（ \lambda_1 ）のみで局所的な意味的変動の57.8%〜59.9%を説明するという「経験的支配（Empirical Dominance）」が観察されている [18]。
Jacobian Spectrumの極端な偏りや特定の特異値の崩壊（0への収束）は、多様体の局所的な構造が特定の方向に強く潰されていること、すなわち勾配の消失や多様体の特異点（Singularities）の発生を意味する [20]。逆に、スペクトルを明示的に制御し、グラスマン多様体やスティーフェル多様体上での等長性を強制するアプローチ（JPmHCなど）は、ネットワークの安定性と幾何学的構造の保持を劇的に向上させることが示されている [20]。

## 3. 大域的トポロジー保存の数理と評価指標

局所的な指標が微小な近傍関係に焦点を当てるのに対し、大域的なトポロジー指標は、遠く離れたデータポイント間の相対的関係や、多様体全体にわたる経路（マクロな幾何学構造）が保存されているかを評価する [6]。局所構造が保たれていても、データセット全体が大きく歪んで折り畳まれてしまう現象を防ぐためには、これらの指標による監査が不可欠である。

### 3.1 Distance-rank Spearman

元の高次元空間と低次元表現空間との間で、全データポイントペアの距離の「順位」が大域的に維持されているかを評価する指標が、Distance-rank Spearman（距離順位スピアマン相関）である [27]。ピアソンの積率相関係数が線形な関係を仮定するのに対し、スピアマンの順位相関係数は単調性（Monotonicity）のみを仮定するため、非線形な次元削減の評価に適している [28]。
数学的には、各空間におけるデータペア間の距離の順位を算出し、その順位の差 d_i を用いて以下の式で相関を計算する。
（ここで、N は比較対象となるデータペアの総数である。）
この指標は、線形な主成分分析（PCA）や古典的な多次元尺度構成法（MDS）の評価において極めて有効であり、データセットの全体的な構造の保存度を示す指標として機能する [28]。しかし、深刻な欠点も存在する。データが高度に非線形で湾曲した多様体（スイスロールなど）の上に存在する場合、元の高次元空間における単純なユークリッド距離は多様体上の真の距離を反映していない。そのため、ユークリッド距離に基づく順位を低次元空間で強制することは、逆に多様体の展開（Unrolling）を阻害し、本質的なトポロジーの破壊を招くリスクがある [28]。

### 3.2 Graph Geodesic Correlation

上記のような高次元空間におけるユークリッド距離の欠点を克服し、真の大域的トポロジーと多様体の連続性を評価するために用いられるのが「グラフ測地線相関（Graph Geodesic Correlation）」である [5]。このアプローチは、データが多様体上に制約されているという前提に立ち、空間の直線距離ではなく、多様体の表面に沿った最短経路（測地線）を距離の尺度として採用する。
Graph Geodesic Correlationの計算プロセスは以下の通りである。
 1. **近傍グラフの構築:** 高次元データ上で局所的な k 近傍グラフを構築し、多様体上の構造を離散的に近似する [32]。
 2. **測地線距離の推計:** 構築されたグラフ上で最短経路アルゴリズム（Dijkstra法など）を実行し、すべてのノードペア間の重み付き最短経路距離（Geodesic distance）を計算する [31]。
 3. **相関の算出:** 潜在空間においても同様に距離（あるいは表現空間上での測地線）を計算し、元の高次元多様体上の測地線距離と、低次元空間での距離との間のSpearman相関またはPearson相関を算出する [32]。
この指標は、多様体の「中規模から大規模（Meso- to Global-scale）」の幾何学構造が保存されているかを測る上で最も信頼性の高いアプローチの一つである [33]。例えば、単一細胞解析のための次元削減ツールであるTopOMetryの枠組みでは、細胞の分化プロセスのような大域的かつ連続的な多様体構造（軌跡）が、潜在空間において正しく維持されているかを証明するために測地線相関が用いられている [5]。さらに、脳波（EEG）などの空間共分散行列に基づく対称正定値（SPD）多様体の表現学習においても、DeepGeoCCAなどの手法が測地線相関の最大化を目的関数として組み込んでいる [35]。
ただし、測地線相関にも構造上の弱点が存在する。データセットが互いに交わらない複数の独立した部分多様体（Disjoint submanifolds）で構成されている場合、別々のコンポーネント間にパスが存在しないため測地線距離は無限大（または計算不可）となり、大域的な距離スコアに著しいバイアスが生じるリスクがある [5]。したがって、単一細胞データのように複数の独立した細胞株が混在する環境では、解釈に注意を要する。

## 4. 調査課題1：局所近傍保存と大域Topology保存の分離評価戦略

前述の数理的背景を踏まえ、最初の調査課題である「局所近傍保存と大域topology保存を分けて評価するには何を使うべきか？」に対する実践的かつ理論的な戦略を提示する。
次元削減や表現学習の出力を評価する際、単一の指標でモデルの良し悪しを決定することは幾何学的に不可能である。局所的な近傍構造の破綻（例：局所ノイズの混入による近傍の誤配置）と、大域的な距離関係の歪み（例：遠くのクラスター同士が誤って接近する現象）は、根本的に異なる幾何学的メカニズムに起因するからである [13]。したがって、評価は「局所적忠実度の監査」と「大域的構造維持の監査」の2軸で直交的に実施されるべきである。

### 4.1 評価フレームワークの構成
以下の表は、各評価指標が監査する幾何学的性質と、その適用限界を体系的に整理したものである。

| 評価の目的 | 推奨される評価指標群 | 測定する幾何学的性質とメカニズム | 適用限界・注意点 |
|---|---|---|---|
| **局所近傍保存** | Trustworthiness / Continuity | 近傍への侵入（Intrusions）と押し出し（Extrusions）の非対称エラーを順位ペナルティで検出する。 | 局所的な順位のみを考慮し、大域的なクラスタ間の相対的な配置関係は完全に無視される。 |
|  | kNN Overlap | 低次元空間と高次元空間でのk近傍集合の一致率を算出する。 | 集合内の距離の順位の変動を捉えられず、二値的な評価に留まる。 |
|  | Local Intrinsic Dimension (LID) | 局所多様体の次元の潰れや、多様体の厚み・曲率の変化を推定器（MLE等）で検証する。 | サンプルサイズが小さい場合やノイズが多い場合、推定器の分散が大きくなる。 |
|  | Jacobian Spectrum | マッピングの局所等長性と歪みを、ヤコビ行列の特異値分布の崩壊から直接監査する。 | 微分可能なニューラルネットワークモデルにのみ適用可能であり、解析的定式化が必要。 |
| **大域Topology保存** | Graph Geodesic Correlation | 局所kNNグラフ上の最短経路を通じて、多様体全体の大域的な連続性と軌跡構造を評価する。 | 孤立した部分多様体（Disjoint submanifolds）間の評価が困難であり、距離が発散する。 |
|  | Distance-rank Spearman | データセット全体を通した相対距離の順位を相関係数として算出する。 | 高次元で湾曲が強い多様体の場合、ユークリッド距離に基づく順位付け自体が不適切となる。 |
|  | Spectral Procrustes (拡散写像) | 推移確率と拡散座標（Diffusion coordinates）を直交プロクルステス法で整列させ、メソスケールの幾何学を比較する。 | 大規模データセットにおける固有値分解と行列のアライメントは計算コストが極めて高い。 |

**結論と推奨アプローチ:** 局所と大域を分離して評価するための最適解は、単一の指標に依存せず、各軸から最も堅牢な指標を選択してプロファイリングすることである。具体的には、**「Trustworthiness / Continuity」のペア**を用いて微小な構造破綻（偽の近傍関係の構築や多様体の裂け目）を監視し、同時に**「Graph Geodesic Correlation」**を用いて、局所的には捉えきれない多様体全体の大域的配置や連続的な推移が維持されているかを交差検証するアプローチが必須である [14]。モデル開発において内部の幾何学を最適化する場合は、Jacobian Spectrumの監視を追加することで、等長性の維持という観点から勾配レベルでの監査が可能となる [22]。

## 5. 調査課題2：Patch VAEとAtlas VAEの比較に向くトポロジー指標

第二の調査課題は、生成モデルにおける表現拡張アーキテクチャである「Patch VAE」と「Atlas VAE」を比較するために、どの指標が最適であるかという問いである。この問いに答えるためには、まず両アーキテクチャの多様体に対するアプローチ（局所と大域の扱い方）の根本的な違いを理解する必要がある。

### 5.1 アーキテクチャの幾何学的・位相的特性の比較

 * **Patch VAE（パッチベースVAE）:** Patch VAEは、入力画像などの高次元データを独立した複数のパッチ（局所的な小領域）に分割し、それぞれのパッチの潜在表現をボトルネックを介して学習するアーキテクチャである [37]。この手法は、少数のサンプルからの多様なデータ生成（単一画像からのビデオ生成など）や、微細なテクスチャ、部品のパターン、中間レベルのスタイル表現（mid-level style representations）の局所的な学習に極めて優れている [40]。しかしながら、Patch VAEは「局所的な多様性と解像度」を確保することには長けている反面、各パッチの潜在コードが独立して処理されるため、多様体全体の大域的な位相的制約（Global topological constraints）を統合し、維持するメカニズムを本質的に持たない [38]。
 * **Atlas VAE（多様体アトラス・オートエンコーダ）:** これに対し、Atlas VAE（またはMixture of Autoencoders）は、微分幾何学における「多様体（Manifold）は局所的にはユークリッド空間と同相である（チャート/座標近傍）が、大域的には複雑な位相構造を持ち得る」という概念を、ニューラルネットワークに直接組み込んだアーキテクチャである [2]。Atlas VAEは、データの多様体を覆う複数の「チャート（局所的な微分同相写像）」を、個別のエンコーダ・デコーダのペアとして学習する [2]。Patch VAEとの決定的な違いは、画像空間を物理的に切り刻むのではなく、多様体の状態空間をチャートで覆い、各チャートが重なり合う領域（Overlap regions）において「推移写像（Transition maps）」を計算する点にある [43]。このチャート間のオーバーラップの重みを用いて神経複体（Nerve complex）を構築し、ホッジ・ラプラシアンやベッチ数（Betti numbers）といったトポロジカルな不変量を抽出することで、多様体の連結性、穴の数、メビウスの帯のような非向き付け可能性（non-orientability）などの大域적位相構造を明示的に保持し、再構成することが可能となる [43]。
 
### 5.2 結論：Patch VAEとAtlas VAEの比較に最適な指標群

両者の性能を本質的に比較するためには、単なる画像の再構成誤差ではなく、「局所表現の精度（チャート/パッチの歪み）」と「境界領域のトポロジカルな整合性（大域への統合）」を定量化する指標群が適している。
 1. **Jacobian Spectrum（局所同相性とマッピングの歪みの比較）** Atlas VAEの理論的根拠は、各オートエンコーダが「微分同相写像（Diffeomorphism）」として機能することである [43]。微分同相であるためには、ヤコビ行列が特異にならない（非退化である）ことが必須条件である。Jacobian Spectrumを評価することで、モデルがデータを潜在空間にマッピングする際に生じる幾何学的歪みを数値化できる [20]。Atlas VAEはチャートの定義上、ヤコビアンスペクトルが安定し、極端な0特異値を持たない局所等長写像に近い挙動を示すことが期待される。一方、Patch VAEはパッチの境界付近や、大域的文脈を失った領域においてスペクトルが崩壊しやすい。この指標は、局所マッピングの「品質」を比較する上で極めて有効である。
 2. **Graph Geodesic Correlation（大域的なパッチ統合とチャート重なりの評価）** Patch VAEの最大の弱点は、パッチ間にまたがる大域的な構造の分断である。一方、Atlas VAEは神経複体と推移写像を用いて大域トポロジーを構築する [2]。Graph Geodesic Correlationを用いることで、遠く離れた2点間の距離（複数のパッチやチャートを跨ぐ経路）が、モデルを通過した後に正しく維持されているかを評価できる。局所パッチ内でのユークリッド距離は両モデルとも保存するが、大域的な測地線相関においては、位相を明示的にモデリングするAtlas VAEが、Patch VAEを圧倒的に上回るスコアを示す。これにより、アーキテクチャが「大域構造を理解しているか」を厳密に比較できる。
 3. **Local Intrinsic Dimension (LID) の境界領域における連続性分析** 多様体の局所内在次元（LID）は、データ構造の特異点や複雑さを捉える [1]。Patch VAEではパッチごとに独立して次元削減が行われるため、パッチ間の境界領域（画像空間での継ぎ目など）において、潜在空間内のLIDが不自然に急変、または崩壊する現象が予測される。一方、Atlas VAEでは、チャートのオーバーラップ領域において厳密な推移写像が保証されているため [46]、多様体全体にわたってLIDの連続性が保たれる。したがって、データポイント群に対するLIDの空間的変動や分散を測ることで、表現空間の「滑らかさ（Smoothness）と継ぎ目のなさ」を比較することが可能となる。

| 比較観点 | 最適な評価指標 | Atlas VAEの特性と期待される結果 | Patch VAEの特性と期待される結果 |
|---|---|---|---|
| **局所マッピングの品質と歪み** | Jacobian Spectrum | 各チャートが微分同相写像として機能し、等長的なスペクトルを維持する。 | 独立したテクスチャ抽出に特化し、境界付近でスペクトルが歪む傾向にある。 |
| **複数領域を跨ぐ構造の統合性** | Graph Geodesic Correlation | 推移写像により大域的な測地線相関が高く維持される。 | 大域構造の統合機能が弱く、長距離の測地線相関が大きく低下する。 |
| **境界領域の幾何学的連続性** | LID (Local Intrinsic Dimension) | オーバーラップ領域でもLIDが特異点を持たず滑らかに推移する。 | パッチの継ぎ目においてLIDが急激に崩壊、または不連続になる。 |

## 6. 調査課題3：Seed間Variabilityの数値化と幾何学的安定性

最後の調査課題は、深層学習モデルにおける「初期化シード（Seed）間のばらつき（Variability）」をどのように数値化するかという問いである。ニューラルネットワークは、同一のアーキテクチャ、同一のハイパーパラメータ、同一のデータセットを用いて訓練した場合であっても、初期化のランダムシードが異なるだけで、確率的勾配降下法（SGD）による学習過程の軌跡が大きく分岐し、最終的に形成される潜在空間（Latent Space）の絶対座標が全く異なるものとなる [49]。この「表現のばらつき」を定量化し、モデルが普遍的な概念的構造へ収束しているか（Convergence to universal conceptual structures）を評価することは、AIの信頼性、転移学習の性能予測、および表現のアライメント監査において極めて重要である [50]。

### 6.1 既存の類似度指標（Similarity Metrics）とその盲点

これまで、シード間のばらつきを評価するためには、絶対的な座標空間の違いを吸収し、直交変換（回転など）に対して不変な形で「表現の類似度（Similarity）」を測る手法が標準的に用いられてきた [49]。
 * **Procrustes Analysis（プロクルステス解析）:** 2つの異なるシードから得られた潜在空間表現のセットに対し、直交変換（回転・反転）、スケーリング、および並行移動を適用し、両者のユークリッド距離（フロベニウスノルム）を最小化するような最適な幾何学的整列（Alignment）を見つけ出す手法である [49]。
 * **CKA (Centered Kernel Alignment):** 各表現空間におけるデータペアの内積からグラム行列（またはカーネル行列）を計算し、Hilbert-Schmidt Independence Criterion (HSIC) に基づいて、2つのカーネル行列間の類似度を0から1の範囲で正規化して出力する [49]。
 * **RSA (Representational Similarity Analysis):** 各シードの潜在空間において表現非類似度行列（RDM: Representational Dissimilarity Matrix）を個別に計算し、その行列間のSpearman相関を比較する手法である [49]。
**既存指標の深刻な欠点（The Spectral Bias / Low-rank Dominance）:** 近年、CKAやProcrustes解析といった従来の「類似度指標（Similarity Metrics）」が、多様体の本質的な構造破壊に対して盲目（Blind spot）を持つことが明らかになっている [52]。これらの指標は、固有値スペクトル（Eigenspectrum）の上位の少数の主成分（Principal Components）に極端に支配される（Low-rank dominance）という数学的性質を持つ [52]。例えば、モデルの圧縮や表現の劣化によって、多様体の詳細なトポロジーを構成する低分散の次元（ハイランクな次元）が完全に破壊されたとしても、分散の大部分を占める上位の数次元が一致していれば、CKAは「シード間で非常に類似している（スコアが高い）」という誤ったシグナルを返してしまうのである [55]。すなわち、CKAやProcrustesは表現の「類似性（Similarity）」は測れても、表現の「構造的完全性・堅牢性（Structural Integrity）」を測ることはできない [52]。

### 6.2 結論：幾何学的安定性の新指標「Shesha」によるVariabilityの数値化

シード間のばらつき（Variability）を、表面的なアライメントの誤謬を避けて「構造的安定性の維持」という観点から厳密に数値化するために導入された最新の枠組みが、幾何学的安定性（Geometric Stability）を測定する指標**「Shesha」**である [52]。

#### Sheshaの数学的メカニズム

Sheshaは、単一の表現システム内部の「自己一貫性（Self-consistency）」と「摂動に対する堅牢性」を、スプリット・ハーフ相関（Split-half correlation）を用いて評価することで、シード間の幾何学的崩壊を検知する [56]。
 1. **特徴量の分割:** 評価対象となる各シードの潜在特徴量の次元（またはデータ入力）を、ランダムに2つの相補的なサブセット（例えば奇数次元と偶数次元の半分ずつ）に分割する [57]。
 2. **RDMの構築:** 分割された各半分（Split-half）の表現空間において、データポイントの重心間のコサイン距離等を用い、表現非類似度行列（RDM）を個別に計算する [57]。
   
 3. **相関の算出:** 得られた2つのRDMの上三角行列をベクトル化し、それらの間のSpearman順位相関を計算する。これを複数回のランダム分割（例： K = 30 回）にわたって平均化し、最終的なSheshaスコアとする [56]。
 
#### SheshaがSeed間Variabilityの数値化に最適な理由

 * **完全な固有値スペクトルへの感度 (Sensitivity to the full eigenspectrum):** CKAがトップ1の主成分を除去しただけでスコアが崩壊するのに対し、Sheshaはスペクトルのテール部分（微細な多様体構造）まで一貫して感度を維持する [52]。これにより、シード間の変動において「主要な軸は似ているが、細部の幾何学が崩壊している」という現象を正確に検出できる。
 * **幾何学的税制 (Geometric Tax) の可視化:** シード間の転移可能性（Transferability）を高めるために最適化されたモデルは、しばしば表現の幾何学的情報を少数の次元に極端に集中させる（Compressions）。この場合、ランダムに特徴量を分割した際、一方は情報過多、もう一方はノイズのみとなり、Sheshaスコアは急激に低下する [52]。Sheshaは、この「幾何学的な税（Geometric Tax）」を支払って構造的冗長性を失っていることを明確に数値化する [52]。
 * **直交変換への非不変性 (Non-invariance to orthogonal transformations):** CKAやProcrustesは直交変換に対して不変（Invariant）であるように設計されている [52]。しかし、現実のニューラルネットワークにおいて、特徴空間の回転や座標基底への分散の再配置は、下流タスクにおける特徴の読み出し（Readout）や解釈可能性に決定的な違いを生む。Sheshaは直交変換に対してあえて不変性を持たないため、シードの違いによる幾何学的情報の基底への「再配分（Redistribution）」をVariabilityとして厳密に検知する [52]。
 
### 6.3 シード間Variabilityの総合的な数値化戦略

以上の知見から、シード間のばらつきを包括的に数値化するための評価プロトコルは、以下の2階層の枠組みで構築されるべきである。

| 評価階層 | 推奨される評価指標 | 評価の目的とメカニズム |
|---|---|---|
| **第1段階：マクロなアライメントと類似度の測定** | Centered Kernel Alignment (CKA) または Procrustes Analysis | 異なるシード間で、主要な分散を説明するトップレベルの多様体配置がどの程度一致しているかを、直交変換の不変性を前提としてベースライン評価する [49]。 |
| **第2段階：構造の堅牢性と微細な多様体歪みの測定** | Shesha (Geometric Stability) | 表現が少数の次元に過剰依存していないか、シードの差異が多様体の本質的な距離関係（フル・固有値スペクトル）をどの程度劣化・変動させているかを、RDMのスプリット・ハーフ相関によって厳密に監査する [52]。 |

この「Similarity（類似度）」と「Stability（安定性）」という、経験的にも無相関（ \rho \approx -0.01 ）であることが証明されている2つの独立した軸を併用することで [57]、シード間変動の実態を、表面的なアライメントの誤謬に陥ることなく極めて高精度に数値化することが可能となる。

## 7. 総括

本報告書では、表現学習および次元削減アルゴリズムにおける潜在空間の評価について、特にNeighborhood metrics群を中心とした幾何学的・トポロジー的評価指標の理論的背景と適用戦略を詳細に論じた。局所的なトポロジーの破綻はTrustworthinessやLID、Jacobian Spectrumによって検知され、大域的な構造の崩壊はGraph Geodesic Correlation等によって監査される。これらの指標は、Patch VAEとAtlas VAEのようなアーキテクチャの多様体学習能力を本質的に比較するための強力なツールとなる。さらに、シード間のばらつきといった表現の安定性の評価においては、従来のCKAが抱える低ランク次元への偏重という盲点を克服するため、固有値スペクトル全体に感度を持つGeometric Stability（Shesha）の導入が不可欠であることを示した。多様体仮説に基づくこれらの高度な計量手法の統合的運用は、次世代の深層学習モデルにおける表現の堅牢性と解釈可能性を担保するための基盤となる。

## 8. 引用文献

[1]: Estimating local intrinsic dimension via density estimation - The Institutional Repository of the University of Warsaw (ReIn UW), 5月 25, 2026にアクセス、 https://repozytorium.uw.edu.pl/bitstreams/e7b0a3f9-bd74-4a07-aef1-15154a02005a/download

[2]: Manifold Learning by Mixture Models of VAEs for Inverse Problems, 5月 25, 2026にアクセス、 https://jmlr.org/papers/volume25/23-0396/23-0396.pdf

[3]: pyDRMetrics - A Python toolkit for dimensionality reduction quality assessment - PMC, 5月 25, 2026にアクセス、 https://pmc.ncbi.nlm.nih.gov/articles/PMC7887408/

[5]: TopoMetry systematically learns and evaluates the latent geometry ..., 5月 25, 2026にアクセス、 https://www.biorxiv.org/content/10.1101/2022.03.14.484134v5.full-text

[6]: Trustworthy Dimensionality Reduction A dissertation report presented for the completion of the degree of Master of Statistics (M. Stat.) from Indian Statistical Institute - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2405.05868v1

[7]: The landscape of biomedical research | bioRxiv, 5月 25, 2026にアクセス、 https://www.biorxiv.org/content/10.1101/2023.04.10.536208v2.full-text

[8]: scLENS: data-driven signal detection for unbiased scRNA-seq data analysis - PMC - NIH, 5月 25, 2026にアクセス、 https://pmc.ncbi.nlm.nih.gov/articles/PMC11519519/

[9]: Fast Approximate kNN Graph Construction for High Dimensional Data via Recursive Lanczos Bisection - Journal of Machine Learning Research, 5月 25, 2026にアクセス、 https://www.jmlr.org/papers/volume10/chen09b/chen09b.pdf

[13]: The advantages of our proposed Saturn coefficient over continuity and trustworthiness for UMAP dimensionality reduction evaluation - PeerJ, 5月 25, 2026にアクセス、 https://peerj.com/articles/cs-3424/

[14]: Evaluating Manifold Learning Techniques for Dimensionality Reduction on Industrial IoT Cybersecurity Data - IEEE Xplore, 5月 25, 2026にアクセス、 https://ieeexplore.ieee.org/iel8/6287639/11323511/11373316.pdf

[18]: Geometric Decoupling: Diagnosing the Structural Instability of Latent - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/pdf/2604.18804

[20]: JPmHC Dynamical Isometry via Orthogonal Hyper-Connections - ChatPaper, 5月 25, 2026にアクセス、 https://chatpaper.com/paper/239240

[21]: The Emergence of Spectral Universality in Deep Networks - Proceedings of Machine Learning Research, 5月 25, 2026にアクセス、 https://proceedings.mlr.press/v84/pennington18a/pennington18a.pdf

[22]: Geometric Decoupling: Diagnosing the Structural Instability of Latent - ResearchGate, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/404059013_Geometric_Decoupling_Diagnosing_the_Structural_Instability_of_Latent

[27]: Correlation Coefficient based Supervised Locally Linear Embedding for Pulmonary Nodule Recognition - PMC, 5月 25, 2026にアクセス、 https://pmc.ncbi.nlm.nih.gov/articles/PMC5076559/

[28]: Non-Linear Dimensionality Reduction :, 5月 25, 2026にアクセス、 https://proceedings.mlr.press/r0/vel95a/vel95a.pdf

[31]: TopOMetry systematically learns and evaluates the latent dimensions of single-cell atlases - eLife, 5月 25, 2026にアクセス、 https://elifesciences.org/reviewed-preprints/100361v1

[32]: TopoMetry systematically learns and evaluates the latent geometry of single-cell data - eLife, 5月 25, 2026にアクセス、 https://elifesciences.org/reviewed-preprints/100361

[33]: Riemannian generative decoder - OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=zgKHfTXG92

[35]: DEEP GEODESIC CANONICAL CORRELATION ANALY- SIS FOR COVARIANCE-BASED NEUROIMAGING DATA - ICLR Proceedings, 5月 25, 2026にアクセス、 https://proceedings.iclr.cc/paper_files/paper/2024/file/9f4d04276a5277b3c8478d05da701bf6-Paper-Conference.pdf

[37]: PyTorch implementation of "PatchVAE: Learning Local Latent Codes for Recognition" to appear in CVPR 2020 - GitHub, 5月 25, 2026にアクセス、 https://github.com/kampta/PatchVAE

[38]: Hierarchical Patch VAE-GAN: Generating Diverse Videos from a Single Sample, 5月 25, 2026にアクセス、 https://proceedings.neurips.cc/paper_files/paper/2020/file/c2f32522a84d5e6357e6abac087f1b0b-Paper.pdf

[40]: PatchVAE: Learning Local Latent Codes for Recognition | OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=r1x1kJHKDH

[43]: [2303.15244] Manifold Learning by Mixture Models of VAEs for Inverse Problems - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/abs/2303.15244

[46]: Autoencoder atlas for the Möbius band (Section 7.3). The two-chart... - ResearchGate, 5月 25, 2026にアクセス、 https://www.researchgate.net/figure/Autoencoder-atlas-for-the-Moebius-band-Section-73-The-two-chart-cover-produces-an_fig4_401278940

[49]: Relative Geometry of Neural Forecasters: Linking Accuracy and Alignment in Learned Latent Geometry - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2602.15676v1

[50]: Diagnosing Neural Convergence with Topological Alignment Spectra - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2411.08687v2

[52]: Geometric Stability: The Missing Axis of Representations - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2601.09173v4

[55]: Universal scale-free representations in human visual cortex - PMC, 5月 25, 2026にアクセス、 https://pmc.ncbi.nlm.nih.gov/articles/PMC12654933/

[56]: Geometric Stability: The Missing Axis of Representations, 5月 25, 2026にアクセス、 https://raju.ai/Articles/Shesha_foundation_Preprint.pdf

[57]: Geometric Stability: The Missing Axis of Representations - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2601.09173v1

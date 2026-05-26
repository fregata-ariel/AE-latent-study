# 深層生成モデルにおけるトポロジーの不一致とSynthetic Benchmarkの包括的評価

## 1. 導入：深層生成モデルにおけるトポロジー的ミスマッチの数理的背景

深層生成モデル、とりわけ変分オートエンコーダ（Variational Autoencoder: VAE）は、高次元の観測データを連続的な低次元潜在空間へと圧縮し、元のデータ分布を再構成する強力な枠組みとして広く普及している。しかし、連続的な潜在変数を扱う標準的なVAEは、「圧縮率（Compression Rate）」「再構成の忠実度（Reconstruction Fidelity）」「潜在空間の規則性（Latent Space Regularity）」という3つの最適化目標の間で根本的な対立状態に陥ることが知られており、これは「Rate-Distortion-Regularity Trilemma」として厳密に定式化されている [1]。このトリレンマの深層に潜む最も決定的な要因が、本報告書の主題である「トポロジーの不一致（Topological Mismatch）」である。
標準的なVAEのアーキテクチャでは、潜在空間に対して等方性ガウス分布（Isotropic Gaussian Prior）などのユークリッド空間上の単純で平坦な事前分布が仮定される [1]。しかし、実世界のデータや高度なSynthetic Benchmarkが構成するデータ多様体（Data Manifold）は、しばしばユークリッド空間と同相（Homeomorphic）ではない。多様体仮説に従えば、データは高次元空間内の低次元多様体上に分布するが、その多様体がコンパクトであったり、複数のサイクル（穴）を持っていたり、非向き付け可能であったりする場合、それを単一の平坦なユークリッド空間へと連続的かつ全単射にマッピングしようとする試みは、微分位相幾何学の定理により必然的に破綻する [2]。
このトポロジーの不一致は、潜在空間における空間の極端な「引き伸ばし（Stretching）」や「収縮（Shrinkage）」、あるいは位相的連続性の致命的な破れである「引き裂き（Tearing）」といった形で発現する [3]。連続性が破綻した場合、潜在空間内での測地線補間（Geodesic Interpolation）やゼロショット分類は意味をなさなくなり、未知のデータポイントに対してモデルは完全に信頼性を失う [4]。さらに、潜在表現の引き伸ばしや圧縮によるアーティファクトは、フィッシャー情報量などの計量に基づく内発的ノイズゲインの評価を歪め、真のデータ複雑性の把握を困難にする [6]。
したがって、生成モデルのトポロジー適合性を検証し、研究を数学的に安定させるためには、トポロジー的障害を意図的に内包したSynthetic Benchmark（人工的な多様体データセット）の導入が不可欠である。本報告書では、単一のVAE（単一のユークリッドチャート）が持つ限界を明確に露呈させるSynthetic Manifoldの選定基準、局所的なパッチ入力と大域的なAtlas（地図帳）アンサンブルの差異を浮き彫りにするベンチマーク構造、および既存の最先端研究で用いられているトポロジー評価指標について、網羅的かつ深層的な分析を提供する。

## 2. 単一チャート（単一VAE）の限界を証明するSynthetic Manifolds

多様体学習の究極の目標は、高次元空間に非線形に埋め込まれた低次元多様体を表現することである。多様体の数学的定義上、多様体上の任意の点は局所的にはユークリッド空間と同相であるが、大域的なトポロジーが非自明な場合、多様体全体を覆う単一の座標系（単一のチャート）を構築することは不可能である。標準的なVAEは、エンコーダとデコーダの単一のペアによってデータ空間と潜在空間の間に大域的な連続写像（Global Chart）を強制的に構築しようとするため、非自明なトポロジーを持つデータに対してその表現能力の限界を明確に露呈する [7]。
単一のVAEアーキテクチャが直面する幾何学的および位相的な限界を明らかにするため、以下に主要なSynthetic Manifoldの候補とその数理的特性、およびVAEにおける特有の破綻モードを詳述する。

### 2.1. 円 (S^1) と球面 (S^2)：特異点の発生と測度の集中

円（S^1）および球面（S^2）は最も基礎的な閉多様体であるが、単一のユークリッド空間とは決して同相にならない。円上のデータを標準的なガウス事前分布を持つVAEで学習させた場合、潜在空間である直線（\mathbb{R}^1）上に周期的な構造をマッピングしなければならず、結果として多様体の連続性がどこか一点で必ず切断（Tearing）される現象が発生する [9]。
さらに次元を上げた球面（S^2）においては、事態はより深刻である。球面データを平坦な平面（\mathbb{R}^2）にマッピングしようとする操作は、地球儀をメルカトル図法などの平面地図に投影する試みと同義であり、必然的に極点において無限の引き伸ばしや「特異点（Singularities）」の形成を余余儀なくされる [11]。既存研究においては、球面データを自然に扱うために、ガウス分布の代わりにVon Mises-Fisher（vMF）分布を事前分布として用いるSpherical VAE（Hyperspherical VAE）が提案されている [11]。これにより、データのセマンティックな同一性がベクトルの大きさ（Magnitude）ではなく角度（Angular orientation）に基づいてエンコードされ、「トポロジーの整合性（Topological Alignment）」と高次元空間特有の「測度の集中（Concentration of Measure）」の問題が同時に解決されることが実証されている [11]。逆に言えば、S^2 ベンチマークは、標準的なガウス事前分布がいかにデータを歪め、特定領域のデータポイントを潜在空間上で崩壊させるかを定量化するための最も単純かつ強力な基準となる。

### 2.2. トーラス (T^2)：周期境界の喪失とサイクルの崩壊

トーラス T^2 （数学的には S^1 \times S^1 と定義される）は、2つの独立した1次元のサイクル（ループ）を持つ曲面である。トーラス上のデータを標準的なVAEで近似しようとすると、多様体と潜在空間のトポロジーが一致しないため、モデルは2つの独立した周期境界条件を同時に満たすことができず、生成される多様体が不自然に引き裂かれたり、複数のコンポーネントが誤って接続されたりする現象が数値実験により明確に確認されている [14]。
トーラスは、生成モデルが「局所的な平坦性」のみならず「大域的な閉じた経路」を学習できるかを評価する上で極めて重要である。局所的な主成分分析（Local PCA）のような平坦性に基づく次元推定量は、トーラスのような非線形に埋め込まれた多様体において著しく性能が劣化する [16]。トーラスをSynthetic Benchmarkとして用いることで、単一のVAEがサイクルの終端と始端を接続できずに生じる大域的な「ギャップ（断絶）」や、無理なマッピングによる「重なり」に起因する再構成誤差を視覚的かつ定量的に評価することが可能となる [17]。

### 2.3. クラインの壺（Klein Bottle）：非向き付け可能性による致命的歪み

単一のVAEの限界を最も残酷かつ明確に露呈させる多様体が「クラインの壺（Klein Bottle）」である。クラインの壺は、境界を持たない非向き付け可能（Non-orientable）な2次元多様体である。この多様体を3次元ユークリッド空間 \mathbb{R}^3 に埋め込もうとすると必ず自己交差（Self-intersection）が生じるため、歪みなく完全に埋め込むには少なくとも4次元、あるいは5次元の空間が必要となる [19]。
既存研究（例えばTopoVAEやGD-VAE、KleinVAEなど）において、クラインの壺は極めて高度なストレステストとして利用されている [9]。t-SNE、UMAP、標準VAEなどの次元削減手法を用いてクラインの壺を低次元（例えば3次元以下）にマッピングしようとすると、自己交差部分で「ピンチ（Pinch）」や極端な歪みが発生し、元のトポロジー構造が完全に破壊されてしまう [19]。クラインの壺のトポロジーを深層生成モデルに学習させるためには、単純な指数写像（Lie-exp map）ではなく、無限葉被覆（Infinite-sheet covering map）を通じた高度な再パラメータ化トリックが必要となることが数理的に示されている [9]。さらに興味深いことに、自然画像データのパッチ空間（3x3ピクセルの高コントラストパッチ）がクラインの壺と同相になるという古典的な発見があり、クラインの壺は単なる人工的なトイデータに留まらず、視覚野の受容野モデルやテクスチャ表現の検証においても極めて実用的なベンチマークとして機能する [23]。

### 2.4. メビウスの帯（Möbius Strip）：ツイストによる局所的破綻

メビウスの帯は非向き付け可能性と「ツイスト（Twist）」および「境界（Edge）」を持つため、標準的な連続正規化流（Continuous Normalizing Flows）やVAEにとって学習が極めて困難な多様体である [25]。既存研究では、Möbius多様体の非自明なトポロジーが、直交基底の学習やスパース学習を著しく阻害することが指摘されている [12]。メビウスの帯上を一周して元の位置に戻ると法線ベクトルが反転するため、単一の連続的なチャートではこの反転を表現できず、ネットワークはツイスト部分で必ず特異点を生み出すか、勾配消失を引き起こす。この性質により、メビウスの帯は単一チャートモデルの「大域的一貫性の欠如」を証明するための強力なベンチマークとなる。

## 3. 高度な対称性と商空間に基づくSynthetic Benchmarks

単一VAEの限界を示すベンチマークとして、S^1 やトーラスのような基本的な位相空間に加え、より高度な対称性や代数的構造を持つ空間も近年注目を集めている。これらは、データセットに内在する変換不変性や同変性（Equivariance）をモデルが正しく獲得できているかを評価するために用いられる。

### 3.1. Lie群 SO(2) および SO(3)：連続変換と対称性欠陥

回転群である SO(2) や SO(3) は、それ自体が滑らかな多様体（Lie群）を構成する。特に SO(3) は、3次元空間における物体の回転やカメラ視点の変化など、実世界の視覚データに頻繁に現れる物理的変換を表現する [26]。これらの群構造を単一のユークリッド潜在空間にマッピングしようとすると、回転の周期性や非可換性（SO(3) において回転の順序が結果を変える性質）がユークリッド空間の線形性と衝突する。
このミスマッチを定量化するために、「Symmetry Defect（対称性欠陥）」や「Lie代数交換子の不一致（Lie-algebraic commutator mismatch）」という指標が用いられる [28]。標準的なVAEでは、潜在空間上での線形補間がデータ空間上での滑らかな回転に対応しないため、このSymmetry Defectが著しく大きくなる。モデルがこれらの群構造を正しく学習できるかを問うことは、物理法則や対称性を維持した生成が可能かどうかを検証することに他ならない。

### 3.2. 商空間（Quotient Space）と基本領域（Fundamental Domain）

商空間は、ある位相空間に対して同値関係（Equivalence Relation）を導入し、同値な点同士を同一視することによって得られる空間である。例えば、実数直線 \mathbb{R} を整数 \mathbb{Z} の加法群による作用で割った商空間 \mathbb{R}/\mathbb{Z} （すなわち x \equiv y \pmod 1）は、円 S^1 と同相になる [9]。また、球面の対蹠点（Antipodal points）を同一視した射影空間（\mathbb{R}P^2）も代表的な商空間である [10]。
群作用を伴う基本領域（Fundamental domain with group action）をベンチマークとして用いる場合、データは特定の対称性を持つ領域内に制限され、境界を越えると群作用によって反対側の境界へとワープするような挙動を示す。単一のVAEはこのような「空間のジャンプ（同一視）」をユークリッド的な距離関数（MSEなど）で評価してしまうため、同一視されるべき点が潜在空間上で最も遠く離れて配置されるという致命的なエラーを引き起こす。

### 3.3. モジュラー領域（Modular Domain）と格子・組成データ

格子（Lattice）構造やテータ関数的な周期データ、あるいはモジュラー領域におけるデータ分布は、結晶構造や分子配置、さらには組成データ（Compositional data）の表現において極めて重要である。例えば、Aitchison組成空間は、ホモロジー鎖複体（Homological chain complex）として捉えることが可能であり、単体の頂点が純粋な成分を、内部の点が混合物を表す [29]。
このような空間では、通常のガウス混合モデルやユークリッド的クラスタリングは機能せず、ディリクレ分布に基づく相関推定や、空間の閉包（Closure）、摂動（Perturbation）といった代数的構造を維持する特殊なマッピングが必要となる [29]。モジュラー領域をSynthetic Benchmarkとして設定することで、生成モデルがデータの「比率」や「離散的な格子対称性」を維持しつつ、連続的な潜在表現を獲得できるか（あるいは標準VAEのように構造を破壊してしまうか）を厳格に評価することができる。

### 【表1: 単一VAEの限界を示すSynthetic Manifoldの特性と破綻モード】

| 多様体 / 空間 (Manifold) | 固有次元 | 非自明なトポロジー・代数的特性 | 単一VAE (Euclidean Chart) における主な破綻モード |
|---|---|---|---|
| 円 (S^1) | 1 | 1つのサイクル、周期性 | 周期境界の切断（連続性の喪失と外れ値の生成） |
| 球面 (S^2) | 2 | 閉曲面、空洞 (Void) | 極点における特異点の発生、確率測度の無限集中 |
| トーラス (T^2) | 2 | 2つの独立したサイクル | 二重の周期境界の不一致、サンプルの大域的断絶 |
| メビウスの帯 | 2 | 非向き付け可能、ツイスト | ツイスト部での勾配消失、大域的法線の崩壊 |
| クラインの壺 | 2 | 非向き付け可能、境界なし | 自己交差領域での次元の潰れと致命的な幾何学的歪み |
| Lie群 SO(3) | 3 | 非可換群、回転の連続対称性 | 对称性欠陥 (Symmetry Defect)、Lie代数交換子の不一致 |
| 射影空間 (\mathbb{R}P^2) | 2 | 商空間、対蹠点の同一視 | 同一視点のユークリッド的乖離、大域的距離関係の崩壊 |
| モジュラー/格子領域 | 任意 | 離散的並進対称性、鎖複体 | 格子構造の平滑化（ぼやけ）、組成比率の破壊 |

## 4. 局所パッチ入力とAtlas Ensembleの差異を可視化するベンチマーク

データ多様体が単一の連続写像（単一のVAE）で表現できないことが明白である場合、微分幾何学的に正しいアプローチは、多様体を複数の局所的なチャート（パッチ）で被覆し、それらの重なり合う領域で「遷移写像（Transition Maps）」を定義して多様体全体を構成する「アトラス（地図帳: Atlas）」を構築することである。深層生成モデルの文脈では、この理論は Atlas Generative Models (AGMs) や Mixture of VAEs（VAEの混合モデル）として実装されている [14]。
ここで生じる極めて重要な学術的問いは、「単に多様体を局所パッチに分割して別々に学習させたモデル（Patch Input）」と、「多様体全体の微分幾何学的構造を維持するように協調して学習するアトラスモデル（Atlas Ensemble）」の差異を、どのベンチマークが最も明確に可視化できるかである。

### 4.1. Patch Input と Atlas Ensemble の決定的な違い

単なるパッチベースのアプローチ（例えば、独立した複数のAutoencoderを並べ、入力を距離に応じて割り当てる手法）は、多様体を切り刻むことはできても、パッチ間の滑らかな接続関係を学習しない。その結果、パッチの境界においてデータ分布が不連続になり、生成プロセスにおいて多様体から大きく外れた外れ値（Outliers）やアーティファクトが頻発する [32]。
一方、Mixture of VAEsやAGMのようなAtlas Ensembleは、各エンコーダ・デコーダペアが多様体の1つのチャートを担うと同時に、データ空間における1の分割（Partition of Unity）を用いて、チャート間の滑らかな遷移を学習する [30]。さらに、ある研究が指摘するように、再構成に一貫性を持つAutoencoder Atlasesは、チャート間の遷移写像がコサイクル条件（Cocycle Condition）を満たすように自律的に最適化され、多様体上の「接束（Tangent Bundle）」や第一スティーフェル・ホイットニー類（First Stiefel-Whitney Class）といった高度な微分トポロジー的不変量を構成する能力を持つ [7]。

### 4.2. 差異を明確にするベンチマークの選定

パッチとアトラスの差を劇的に見せつけるためには、多様体が単一のチャートで覆えないだけでなく、「近接しているが測地線距離は遠い」領域や、「複数の穴が干渉する」領域を持つベンチマークが必要である。

#### ① 二重トーラス（Double Torus / Genus-2 Surface）

二重トーラスは、2つの独立した穴を持つ曲面であり、正の曲率と負の曲率が複雑に交錯する領域を持つ。
 * **Patch Inputの失敗**: 単純なパッチ入力では、多様体を局所的な平面（\mathbb{R}^2）として切り取ることはできるが、パッチ同士が穴の周囲でどのように接続されるべきかの大域的情報が欠落する。その結果、あるパッチから別のパッチへ移動する際に、穴を無視してショートカットするようなアーティファクトが生成される [32]。
 * **Atlas Ensembleの成功**: Complexity Decoupled Chart Autoencodersなどのアトラスベースのモデルは、複数のチャート（例えば7つの色分けされたチャート）を用いて多様体を過不足なく被覆し、チャート同士の重なり（Overlap）を明示的に学習する。これにより、2つの穴という大域的トポロジーを忠実に保持したまま、新しい点を多様体上に正確に生成することができる [34]。
 
#### ② スイスロール（Swiss Roll）上の外因的・内因的距離の分離

スイスロール自体は固有次元2の平坦な多様体であるが、3次元空間（\mathbb{R}^3）において強く巻き込まれているため、外因的なユークリッド距離と内因的な測地線距離が大きく乖離する。
 * **Patch Inputの失敗**: 単なるパッチモデル（あるいはユークリッド距離に基づくK-means分割）では、層と層が近接している領域において、異なる層のデータポイントを同じパッチに混同してしまうリスクが高い。
 * **Atlas Ensembleの成功**: Atlas Ensembleでは、Mixture of VAEsの重み最適化において局所的な構造を維持するようリプシッツ正則化（Lipschitz regularization）などを適用することで、重なり合う層を別々のチャートとして正確に分離し、それぞれの層に沿った関数（ラベルや物理量）を正確に近似・学習することが可能となる [14]。
 
#### ③ クラインの壺の交差領域（Intersection Regions）

前述のクラインの壺を3次元空間へ射影した際などに生じる「自己交差領域」は、パッチとアトラスの違いを見る究極のストレステストである。単なる局所パッチアプローチは、交差する2つの面を1つの面として誤認し、そこでトポロジーを破壊する。しかし、適切に設計されたアトラスモデルは、潜在空間においてこれらが異なるチャートに属することを認識し、遷移写像のヤコビアンの符号から非向き付け可能性を計算することで、交差部分を論理的に分離する [7]。

### 4.3. パッチとアトラスの性能差を測る究極の評価指標：測地線補間

パッチとアトラスの差を定量的に見せるタスクとして、最も効果的なのが「潜在空間における測地線補間（Geodesic Interpolation）」である [33]。2つの離れたデータポイント間を補間する際、単一チャートや未接続のパッチモデルでは、潜在空間上の直線補間がデコーダを経由した際に多様体から大きく逸脱し、無意味な空間を通過する。
対照的にAtlas Ensembleでは、1の分割（Partition of Unity）を用いて異なるチャート間を横断する微分幾何学的に正しいパスを計算するため、生成された点が常に多様体 \mathcal{M} 上に留まる。生成された補間点が元の多様体方程式をどれだけ正確に満たしているか（測地線誤差やMSE）を計測することで、アトラス構造の圧倒的な優位性を証明できる [37]。

## 5. 既存研究におけるトポロジー評価指標とベンチマーク体系

VAEや正規化流（Normalizing Flows）のトポロジー適合性を検証するため、既存の研究パラダイムでは高度な数学的評価指標と標準化されたベンチマークセットが既に確立されている。これらの指標は、単純な再構成誤差（MSE）や生成画像の視覚的品質（FID）が、多様体のトポロジー保存を評価する上で不十分であるという認識から発展してきた。

### 5.1. 代表的なSynthetic Benchmark Suites
既存の主要な論文（例: "Manifold Learning by Mixture Models of VAEs" [14] や "Diffusion Variational Autoencoders" [10]）で共通して採用されているデータセットのセットアップは以下の通りである。

#### Diffusion VAE / Poincaré VAE 向けの標準セット [10]

 * 多様体上のブラウン運動（Brownian motion）を遷移カーネルとして用いる手法や、双曲幾何学を用いる手法の検証に用いられる。
 * データセットには、階層構造を持つノイズ付きベクトル、円上のデータ、球面データ、埋め込みトーラス（Embedded Torus）、平坦なトーラス（Flat Torus）、および射影空間（\mathbb{R}P^2）が含まれる。これにより、モデルが多様体の曲率や大域的トポロジーを学習できているかを比較する [10]。
 
#### Mixture of VAEs / Atlas VAE の標準セット [15]

 * **Two Circles (2つの円)**: 複数の非連結成分（\beta_0 = 2）を1つのモデルで表現する能力を問う。
 * **Ring (リング)**: 単一の1D多様体の検証。
 * **Sphere (球面 S^2) / Swiss Roll**: 曲率と埋め込み次元の検証。
 * **Torus (トーラス)**: 複数のチャートが必要な複雑な閉曲面の検証 [15]。
 
### 5.2. トポロジーの不一致を測る定量評価指標

#### ① パーシステントホモロジー（Persistent Homology）とBetti数

トポロジカルデータ解析（TDA）の核心であるパーシステントホモロジーは、データの連結成分、穴、空洞の数をスケールを変えながら計測する究極のトポロジー評価手法である [29]。
 * **Betti数 (\beta_k)**: \beta_0 は連結成分の数、\beta_1 は1次元の穴（ループ）の数、\beta_2 は2次元の空洞（Void）の数を表す [41]。例えば、球面 S^2 は単一の空洞を持つため (\beta_0 = 1, \beta_1 = 0, \beta_2 = 1) となり [41]、トーラス T^2 は2つのループと1つの空洞を持つため (\beta_0 = 1, \beta_1 = 2, \beta_2 = 1) である [41]。
 * **評価プロセス**: グラウンドトゥルースのデータサンプルのBetti数と、VAEの潜在空間からサンプリングされた生成データのBetti数を比較する。標準的なVAEは、ガウス事前分布の等方性により、生成データの \beta_1 や \beta_2 が0に潰れてしまう（トポロジーの崩壊）ことが多い [43]。
 * **パーシステンス図（Persistence Diagrams）と距離**: トポロジー的特徴の「出現（Birth）」と「消滅（Death）」のスケールをプロットしたパーシステンス図を作成し、元のデータと生成データ間の Bottleneck距離 または Wasserstein距離 を計算する [37]。これにより、局所的なノイズ変動に影響されない、大域的なトポロジー構造の維持度を厳密に定量化できる。
 
#### ② ヤコビアン行列式による幾何学的歪み（Distortion Score）

トポロジーの不一致領域では、モデルが連続性を保とうと無理に空間を引き伸ばしたり収縮させたりするため、多様体から潜在空間への写像のヤコビアン（Jacobian）の行列式に極端な変動が生じる [3]。
 * **評価関数**: \log(|\det J_{\Phi}|)^2 を用いて、潜在空間への写像 \Phi における局所的な体積変化を測定する [3]。
 * **意義**: このスコアが異常に高い領域は、多様体のトポロジーと潜在空間のトポロジー（ユークリッド空間）が衝突し、ネットワークが「トポロジカル・ミスマッチ」を起こしている箇所（特異点や「引き裂き」の境界）を正確に特定する指標となる。
 
#### ③ フィッシャー情報量とリプシッツ連続性バウンド

データの本来の複雑性（Intrinsic Noise Gain）と、潜在空間の無理な引き伸ばしによるアーティファクトを区別するため、エンコーダがBi-Lipschitz（双リプシッツ）連続性を持つかどうかが理論的に評価される。フィッシャー情報量のバウンドを計算することで、多様体が潜在空間で致命的に「引き裂かれて」いないかを監視する指標が提案されている [6]。

### 【表2: 既存研究におけるトポロジー評価指標と対応するSynthetic Manifold】

| 評価指標カテゴリ | 具体的なメトリクス | 測定するトポロジー的現象 | 最も効果的なベンチマーク多様体 |
|---|---|---|---|
| 代数的トポロジー | Betti数 (\beta_0, \beta_1, \beta_2) | 連結成分、サイクル、空洞の保持 | トーラス、球面、二重トーラス |
| 位相的データ解析 (TDA) | Bottleneck距離, Wasserstein距離 | パーシステンス図間の位相的差異 | あらゆるSynthetic Manifold |
| 微分幾何学的歪み | \log(\|\det J_{\Phi}\ | トポロジー不一致による過剰な伸縮 | スイスロール、クラインの壺 |
| 再構成・大域的一貫性 | 測地線誤差 (MSE on Geodesics) | チャート間遷移（Atlas）の連続性 | 球面、クラインの壺 |
| 対称性・群構造 | Symmetry Defect / Lie代数ミスマッチ | 連続変換に対する同変性の保持 | Lie群 SO(2), SO(3) |

## 6. 第二次・第三次洞察：トポロジー的制約が導く深層生成モデルの進化

上述のSynthetic Benchmarkと既存研究のデータを統合することで、表面的な性能比較を超えた、より深遠な数理的洞察が得られる。これは今後の多様体学習と深層生成モデルの設計指針を根本から変革する可能性を配している。

### 6.1. 万能近似定理の幻想と構造的障害（Structural Obstruction）

一般に、深層ニューラルネットワークは普遍近似定理（Universal Approximation Theorem）により、層を深くしパラメータを十分に増やせば、いかなる複雑な関数も近似できると広く信じられている。しかし、トポロジーの観点からはこれは幻想に過ぎない。データ多様体 \mathcal{M} が \mathbb{R}^d と同相でない場合、いかにネットワークの層を深くし、パラメータを無限大に増大させようとも、連続かつ全単射（Bijections）な大域的写像を構築することは数学的に不可能である [32]。
この制約下で無理に近似学習を進めようとすると、ネットワークの重みパラメータが発散するか、極端な過学習を引き起こし、モデルは未知のデータに対して完全に破綻する [32]。すなわち、**トポロジーの不一致は単なる「学習不足や最適化の失敗」ではなく、アーキテクチャに内在する「構造的障害（Structural Obstruction）」である**。したがって、モデルアーキテクチャの評価において、トーラスやクラインの壺のような位相的障害を明示的に備えたSynthetic Benchmarkを導入することは、単なる「追加のテスト」ではなく、モデルの根源的な数理的妥当性を測るための「必須の踏み絵」となる。

### 6.2. 非向き付け可能多様体（クラインの壺）の真の価値と被覆空間の要請

多くの研究が球面やトーラスをベンチマークとして用いているが、これらは「向き付け可能（Orientable）」な多様体である。一方、クラインの壺やメビウスの帯は非向き付け可能であり、大域的な法線ベクトルを一つに定義することができない。
この性質は、VAEの潜在空間における「不確実性（Uncertainty）」のモデリングに対して極めて重要な示唆を与える。非向き付け可能な多様体を学習させるためには、モデルは単一の潜在空間ではなく、対称性を考慮した「被覆空間（Covering Space）」を通じた再パラメータ化トリック（Reparameterization Trick）を獲得しなければならない [9]。クラインの壺を安定して学習・再構成できるモデル（KleinVAEなど）は、畳み込みニューラルネットワーク（CNN）の重み事前分布（Topological Weight Priors）の設計や、ベイズ推論における高度な帰納的バイアス（Inductive Bias）の付与において、実用的な画像認識や物理モデリングタスクにも絶大な波及効果をもたらす可能性が示唆されている [9]。

### 6.3. ブラックボックスからの脱却：ベクトル束としてのAutoencoder Atlases

Mixture of VAEsやAutoencoder Atlasesを、単なる「複数のデコーダの寄せ集めアンサンブル」として捉えるべきではない。最新の理論的進展によれば、これらのモデルがデータから学習する「チャート間の遷移写像（Transition Maps）」は、コサイクル条件を満たすように自己組織化され、多様体上の「接束（Tangent Bundle）」そのものを構成する [7]。
驚くべきことに、学習された遷移写像のヤコビアンの符号から、多様体の向き付け可能性を決定づける第一スティーフェル・ホイットニー類（First Stiefel-Whitney Class）をアルゴリズム的に計算できることが証明されている [7]。これは、**深層生成モデルが単なるブラックボックスなパターン認識器や次元削減ツールから脱却し、データの背後にある代数的トポロジーや微分幾何学的構造を自律的に抽出・証明する「計算幾何学的エンジン」へと進化しつつある**ことを意味する。ベンチマークにおいてパッチとアトラスの差異を厳密に見せることは、モデルが「データを表層的に記憶しているか」ではなく、「データの微分幾何学的真理を理解しているか」を検証することと同義である。

## 7. 結論

本報告書における包括的な分析により、深層生成モデルの研究を安定させ、トポロジー的ミスマッチを克服するためには、用途に応じた階層的なSynthetic Benchmark体系の導入が不可欠であることが明らかになった。
単一のVAE（単一のユークリッドチャート）が持つ限界を最も明確に露呈させる多様体は、特異点を誘発する**球面（S^2）**、大域的なサイクルの閉包を求める**トーラス（T^2）**、そして自己交差と被覆空間の概念を強制する究極のテストである**クラインの壺（Klein Bottle）**である。さらに、連続的な対称性や同一視を評価するためには、Lie群 SO(3) や商空間が不可欠な基準となる。
一方、局所的なパッチ入力と大域的な一貫性を持つAtlas Ensembleの性能差を可視化するためには、複数の穴が干渉する**二重トーラス**や、外因的距離と内因的距離が大きく乖離する**スイスロール**、およびチャート間の滑らかな移動を問う**測地線補間**のタスクが最適である。
これらのSynthetic Manifold의 検証には、単なる再構成誤差や視覚的指標（FID）への依存から脱却し、パーシステントホモロジー（Betti数、Wasserstein距離）や、ヤコビアンの行列式に基づく幾何学的歪みスコアといった厳格な数学的指標を適用しなければならない。パラメータ増大による力技の表現力向上には数学的な限界が存在する。今後は、多様体のトポロジーや微分幾何学的構造を事前知識やアーキテクチャ（Atlas VAE, Spherical VAE, Diffusion VAEなど）に直接組み込む「Topology-Aware」な深層学習パラダイムへの移行が、この領域における最大のブレイクスルーをもたらすであろう。

## 引用文献

[1]: STAR-VAE: Structured Topology-Aware Regularization for Audio Reconstruction and Generation - ICML 2026, 5月 25, 2026にアクセス、 https://icml.cc/virtual/2026/poster/63959

[2]: Decoder ensembling for learned latent geometries - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2408.07507v1

[3]: Spot the Difference: Detection of Topological Changes via Geometric Alignment, 5月 25, 2026にアクセス、 https://proceedings.neurips.cc/paper_files/paper/2021/file/7867d6557b82ed3b5d61e6591a2a2fd3-Paper.pdf

[4]: Universal Latent Homeomorphic Manifolds: A Framework for Cross-Domain Representation Unification - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2601.09025v2

[5]: PCGen: A Fully Parallelizable Point Cloud Generative Model - PMC, 5月 25, 2026にアクセス、 https://pmc.ncbi.nlm.nih.gov/articles/PMC10934358/

[6]: Understanding Latent Diffusability via Fisher Geometry - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2604.02751v1

[7]: Learning Tangent Bundles and Characteristic Classes with Autoencoder Atlases - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/abs/2602.22873

[8]: Novel computational and geometric frameworks for manifold learning, Riemannian optimization, and feature selection - Knowledge UChicago, 5月 25, 2026にアクセス、 https://knowledge.uchicago.edu/record/16739/files/Robinett_Doctoral_Thesis_Title_Revision.pdf

[9]: Reparameterization through Coverings and Topological Weight Priors - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2604.23804v1

[10]: Diffusion Variational Autoencoders - IJCAI, 5月 25, 2026にアクセス、 https://www.ijcai.org/proceedings/2020/0375.pdf

[11]: Beyond Gaussian Bottlenecks: Topologically Aligned Encoding of Vision-Transformer Feature Spaces - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2604.28122v1

[12]: Canonical normalizing flows for manifold learning - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2310.12743v3

[13]: arXiv:2103.01071v1 [cs.LG] 1 Mar 2021 - CSE - IIT Kanpur, 5月 25, 2026にアクセス、 https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter21/readings/vae-survey.pdf

[14]: Manifold Learning by Mixture Models of VAEs for Inverse Problems, 5月 25, 2026にアクセス、 https://jmlr.org/papers/volume25/23-0396/23-0396.pdf

[15]: (PDF) Manifold Learning by Mixture Models of VAEs for Inverse Problems - ResearchGate, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/369556391_Manifold_Learning_by_Mixture_Models_of_VAEs_for_Inverse_Problems

[16]: Manifold Dimension Estimation via Local Graph Structure - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2510.15141v4

[17]: Variational Autoencoders for Learning Nonlinear Dynamics of Physical Systems, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/346701232_Variational_Autoencoders_for_Learning_Nonlinear_Dynamics_of_Physical_Systems

[18]: GD-VAEs: Geometric Dynamic Variational Autoencoders for Learning Nonlinear Dynamics and Dimension Redumptions - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2206.05183v3

[19]: Unsupervised manifold learning using low-distortion alignment of tangent spaces | bioRxiv, 5月 25, 2026にアクセス、 https://www.biorxiv.org/content/10.1101/2024.10.31.621292v2.full-text

[20]: Dimension Estimation and Topological Manifold Learning - IEEE Xplore, 5月 25, 2026にアクセス、 https://ieeexplore.ieee.org/document/8852081/

[21]: GD-VAEs: Geometric Dynamic Variational Autoencoders for Learning Nonlinear Dynamics and Dimension Reductions - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2206.05183v4

[22]: TIOLI-GAN with Wasserstein-1 Distance and Adaptive ... - TechRxiv, 5月 25, 2026にアクセス、 https://www.techrxiv.org/users/869274/articles/1277533/master/file/data/tioliw/tioliw.pdf

[23]: Topological Convolutional Layers for Deep Learning, 5月 25, 2026にアクセス、 https://www.jmlr.org/papers/volume24/21-0073/21-0073.pdf

[24]: A Klein-Bottle-Based Dictionary for Texture Representation - ResearchGate, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/262394841_A_Klein-Bottle-Based_Dictionary_for_Texture_Representation

[25]: Canonical normalizing flows for manifold learning - NIPS papers, 5月 25, 2026にアクセス、 https://papers.nips.cc/paper_files/paper/2023/file/572a6f16ec44f794fb3e0f8a310acbc6-Paper-Conference.pdf

[26]: Embedding-reparameterization procedure for manifold-valued latent, 5月 25, 2026にアクセス、 https://bayesiandeeplearning.org/2018/papers/159.pdf

[27]: Daily Papers - Hugging Face, 5月 25, 2026にアクセス、 https://huggingface.co/papers?q=manifold%20regularization

[28]: Equivariant Transition Matrices for Explainable Deep Learning: A Lie Group Linearization Approach - MDPI, 5月 25, 2026にアクセス、 https://www.mdpi.com/2504-4990/8/4/92

[29]: Mixtures Recomposition by Neural Nets: A Multidisciplinary Overview | Journal of Chemical Information and Modeling - ACS Publications, 5月 25, 2026にアクセス、 https://pubs.acs.org/doi/10.1021/acs.jcim.3c01633

[30]: Atlas Generative Models and Geodesic Interpolation - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/pdf/2102.00264

[31]: Atlas generative models and geodesic interpolation - University of, 5月 25, 2026にアクセス、 https://researchprofiles.ku.dk/en/publications/atlas-generative-models-and-geodesic-interpolation/

[32]: UNIVERSITY OF CALIFORNIA SAN DIEGO Implications of Geometry and Topology on Deep Learning Capabilities A dissertation submitted - eScholarship.org, 5月 25, 2026にアクセス、 https://escholarship.org/content/qt2xh3d2m6/qt2xh3d2m6_noSplash_26daf7f0b7f316f27eef907903710fe6.pdf

[33]: Fast Approximate Geodesics for Deep Generative Models | Request PDF - ResearchGate, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/335699526_Fast_Approximate_Geodesics_for_Deep_Generative_Models

[34]: Semi-Supervised Manifold Learning with Complexity Decoupled Chart Autoencoders - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2208.10570v2

[35]: Semi-Supervised Manifold Learning with Complexity Decoupled Chart Autoencoders, 5月 25, 2026にアクセス、 https://www.researchgate.net/publication/362886993_Semi-Supervised_Manifold_Learning_with_Complexity_Decoupled_Chart_Autoencoders

[36]: Finding geodesics with the Deep Ritz method - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/pdf/2510.15177

[37]: 1 INTRODUCTION - arXiv, 5月 25, 2026にアクセス、 https://arxiv.org/html/2505.24665v2

[38]: Diffusion Variational Autoencoders - IJCAI, 5月 25, 2026にアクセス、 https://www.ijcai.org/proceedings/2020/0375

[39]: Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders - NIPS, 5月 25, 2026にアクセス、 http://papers.neurips.cc/paper/9420-continuous-hierarchical-representations-with-poincare-variational-auto-encoders.pdf

[40]: arXiv:2205.05279v1 [cs.LG] 11 May 2022, 5月 25, 2026にアクセス、 https://arxiv.org/pdf/2205.05279

[41]: Persistent Homology in Multivariate Data Visualization - Bastian Rieck, 5月 25, 2026にアクセス、 https://bastian.rieck.me/research/Dissertation_Rieck_2017.pdf

[42]: Persistent homology of the cosmic web – I. Hierarchical topology in \LambdaCDM cosmologies | Monthly Notices of the Royal Astronomical Society | Oxford Academic, 5月 25, 2026にアクセス、 https://academic.oup.com/mnras/article/507/2/2968/6353532

[43]: Principle Component Trees and Their Persistent Homology - OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/pdf/18f100121105457a4e6f32d0931ff3a3b2a27bed.pdf

[44]: Topology Understanding and Topology Control for 3D Models by Computational Topology: A Survey - TechRxiv, 5月 25, 2026にアクセス、 https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176462324.46233242

[45]: Spot the Difference Detection of Topological Changes via Geometric Alignment - DTU Research Database, 5月 25, 2026にアクセス、 https://orbit.dtu.dk/files/282194661/spot_the_difference_detection_.pdf

[46]: Reparameterization through Coverings and Topological Weight Priors - OpenReview, 5月 25, 2026にアクセス、 https://openreview.net/forum?id=kZ5sWPK2p8

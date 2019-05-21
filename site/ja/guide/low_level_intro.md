# イントロダクション

このガイドでは、低レベルのTensorFlow API(TensorFlow Core)でプログラミングを始めます。
以下の方法を説明します。

  * Estimatorを使用せず、TensorFlowプログラム（`tf.Graph`）とTensorFlowランタイム（`tf.Session`）を自分自身で管理する方法。
  * `tf.Session`を使ってTensorFlowオペレーションを実行する方法。
  * 低レベルの環境で高レベルのコンポーネント([datasets](＃datasets), [layers](＃layers), および[feature_columns](＃feature_columns))を使用する方法。
  * [Estimatorが提供する学習ループ](../guide/premade_estimators.md)を使用するのではなく、独自の学習ループを構築する方法。

可能であれば、より高レベルのAPIを使用してモデルを構築することをお勧めします。
TensorFlow Coreを知ることは、次のような理由で有益です。

  * 低レベルのTensorFlowオペレーションを直接使用すると、実験とデバッグがより容易になります。
  * TensorFlowの内部がどのように機能するかのメンタルモデルを得ることができ、より高レベルのAPIを使用する際に役に立ちます。

## セットアップ

このガイドを使用する前に、[TensorFlowをinstallしてください。](../install)

このガイドを最大限に活用するには、次のことを知っておく必要があります。

* Pythonでのプログラム方法
* 少なくとも配列についての知識
* 理想的には、機械学習の知識

```
[訳しにくい]
気軽に `python`を起動し、この解説の手順を追ってみてください。
```
Python環境をセットアップするには、以下のコードを実行してください。

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
```

## tensorの値

TensorFlowのデータの中心単位は**tensor**です。tensorは、不可分な一連の値のセットで構成され、
任意の次元数の配列に整形されたものです。tensorの**rank**はその次元数を表しますが、**shape**は整数のタプルであり、その配列の各次元の大きさを表しています。
tensorの値の例：

```python
3. # a rank 0 tensor; a scalar with shape [],
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

TensorFlowはtensorの値を表すのにnumpyのarrayを使用します。

## TensorFlowコアのチュートリアル

TensorFlow Coreプログラムは、2つのディスクリートからなると考えるかもしれません。
セクション：

1.  計算グラフの作成 (`tf.Graph`).
2.  計算グラフの実行 (`tf.Session`を使用する).

### Graph

**計算グラフ**は、次のように配列された一連のTensorFlow操作です。
グラフ。グラフは2種類のオブジェクトで構成されています。

  * `tf.Operation` (もしくは "ops"): グラフの節点。
    操作はテンソルを消費し生成する計算を記述します。
  * `tf.Tensor`: グラフ内の辺これらは値を表します
    それはグラフを流れます。ほとんどのTensorFlow関数は戻ります
    `tf.Tensors`。

重要： `tf.Tensors`は値を持っていません、それらは単に要素へのハンドルです
計算グラフで

簡単な計算グラフを作りましょう。最も基本的な操作は
定数。操作を構築するPython関数は次のようにテンソル値を取ります。
入力。結果の操作は入力を取りません。実行すると、
コンストラクタに渡された値。 2つの浮動小数点を作成できます
以下の定数 `a`と` b`：

```python
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
```

printステートメントは次のものを生成します。

```
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("add:0", shape=(), dtype=float32)
```

テンソルを表示しても値 `3.0`、` 4.0`、
ご想像のとおり `7.0`。上記のステートメントは計算を構築するだけです
グラフ。これらの `tf.Tensor`オブジェクトは単に操作の結果を表しています
それが実行されます。

グラフ内の各操作には一意の名前が付けられています。この名前は独立しています
オブジェクトがPythonで割り当てられている名前。テンソルの名前は
次のように、それらに続いて出力インデックスを生成する操作
上記の `" add：0 "`。

### TensorBoard

TensorFlowはTensorBoardというユーティリティを提供します。 TensorBoardの多数のうちの1つ
capabilitiesは計算グラフを視覚化しています。あなたは簡単にこれを行うことができます
いくつかの簡単なコマンド。

まず、計算グラフをTensorBoard要約ファイルに次のように保存します。
次のとおりです。

```
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()
```

これはカレントディレクトリに `event`ファイルを作成します。
次の形式

```
events.out.tfevents.{timestamp}.{hostname}
```

それでは、新しい端末で、次のシェルコマンドを使ってTensorBoardを起動します。

```bsh
tensorboard --logdir .
```

それからTensorBoardの[graph page]（http：// localhost：6006 /＃graph）を開いてください。
ブラウザを起動すると、次のようなグラフが表示されます。

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_add.png)

TensorBoardのグラフ可視化ツールの詳細については、[TensorBoard：Graph Visualization]（../ guide / graph_viz.md）を参照してください。

### Session

テンソルを評価するには、 `tf.Session`オブジェクトをインスタンス化します。
**セッション**。セッションはTensorFlowランタイムの状態をカプセル化します。
TensorFlowオペレーションを実行します。 `tf.Graph`が` .py`ファイルのようなものであれば、 `tf.Session`
`python`実行ファイルに似ています。

次のコードは `tf.Session`オブジェクトを作成してからその` run`を呼び出します
上で作成した `total`テンソルを評価するメソッド

```python
sess = tf.Session()
print(sess.run(total))
```

`Session.run` TensorFlowバックトラックでノードの出力を要求したとき
グラフを通して、要求されたノードに入力を提供するすべてのノードを実行します。
出力ノードしたがって、これは期待値7.0を出力します。

```
7.0
```

複数のテンソルを `tf.Session.run`に渡すことができます。 `run`メソッド
次のように、タプルまたは辞書の任意の組み合わせを透過的に処理します。
次の例

```python
print(sess.run({'ab':(a, b), 'total':total}))
```

これは同じレイアウトの構造で結果を返します。

<pre>
{'total': 7.0, 'ab': (3.0, 4.0)}
</pre>

`tf.Session.run`の呼び出し中は、` tf.Tensor`は単一の値しか持ちません。
例えば、次のコードは `tf.random_uniform`を呼び出して
ランダムな3要素ベクトル（ `[0,1）`の値を持つ）を生成する `tf.Tensor`：

```python
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))
```

結果は `run`の呼び出しごとに異なる乱数を示していますが、
単一の `run`の間の一貫した値（` out1`と `out2`は同じものを受け取ります
ランダム入力）：

```
[ 0.52917576  0.64076328  0.68353939]
[ 0.66192627  0.89126778  0.06254101]
(
  array([ 1.88408756,  1.87149239,  1.84057522], dtype=float32),
  array([ 2.88408756,  2.87149239,  2.84057522], dtype=float32)
)
```

いくつかのTensorFlow関数は、 `tf.Tensors`の代わりに` tf.Operations`を返します。
Operationで `run`を呼び出した結果は` None`です。操作を実行した
値を取得するのではなく、副作用を引き起こすため。この例としては、
[初期化]（＃初期化レイヤ）、および[トレーニング]（＃トレーニング）の操作
後で実証した。

### Feeding

それが立っているように、このグラフは常にそれがあるので特に面白くない
一定の結果が得られます。グラフは外部を受け入れるようにパラメータ化することができます
**プレースホルダー**として知られる入力。 **プレースホルダー**は、を提供するという約束です。
関数の引数のように、あとで値

```python
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
```

上記の3行は、次のような関数に少し似ています。
2つの入力パラメータ（ `x`と` y`）を定義してからそれらに対する操作を定義します。私たちはできる
の `feed_dict`引数を使ってこのグラフを複数の入力で評価する
具体的な値をプレースホルダーに渡すための `tf.Session.run`メソッド

```python
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
```

これにより、次のような出力が得られます。

```
7.5
[ 3.  7.]
```

また、 `feed_dict`引数を使ってテンソルを上書きすることができます。
グラフプレースホルダと他の `tf.Tensors`の唯一の違いは
値が与えられない場合、そのプレースホルダはエラーをスローします。

## Datasets

プレースホルダーは簡単な実験のために働きますが、 `tf.data`は
データをモデルにストリーミングするための好ましい方法。

データセットから実行可能な `tf.Tensor`を取得するには、まずそれをに変換しなければなりません。
`tf.data.Iterator`を呼び出してから、Iteratorを呼び出します。
`tf.data.Iterator.get_next`メソッド

イテレータを作成する最も簡単な方法は、
`tf.data.Dataset.make_one_shot_iterator`メソッド
例えば、次のコードでは `next_item`テンソルはから行を返します。
`run`を呼び出すたびに` my_data`配列

``` python
my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()
```

データストリームの終わりに到達すると `Dataset`は
`tf.errors.OutOfRangeError`。例えば、以下のコード
読み込むデータがなくなるまで `next_item`を読み込みます。

``` python
while True:
  try:
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break
```

`Dataset`がステートフルな操作に依存している場合はあなたがする必要があるかもしれません
次に示すように、使用する前にイテレータを初期化します。

``` python
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break
```

データセットとイテレータの詳細については、[データのインポート]（../ guide / datasets.md）を参照してください。

## Layers

トレーニング可能なモデルは、グラフ内の値を修正して、次のようにして新しい出力を取得します
同じ入力`tf.layers`はトレーニング可能なものを追加するのに好ましい方法です
グラフへのパラメータ。

レイヤーは、動作する変数と操作の両方をまとめたものです。
それらの上に。例えば、
[密接に接続された層]（https://developers.google.com/machine-learning/glossary/#fully_connected_layer）
すべての入力にわたって加重合計を実行します
各出力に対して、オプションを適用
[アクティベーション機能]（https://developers.google.com/machine-learning/glossary/#activation_function）
接続ウェイトとバイアスは、レイヤオブジェクトによって管理されます。

### Creating Layers

次のコードは、 `tf.layers.Dense`レイヤーを作成します。
入力ベクトルのバッチを作成し、それぞれに対して単一の出力値を生成します。を適用する
入力にレイヤを追加するには、レイヤを関数のように呼び出します。例えば：

```python
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
```

レイヤは入力を調べて、内部変数のサイズを決定します。そう
ここで、レイヤができるように `x`プレースホルダの形状を設定しなければなりません。
正しいサイズの重み行列を作成します。

出力の計算 `y`を定義したので、もう1つあります。
詳細計算を実行する前に注意が必要です。

### Initializing Layers

レイヤーには、初期化する前に**初期化**する必要がある変数が含まれています。
中古。変数を個別に初期化することは可能ですが、簡単にできます。
TensorFlowグラフのすべての変数を次のように初期化します。

```python
init = tf.global_variables_initializer()
sess.run(init)
```

重要： `tf.global_variables_initializer`だけを呼び出す
TensorFlowオペレーションへのハンドルを作成して返します。あのop
`tf.Session.run`で実行すると、すべてのグローバル変数を初期化します。

また、この `global_variables_initializer`は変数を初期化するだけです
これは、初期化子が作成されたときにグラフに存在していました。イニシャライザ
グラフ作成中に追加される最後のものの1つであるべきです。

### Executing Layers

レイヤが初期化されたので、 `linear_model`の出力を評価できます。
他のテンソルと同じようにテンソル。たとえば、次のようなコードです。

```python
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
```

次のような2要素の出力ベクトルを生成します。

```
[[-3.41378999]
 [-9.14999008]]
```

### Layer Function shortcuts

（ `tf.layers.Dense`のような）各層クラスに対して、TensorFlowはまた、
ショートカット関数（ `tf.layers.dense`のようなもの）。唯一の違いは、
ショートカット機能バージョンは、1回の呼び出しでレイヤを作成して実行します。にとって
たとえば、次のコードは以前のバージョンと同じです。

```python
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
```

便利ですが、この方法では `tf.layers.Layer`にアクセスできません
オブジェクトこれは、イントロスペクションとデバッグをより困難にします。
そしてレイヤーの再利用は不可能です。

## Feature columns

フィーチャ列を試す最も簡単な方法は、
`tf.feature_column.input_layer`関数この関数は
[密な列]（../ guide / feature_columns.md）を入力として、結果を表示する
カテゴリカル列の場合は、
`tf.feature_column.indicator_column`。例えば：

``` python
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)
```

`input`テンソルを実行することは` features`をベクトルのバッチに分解します。

フィーチャ列はレイヤのように内部状態を持つことができるので、それらはしばしば存在する必要があります。
初期化されました。カテゴリカルカラムは `tf.contrib.lookup`を使います
内部的にこれらは別々の初期化操作を必要とします、
`tf.tables_initializer`。

``` python
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))
```

内部状態が初期化されたら、他のように `input`を実行できます。
他の `tf.Tensor`：

```python
print(sess.run(inputs))
```

これは、フィーチャ列がどのように入力ベクトルをパックしたかを示します。
最初の2つのインデックスとして1ホット "部門"と3番目のインデックスとして "売上高"。

<pre>
[[  1.   0.   5.]
 [  1.   0.  10.]
 [  0.   1.   8.]
 [  0.   1.   9.]]
</pre>

## Training

これで、TensorFlowコアの基本についての知識が深まりました。
手動での小さな回帰モデル

### Define the data

最初にいくつかの入力 `x`と各入力に期待される出力を定義しましょう。
`y_true`：

```python
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
```

### Define the model

次に、1出力の単純な線形モデルを作成します。

``` python
linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
```

次のように予測を評価できます。

``` python
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))
```

モデルはまだ訓練されていないので、4つの「予測」値はあまりよくありません
良い。これが私たちが得たものです。あなた自身の出力はほぼ確実に異なります。

<pre>
[[ 0.02631879]
 [ 0.05263758]
 [ 0.07895637]
 [ 0.10527515]]
</pre>

### Loss

モデルを最適化するには、まず損失を定義する必要があります。平均値を使います
二乗誤差、回帰問題に対する標準的な損失。

低レベルの数学演算でこれを手動で行うことができますが、
`tf.losses`モジュールは一般的な損失関数のセットを提供します。あなたはそれを使うことができます
次のように平均二乗誤差を計算します。

``` python
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

print(sess.run(loss))
```
これは損失値を生み出します。

<pre>
2.23962
</pre>

### Training

TensorFlowは提供します
[**オプティマイザー**]（https://developers.google.com/machine-learning/glossary/#optimizer）
標準の最適化アルゴリズムを実装する。これらは以下のように実装されています。
`tf.train.Optimizer`のサブクラス。それらはそれぞれ徐々に変化する
損失を最小限に抑えるために可変です。最も簡単な最適化アルゴリズムは、
[**グラデーション降下**]（https://developers.google.com/machine-learning/glossary/#gradient_descent）、
`tf.train.GradientDescentOptimizer`によって実装されています。それはそれぞれ修正します
に関する損失の導関数の大きさに応じて可変
その変数例えば：

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

このコードは、最適化に必要なすべてのグラフコンポーネントを構築します。
トレーニング操作を返します。実行されると、トレーニングopは変数を更新します
グラフ内。あなたはそれを次のように実行するかもしれません：

```python
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)
```

`train`はテンソルではなく演算子なので、実行時に値を返しません。
訓練中の損失の進行を見るために、損失テンソルをつぎのように実行します。
同時に、次のような出力が生成されます。

<pre>
1.35659
1.00412
0.759167
0.588829
0.470264
0.387626
0.329918
0.289511
0.261112
0.241046
...
</pre>

### Complete program

```python
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
```

## Next steps

TensorFlowでモデルを構築することについてもっと学ぶために、以下を考えてください：

* [カスタム見積もりツール]（../ guide / custom_estimators.md）、ビルド方法を学ぶ
  TensorFlowによるカスタマイズモデルTensorFlow Coreに関するあなたの知識は
  自分のモデルを理解しデバッグするのに役立ちます。

TensorFlowの内部動作についてもっと知りたい場合は、
以下の文書は、議論されたトピックの多くについてさらに深く掘り下げます。
ここに：

* [Graphs and Sessions](../guide/graphs.md)
* [Tensors](../guide/tensors.md)
* [Variables](../guide/variables.md)

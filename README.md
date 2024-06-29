# PyTorch Template Project
PyTorch deep learning project made easy.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [PyTorch Template Project](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss](#loss)
		* [metrics](#metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
	* [Contribution](#contribution)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features
* 多くのディープラーニングプロジェクトに適したクリアなフォルダ構造。
* パラメータ調整を容易にするための`.json`設定ファイルサポート。
* パラメータ調整をより便利にするためのカスタマイズ可能なコマンドラインオプション。
* チェックポイントの保存と再開。
* 高速開発のための抽象ベースクラス：
  * `BaseTrainer`はチェックポイントの保存/再開、トレーニングプロセスのログ記録などを扱います。
  * `BaseDataLoader`はバッチ生成、データシャッフル、バリデーションデータの分割を扱います。
  * `BaseModel`は基本的なモデル要約を提供します。

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - トレーニングを開始するメインスクリプト
  ├── test.py - 訓練されたモデルの評価
  │
  ├── config.json - トレーニングの設定を保持
  ├── parse_config.py - 設定ファイルとCLIオプションを扱うクラス
  │
  ├── new_project.py - テンプレートファイルで新しいプロジェクトを初期化
  │
  ├── base/ - 抽象ベースクラス
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - データローダに関するすべてがここに含まれる
  │   └── data_loaders.py
  │
  ├── data/ - 入力データを保存するデフォルトディレクトリ
  │
  ├── model/ - モデル，損失，メトリクス
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - 訓練されたモデルがここに保存される
  │   └── log/ - tensorboardとログ出力のためのデフォルトログディレクトリ
  │
  ├── trainer/ - トレーナー
  │   └── trainer.py
  │
  ├── logger/ - tensorboardの可視化とログ記録のためのモジュール
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - 小さなユーティリティ
      ├── util.py
      └── ...
  ```

## Usage
このリポジトリのコードはMNISTデータに対するテンプレートです．
コードを実行するには、`python train.py -c config.json`を試してください。

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Mnist_LeNet",        // トレーニングセッション名
  "n_gpu": 1,                   // トレーニングに使用するGPUの数
  
  "arch": {
    "type": "MnistModel",       // トレーニングするモデルアーキテクチャの名前
    "args": {

    }                
  },
  "data_loader": {
    "type": "MnistDataLoader",         // データローダーの選択
    "args":{
      "data_dir": "data/",             // データセットのパス
      "batch_size": 64,                // バッチサイズ
      "shuffle": true,                 // トレーニングデータを分割前にシャッフル
      "validation_split": 0.1          // バリデーションデータセットのサイズ。float(比率)またはint(サンプル数)
      "num_workers": 2,                // データローディングに使用するCPUプロセスの数
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // 学習率
      "weight_decay": 0,               // 重み減衰
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // 損失
  "metrics": [
    "accuracy", "top_k_acc"            // 評価に使用するメトリクスのリスト
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // 学習率のスケジューラ
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // トレーニングのエポック数
    "save_dir": "saved/",              // チェックポイントはsave_dir/models/nameに保存される
    "save_freq": 1,                    // save_freqエポックごとにチェックポイントを保存
    "verbosity": 2,                    // 0: 静か, 1: per エポックごと, 2: 詳細
  
    "monitor": "min val_loss"          // モデルパフォーマンスの監視のモードとメトリック。無効にするには'off'を設定。
    "early_stop": 10	                 // 早期停止する前に待つエポック数。0に設定すると無効になります。
  
    "tensorboard": true,               // tensorboardの可視化を有効にする
  }
}
```

設定が必要な場合は、追加の設定を追加してください。

### Using config files
`.json` を変更してから実行します

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
以前に保存されたチェックポイントから再開することができます:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
設定ファイルの`n_gpu`引数を大きい数値に設定することで，マルチGPUトレーニングを有効にできます．使用可能なGPUの数よりも少ない数を設定するとデフォルトでnデバイスが使用されます．使用可能なGPUnおインデックスを環境変数によって指定します．
  ```
  python train.py --device 2,3 -c config.json
  ```
  これは以下と同等です．
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization

### Project initialization
`new_project.py` を使用して，テンプレートファイルで新しいプロジェクトディレクトリを作成します．
`python new_project.py ../NewProject` を実行すると"NewProject"という名前の新しいプロジェクトフォルダが作成されます．このスクリプトはキャッシュ，gitファイル，readmeファイルなどの不要なファイルをフィルタリングします．

### Custom CLI options

設定ファイルの値を変更することは、ハイパーパラメータを調整するためのクリーンで安全かつ簡単な方法です。しかし、時にはいくつかの値を頻繁にまたは迅速に変更する必要がある場合、コマンドラインオプションを持つ方がよいこともあります。

このテンプレートはデフォルトでjsonファイルに格納された設定を使用しますが、以下のようにカスタムオプションを登録することで、CLIフラグを使用していくつかの設定を変更することができます。

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` 引数は、設定辞書内でそのオプションにアクセスするために使用されるキーのシーケンスである必要があります。
この例では、学習率オプションのtargetは`('optimizer', 'args', 'lr')`です。これは`config['optimizer']['args']['lr']`が学習率を指すためです。
`python train.py -c config.json --bs 256`は`config.json`で指定されたオプションでトレーニングを実行しますが、`batch size`はコマンドラインオプションによって256に増やされます。

### Data Loader
* **独自のデータローダを作成する**

1. **```BaseDataLoader```を継承する**

    `BaseDataLoader` は `torch.utils.data.DataLoader`のサブクラスでいずれかを使用できます．

    `BaseDataLoader` が扱うこと:
    * 次のバッチの生成
    * データのシャッフル
    * `BaseDataLoader.split_validation()`を呼び出すことでバリデーションデータローダーを生成

* **DataLoader Usage**

  `BaseDataLoader` はバッチを繰り返し処理するためのイテレータです:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  MNISTデータ読み込みの例については `data_loader/data_loaders.py`を参照してください

### Trainer
* **独自のトレーナーを作成する**

1. **```BaseTrainer```を継承する**

    `BaseTrainer`は以下の処理をします:
    * トレーニングプロセスのログ記録
    * チェックポイントの保存
    * チェックポイントの再開
    * モデルのパフォーマンス監視機能を再構成可能で、現在のベストモデルを保存し、早期停止トレーニングを行います。
      * 設定で `monitor` が `max val_accuracy` に設定されている場合、エポックの `validation accuracy` が現在の `maximum` を置き換えるときにトレーナーはチェックポイント `model_best.pth` を保存します。
      * 設定で `early_stop` が設定されている場合、モデルのパフォーマンスが指定されたエポック数改善しないとトレーニングが自動的に終了します。この機能は `early_stop` オプションに0を渡すか、設定の行を削除することでオフにできます。

2. **抽象メソッドを実装する**

    トレーニングプロセスのために `_train_epoch()` を実装する必要があります。検証が必要な場合は、 `trainer/trainer.py` にあるように `_valid_epoch()` を実装することができます。

* **Example**

  MNISTトレーニングについては `trainer/trainer.py` を参照してください。

* **Iteration-based training**

  `Trainer.__init__`はオプションの引数 `len_epoch` を取り、各エポックのバッチ（ステップ）数を制御します。

### Model
* **Writing your own model**

1. **`BaseModel`を継承する**

    `BaseModel`は以下を処理します:
    * `torch.nn.Module` から継承
    * `__str__`: ネイティブの `print` 関数を修正して、トレーニング可能なパラメータの数を表示します。

2. **抽象メソッドを実装する**

    フォワードパスメソッド `forward()` を実装する

* **Example**

  LeNetの例については `model/model.py`を参照してください。

### Loss
カスタム損失関数は `model/loss.py` に実装できます。設定ファイルの loss に対応する名前を変更して使用します。

### Metrics
メトリクス関数は `model/metric.py` にあります

設定ファイルでリストを提供することで複数のメトリクスを監視できます。例：
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```

### Additional logging
トレーナークラスの `_train_epoch()` で追加の情報をログに記録する場合は、以下のようにして `log` にマージしてから返します：

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
訓練済みモデルをテストするには、訓練済みチェックポイントへのパスを `--resume` 引数で渡して `test.py` を実行します。

### Validation data
データローダーから検証データを分割するには、`BaseDataLoader.split_validation()` を呼び出します。すると、設定ファイルで指定されたサイズの検証用データローダーが返されます。
`validation_split` は、検証セットの全データに対する比率（0.0 <= float < 1.0）またはサンプル数（0 <= int < n_total_samples）にできます。

**Note**: `split_validation()` メソッドは元のデータローダーを変更します
**Note**: "validation_split" が 0 に設定されている場合、`split_validation()` は None を返します

### Checkpoints
設定ファイルでトレーニングセッションの名前を指定できます：
  ```json
  "name": "MNIST_LeNet",
  ```

チェックポイントは `save_dir/name/timestamp/checkpoint_epoch_n` に `mmdd_HHMMSS` 形式のタイムスタンプで保存されます。

設定ファイルのコピーも同じフォルダに保存されます。

**Note**: チェックポイントには以下が含まれます：
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard Visualization
このテンプレートは`torch.utils.tensorboard` または [TensorboardX](https://github.com/lanpa/tensorboardX)を使用してTensorboardの可視化をサポートします

1. **Install**

    pytorch 1.1 以降を使用している場合は pip install tensorboard>=1.14.0 でtensorboardをインストールします。

    それ以外の場合はtensorboardxをインストールします．[TensorboardX](https://github.com/lanpa/tensorboardX)のインストールガイドに従います．

2. **Run training** 

    設定ファイルで `tensorboard` オプションがオンになっていることを確認します。

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    プロジェクトルートで `tensorboard --logdir saved/log/` と入力すると、サーバーが `http://localhost:6006` で開きます

デフォルトでは、設定ファイルに指定された損失とメトリクスの値、入力画像、およびモデルパラメータのヒストグラムがログに記録されます。
より多くの可視化が必要な場合は、`trainer._train_epoch` メソッドで `add_scalar('tag', data)`, `add_image('tag', image)` などを使用します。
このテンプレートの `add_something()` メソッドは基本的に `tensorboardX.SummaryWriter` および `torch.utils.tensorboard.SummaryWriter` モジュールのそれらのラッパーです。

**Note**: 現在のステップを指定する必要はありません。`logger/visualization.py` で定義された `WriterTensorboard` クラスが現在のステップを追跡します。

## Acknowledgements
このプロジェクトは[pytorch-template](https://github.com/victoresque/pytorch-template) のプロジェクト [victoresque](https://github.com/victoresque)に触発されています．

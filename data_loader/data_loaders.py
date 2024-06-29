from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNISTデータセットを読み込むためのデモクラスでBaseDataLoaderを使用
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # 画像の前処理設定
        trsfm = transforms.Compose([
            transforms.ToTensor(), # 画像をテンソルに変換
            transforms.Normalize((0.1307,), (0.3081,)) # 画像を正規化
        ])

        # データが保存されているディレクトリのパス
        self.data_dir = data_dir
        # MNISTデータセットを読み込みトレーニングデータかテストデータかを指定
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        # BaseDataLoaderのコンストラクタを呼び出し，データセットとその他のパラメータを渡す
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class DirectoryDataLoader(BaseDataLoader):
    """
    ディレクトリごとに分けられたデータセットをロードするクラス
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, image_size=(224, 224)):
        # 画像の前処理設定
        trsfm = transforms.Compose([
            transforms.Resize(image_size),  # 画像サイズのリサイズ
            transforms.ToTensor(),          # 画像をテンソルに変換
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNetの平均と標準偏差に基づく正規化
        ])
        
        # ImageFolderを使用してデータセットを読み込み
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir, transform=trsfm)
        
        # BaseDataLoaderのコンストラクタを呼び出し、データセットとその他のパラメータを渡す
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

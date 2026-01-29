import torch
import torchvision
import torchvision.transforms as T

from PIL import Image
from typing import Optional, List, Union
from pathlib import Path

import logging
from accelerate.logging import get_logger

logger = get_logger(__file__)


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root: Union[str, Path], train: bool = True):
        if isinstance(root, str):
            root = Path(root)
        root = root / ("train" if train else "eval")
        assert root.is_dir()
        self.paths = list(root.glob("**/*.npy"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.from_numpy(np.load(self.paths[idx])).long()


class FFHQDataset(torch.utils.data.Dataset):
    TRAIN_SPLIT_SIZE = 65_000

    def __init__(self, root: Union[str, Path] = "data/ffhq1024", train: bool = True, transform: T = None):
        if isinstance(root, str):
            root = Path(root)

        assert root.is_dir()
        paths = list(root.glob("**/*.png"))

        if train:
            self.paths = paths[: FFHQDataset.TRAIN_SPLIT_SIZE]
        else:
            self.paths = paths[FFHQDataset.TRAIN_SPLIT_SIZE :]

        self.transform = transform if transform else T.Compose()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.transform(img)


def build_transforms(cfg):
    train_transforms = [T.Resize((cfg.data.height, cfg.data.width)), T.ToTensor()]
    test_transforms = [T.Resize((cfg.data.height, cfg.data.width)), T.ToTensor()]

    if cfg.data.preprocess.vflip:
        train_transforms.append(T.RandomVerticalFlip())
    if cfg.data.preprocess.hflip:
        train_transforms.append(T.RandomHorizontalFlip())
    if cfg.data.preprocess.normalise:
        assert len(cfg.data.preprocess.normalise.mean) == cfg.data.channels
        assert len(cfg.data.preprocess.normalise.std) == cfg.data.channels
        train_transforms.append(T.Normalize(**cfg.data.preprocess.normalise))
        test_transforms.append(T.Normalize(**cfg.data.preprocess.normalise))

    train_transforms, test_transforms = T.Compose(train_transforms), T.Compose(test_transforms)

    return train_transforms, test_transforms


def get_dataset(cfg):
    train_transforms, test_transforms = build_transforms(cfg)
    if cfg.data.name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10("data", train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10("data", train=False, transform=test_transforms, download=True)
    elif cfg.data.name == "mnist":
        train_dataset = torchvision.datasets.MNIST("data", train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.MNIST("data", train=False, transform=test_transforms, download=True)
    elif cfg.data.name in ["ffhq1024", "ffhq256", "ffhq128"]:
        train_dataset = FFHQDataset(f"data/{cfg.data.name}", train=True, transform=train_transforms)
        test_dataset = FFHQDataset(f"data/{cfg.data.name}", train=False, transform=test_transforms)
    elif cfg.data.name in ["lhq1024", "lhq256", "lhq128"]:  # https://github.com/universome/alis/blob/master/lhq.md
        raise NotImplementedError
    elif cfg.data.name in ["afhq512", "afhq256", "afhq128"]:  # https://paperswithcode.com/dataset/afhq
        train_dataset = FFHQDataset(f"data/{cfg.data.name}", train=True, transform=train_transforms)
        test_dataset = FFHQDataset(f"data/{cfg.data.name}", train=False, transform=test_transforms)
    elif cfg.data.name in ["celeba1024", "celeba256", "celeba128"]:  # https://paperswithcode.com/dataset/celeba-hq
        raise NotImplementedError
    else:
        logging.error(f"Unknown dataset {cfg.data.name}. Terminating")
        exit()

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    batch_size = cfg.vqvae.training.batch_size
    workers = cfg.data.num_workers

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)

    return train_loader, test_loader


import torch
import torchvision
from abc import ABC, abstractmethod
from typing import Tuple, Any

# ---------------------------------------------------------
# 1. 戦略のインターフェース定義
# ---------------------------------------------------------
class DatasetStrategy(ABC):
    """データセット取得ロジックの基底クラス"""
    @abstractmethod
    def get_datasets(self, cfg, train_transforms, test_transforms) -> Tuple[Any, Any]:
        pass

# ---------------------------------------------------------
# 2. 具体的な戦略の実装 (Concrete Strategies)
# ---------------------------------------------------------

class TorchVisionStrategy(DatasetStrategy):
    """CIFAR10やMNISTなど、Torchvision標準のデータセット用"""
    def __init__(self, dataset_cls, root="data"):
        self.dataset_cls = dataset_cls
        self.root = root

    def get_datasets(self, cfg, train_transforms, test_transforms):
        train_ds = self.dataset_cls(self.root, train=True, transform=train_transforms, download=True)
        test_ds = self.dataset_cls(self.root, train=False, transform=test_transforms, download=True)
        return train_ds, test_ds

class FFHQStyleStrategy(DatasetStrategy):
    """FFHQ, AFHQなど、フォルダパスベースのデータセット用"""
    def __init__(self, dataset_cls):
        self.dataset_cls = dataset_cls

    def get_datasets(self, cfg, train_transforms, test_transforms):
        # パス構築ロジック: data/{cfg.data.name}
        data_path = f"data/{cfg.data.name}"
        train_ds = self.dataset_cls(data_path, train=True, transform=train_transforms)
        test_ds = self.dataset_cls(data_path, train=False, transform=test_transforms)
        return train_ds, test_ds

class NotImplementedStrategy(DatasetStrategy):
    """まだ実装されていないデータセット用（LHQ, CelebAなど）"""
    def get_datasets(self, cfg, train_transforms, test_transforms):
        raise NotImplementedError(f"Dataset {cfg.data.name} is not implemented yet.")

# ---------------------------------------------------------
# 3. Registry (設定名と戦略の紐付け)
# ---------------------------------------------------------
def get_dataset_registry():
    # FFHQDatasetがこのファイル内でimportされている前提です
    # もし未定義ならダミーなどを入れるか、importが必要です
    
    return {
        # Torchvision Datasets
        "cifar10": TorchVisionStrategy(torchvision.datasets.CIFAR10),
        "mnist":   TorchVisionStrategy(torchvision.datasets.MNIST),
        
        # FFHQ Datasets
        "ffhq1024": FFHQStyleStrategy(FFHQDataset),
        "ffhq256":  FFHQStyleStrategy(FFHQDataset),
        "ffhq128":  FFHQStyleStrategy(FFHQDataset),
        
        # AFHQ Datasets
        "afhq512":  FFHQStyleStrategy(FFHQDataset),
        "afhq256":  FFHQStyleStrategy(FFHQDataset),
        "afhq128":  FFHQStyleStrategy(FFHQDataset),

        "custom_folder": ImageFolderSplitStrategy(), 
        "my_dataset": ImageFolderSplitStrategy(),
        
        # Not Implemented
        "lhq1024":    NotImplementedStrategy(),
        "lhq256":     NotImplementedStrategy(),
        "lhq128":     NotImplementedStrategy(),
        "celeba1024": NotImplementedStrategy(),
        "celeba256":  NotImplementedStrategy(),
        "celeba128":  NotImplementedStrategy(),
    }

# ---------------------------------------------------------
# 4. メイン関数 (Context)
# ---------------------------------------------------------
def build_dataloaders(cfg):
    train_transforms, test_transforms = build_transforms(cfg)
    
    dataset_name = cfg.data.name
    registry = get_dataset_registry()
    
    if dataset_name not in registry:
        logging.error(f"Unknown dataset {dataset_name}. Terminating")
        exit()

    strategy = registry[dataset_name]
    
    try:
        train_dataset, test_dataset = strategy.get_datasets(cfg, train_transforms, test_transforms)
    except NotImplementedError as e:
        # NotImplementedの場合は明示的にエラーログを出して終了する場合
        logging.error(str(e))
        exit()

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    batch_size = cfg.vqvae.training.batch_size
    workers = cfg.data.num_workers

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=workers, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=workers, 
        shuffle=False
    )

    return train_loader, test_loader

from torch.utils.data import Subset, random_split

class ImageFolderSplitStrategy(DatasetStrategy):
    """
    指定されたフォルダ(ImageFolder)を読み込み、
    設定された比率でTrain/Valに分割して返す戦略
    """
    def get_datasets(self, cfg, train_transforms, test_transforms):
        # 1. データディレクトリの取得 (cfg.data.root を使用すると仮定)
        # もし cfg.data.name でパスが決まるなら f"data/{cfg.data.name}" でもOK
        data_dir = getattr(cfg.data, "root", f"data/{cfg.data.root}")
        
        # 注: random_splitを使う場合、Datasetオブジェクトは1つなので
        # ここでは train_transforms を適用します（Val用にも同じ変換が適用されます）
        # ※厳密にTrain/ValでTransformを変えたい場合は別途ラッパーが必要ですが、
        #   今回は元のidx_dataloadersのロジックを優先します。
        full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transforms)

        logging.info(f"クラス情報: {full_dataset.class_to_idx}")
        logging.info(f"元の合計画像数: {len(full_dataset)}")

        # 2. サンプリング処理 (cfg.data.sampling_rate がある場合)
        sampling_rate = getattr(cfg.data, "sampling_rate", None)
        dataset_to_split = full_dataset

        if sampling_rate is not None:
            if not (0.0 < sampling_rate <= 1.0):
                raise ValueError("Sampling_rateは0.0より大きく1.0以下の値でなければなりません。")
            
            num_samples = int(len(full_dataset) * sampling_rate)
            
            # 再現性のためシード固定
            g = torch.Generator()
            g.manual_seed(42)
            # randpermでインデックス生成
            indices = torch.randperm(len(full_dataset), generator=g)[:num_samples]
            
            sampled_dataset = Subset(full_dataset, indices)
            
            logging.info(f"サンプリング適用後 ({sampling_rate * 100}%)")
            logging.info(f" -> サンプリング後の合計画像数: {len(sampled_dataset)}")
            dataset_to_split = sampled_dataset

        # 3. データセットを訓練用と検証用に分割
        split_ratio = getattr(cfg.data, "train_val_split", 0.8)
        train_size = int(split_ratio * len(dataset_to_split))
        val_size = len(dataset_to_split) - train_size

        # 再現性確保のためのシード固定
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset_to_split, [train_size, val_size], generator=generator)

        logging.info(f"訓練データ数: {len(train_dataset)}")
        logging.info(f"検証データ数: {len(val_dataset)}")

        return train_dataset, val_dataset
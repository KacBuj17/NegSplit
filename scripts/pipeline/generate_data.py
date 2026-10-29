from backbone import FrozenCLIP
from config import CACHE, RAW, TRAIN_DATA, VAL_DATA
from utils.data import ARCHIVES, download_archive, download_files, download_negbench, encode_train
from utils.pairs import build, load_coco


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    download_negbench()
    for url, marker in ARCHIVES.items():
        download_archive(url, marker)
    download_files()

    encode_train(FrozenCLIP())

    coco = load_coco()
    for path, val_templates in ((TRAIN_DATA, False), (VAL_DATA, True)):
        df = build(coco, val_templates)
        df.to_parquet(path, index=False)
        print(f"{path.name}: {len(df)} rows")


if __name__ == "__main__":
    main()

import argparse

from peta.tasks.cv import get_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../data")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["GTSRB", "SVHN"],
        choices=["Cars", "DTD", "GTSRB", "RESISC45", "SVHN"],
        help="GTSRB and SVHN download automatically; other tasks validate an existing manual layout.",
    )
    args = parser.parse_args()

    for task in args.tasks:
        print(f"Preparing/validating {task} in {args.data_root}")
        try:
            get_dataset(task + "Val", None, location=args.data_root, batch_size=1)
            get_dataset(task, None, location=args.data_root, batch_size=1)
        except (FileNotFoundError, RuntimeError) as exc:
            raise RuntimeError(
                f"{task} is not ready under {args.data_root}. "
                "See image_classification/README.md for the required layout."
            ) from exc
        print(f"{task}: ready")


if __name__ == "__main__":
    main()

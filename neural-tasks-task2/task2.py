"""Run Task 2. The original Task 1 entry point remains main.py."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Task 2: multilayer backpropagation on penguins')
    parser.add_argument('--csv', type=Path, default=Path(__file__).with_name('penguins.csv'))
    parser.add_argument('--preset', type=Path, help='Load a result JSON configuration and train automatically')
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f'CSV not found: {args.csv}')
    from task2_gui import launch
    launch(args.csv, args.preset)


if __name__ == '__main__':
    main()

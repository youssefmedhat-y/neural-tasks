import argparse
import os
import sys


def _find_csv(given_path):
    if given_path:
        if not os.path.isfile(given_path):
            raise FileNotFoundError(f'CSV not found: {given_path}')
        return given_path

    # Default to penguins.csv in the same directory as main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(script_dir, 'penguins.csv')
    if not os.path.isfile(default):
        raise FileNotFoundError(
            f'penguins.csv not found in {script_dir}.\n'
            'Use --csv <path> to specify its location.')
    return default


def main():
    parser = argparse.ArgumentParser(description='Penguin Perceptron & Adaline Lab')
    parser.add_argument('--csv', default=None, help='Path to penguins.csv (default: ./penguins.csv)')
    args = parser.parse_args()

    try:
        csv_path = _find_csv(args.csv)
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

    from gui import launch
    launch(csv_path)


if __name__ == '__main__':
    main()

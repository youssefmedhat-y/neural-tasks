import argparse
import os
import sys


def _find_csv(given_path):
    if given_path:
        csv_path = given_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'penguins.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='Penguin Backpropagation Lab')
    parser.add_argument('--csv', default=None, help='Path to penguins.csv')
    parser.add_argument('--preset', default=None, help='Load settings from an exported result JSON')
    args = parser.parse_args()
    try:
        csv_path = _find_csv(args.csv)
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
    from task2_gui import launch
    launch(csv_path, args.preset)


if __name__ == '__main__':
    main()

"""Run from the repository root: python scripts/evaluate.py [--output PATH]."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anime_recommender.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_evaluation()
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
        print(json.dumps({'targets_passed':report['targets_passed'], 'aggregate':report['aggregate'], 'output':str(args.output)}, indent=2))
    else:
        print(rendered)
    return 0 if report['targets_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

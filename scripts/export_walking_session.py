"""Regenerate a stopped walking session's video from saved inputs; never re-run models."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.app.services.walking_export import WalkingExporter, read_status

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session",type=Path)
    parser.add_argument("--font",default="",help="Korean TTF/TTC font path")
    args=parser.parse_args()
    directory=args.session.resolve()
    manifest=json.loads((directory/"manifest.json").read_text())
    if not manifest.get("walking_risk") or manifest.get("status")=="running":
        parser.error("종료된 보행 위험 세션만 내보낼 수 있습니다")
    exporter=WalkingExporter(args.font)
    exporter.submit(directory)
    exporter.shutdown()
    status=read_status(directory)
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if not status or status["state"]!="ready":raise SystemExit(1)

if __name__=="__main__":main()

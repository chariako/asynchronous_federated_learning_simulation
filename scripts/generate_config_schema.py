import json
from pathlib import Path

from afl_sim.config import AppConfig


def main() -> None:
    schema = AppConfig.model_json_schema()

    output_path = Path("configs/config-schema.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, sort_keys=True, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()

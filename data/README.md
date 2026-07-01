# Data Directory

This repository does not include raw roadside V2X data or restricted derived tables.

For local experiments, place user-owned CSV files in this directory or pass an absolute path to `--csv`.

Recommended minimal schema:

| column | description |
| --- | --- |
| `proxy` | sparse target scaled to `[0, 1]` |
| `road_lane` | deployment group identifier for group split testing |
| `decoding_rate` | provenance or decoding-completeness feature |
| `packet_quality` | provenance or packet-quality feature |
| other numeric columns | model features |

Do not commit raw `.fbs`, personal data, proprietary deployment records, or manuscript review packages.

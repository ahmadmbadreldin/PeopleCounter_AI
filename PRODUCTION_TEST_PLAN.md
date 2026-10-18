# People Counter production test plan

This plan separates repeatable automated tests from tests that require the
actual branch camera, employees, doorway, lighting, Windows session, and power.
A release is accepted only when both sections pass.

## 1. Automated regression suite

Run `Run Automated Tests.cmd` before copying a release to any branch. It runs
syntax checks and every `test_*.py` test. The suite currently covers:

| Area | Best automated test | Pass condition |
|---|---|---|
| Config parsing | Valid, missing, corrupt, and legacy JSON | Clear rejection or correct upgrade |
| Camera credentials | Required fields, port range, URL encoding | Exact valid RTSP URL |
| Door geometry | Bounds, direction, coordinate conversion | Round trip and invalid zones rejected |
| Tracker | IoU, ID continuity, expiry, reset | No false predicted crossing |
| Crossing | Normal/reversed direction, hysteresis, gaps | Exactly one correct event |
| Staff uniform | Blank shirt, logo reference, repeated evidence | Blank shirt never becomes staff |
| Face profiles | Validation, normalization, replacement, corrupt file | No crash; correct employee selected |
| Enrollment helpers | Pose guidance, chest crop, file replacement | 9 capped valid references |
| YOLO errors | Transient, recovery, 30 consecutive failures | Continue transiently; escalate persistent fault |
| Camera state | Disconnect/reconnect events and downtime | One event per transition |
| Counter state | Reset, snapshot, day rollover | Exact persisted values |
| SQLite | WAL, schema, atomic event/state, restart | No counter loss after reopen |
| Excel | Monthly row values and automatic refresh | Monthly Summary matches SQLite |
| Retention | Old DB rows and reports | Removed at configured age |
| Heartbeat/status | Fresh, stale, missing, corrupt | Correct healthy/unhealthy exit code |
| Watchdog | Hung child termination | Terminate then kill fallback |
| Logs | Small rotation threshold | Backup count remains bounded |
| Windows artifacts | Required launchers/task/power actions | Static safety checks pass |

## 2. Technical branch smoke test

After **Save & Start**, close the counter and run
`Run Branch Acceptance Test.cmd`. It tests the real environment while skipping
the live camera connection:

- Python and required packages;
- active branch config and free disk space;
- real YOLO model load plus an inference;
- real YuNet/SFace model load;
- SQLite write/read and Excel creation.

All checks must show `[PASS]`.

When the camera is connected, its optional RTSP test can be run separately:

```powershell
.\.venv\Scripts\python.exe system_check.py --camera
```

That optional check requires ten valid frames from the configured stream.

## 3. Manual functional trials

Reset the counters before the controlled trial. Record ground truth on paper or
from a second observer. Never tune thresholds from memory.

### Door and customer counting

| Scenario | Repetitions | Expected |
|---|---:|---|
| One customer enters normally | 20 | IN +20, no OUT |
| One customer exits normally | 20 | OUT +20 |
| Customer pauses on the line then continues | 10 | One event per crossing |
| Customer approaches then turns back before crossing | 10 | No event |
| Two customers side by side | 10 pairs | Two events per pair |
| Two customers close behind each other | 10 pairs | Two events per pair |
| Customer carries large bags | 10 | One event each |
| Person walks beside but not through selected door | 10 | No event |

Required acceptance for the pilot: at least 95% correct single-person events,
at least 90% correct paired events, and zero counts outside the door zone.

### Staff filtering

Each consenting employee performs at least 10 entries and 10 exits, including
front, partial side, natural walking, and the normal branch lighting. Then 30
customers pass wearing light or white clothing.

- Target: at least 90% of registered staff crossings are ignored.
- Safety gate: **zero customers may be classified as staff** in the acceptance
  sample. A customer incorrectly ignored is a release blocker.
- Every unresolved/unknown crossing must be counted as a customer.

### Lighting and crowd conditions

Repeat at least 20 crossings in morning light and 20 near closing time. Include
door reflections, bright exterior light, partial occlusion, and the busiest
expected group size. Record accuracy separately rather than mixing it with the
easy trials.

## 4. Failure and recovery trials

Perform these after normal counting passes:

1. Unplug Ethernet for two minutes. Status must show camera disconnected; after
   reconnect, video returns without manual restart and Health records duration.
2. End `people_counter_hikvision.py` from Task Manager. Watchdog must restart it
   and same-day counters must restore from SQLite.
3. Restart Windows. After the same user signs in, the counter starts
   automatically and restores the same-day counters.
4. Shut down normally overnight. Next morning starts a new day's counters and
   yesterday's Excel exists.
5. Keep today's workbook open in Excel, create an event, close Excel, and verify
   the background retry updates the workbook without stopping counting.
6. Run `Check People Counter Status.cmd`; it must show `RUNNING` and `CONNECTED`.
7. Leave the system running for at least 72 hours. Memory should not grow
   continuously, heartbeat must remain fresh, and logs must rotate.

## 5. Data reconciliation

For the controlled trial, compare four sources:

1. observer ground truth;
2. on-screen IN/OUT counters;
3. SQLite `events` and `counter_state`;
4. The current date row in the Excel `Monthly Summary` sheet.

The screen, SQLite, and Excel must agree exactly. Any difference is a software
failure even when the camera accuracy itself is acceptable.

## 6. Release evidence

Keep the following per branch:

- date, branch ID, camera IP, stream, resolution, and camera position photo;
- automated test output and branch smoke-test output;
- manual trial sheet with expected/actual counts;
- staff false-positive and false-negative totals;
- restart, disconnect, Excel-lock, and 72-hour soak results;
- final approved `branch_config.json` stored securely.

# People Counter — Hikvision demo

Local branch people counter using a Hikvision RTSP camera, YOLO person
detection, multi-object tracking, directional entrance counting, durable
SQLite storage, daily Excel reports, and conservative staff filtering.

## One-click branch setup

On Windows, double-click:

```text
Start People Counter.cmd
```

The launcher creates `.venv` when needed, installs missing requirements, and
opens the branch setup screen. In that screen:

1. Enter a unique Branch ID and the camera IP, username, password, RTSP port,
   and stream (`101` is normally the clearer main stream; `102` is lighter).
2. Click **Test Connection**.
3. When the camera preview appears, drag a rectangle around the physical
   entrance. The green line at its center is the crossing line.
4. Select whether entering moves from top to bottom or bottom to top in the
   image.
5. With the employee's consent, click **Enroll Staff**, enter the employee
   name/code, and follow the on-screen FRONT/LEFT/RIGHT prompts. Repeat for
   every employee in that branch.
6. Click **Save & Start**.

The active configuration is written to `branch_config.json`. A separate copy
is also written to `branches/<branch_id>.json`, so the setup screen can load an
existing branch later. `branch_config.json` contains the local camera password
in plain text for this demo: keep the computer and project folder private and
never commit or share that file.

To start the already-active `branch_config.json` without changing it, use
**Start Active Config** in the setup screen. To switch branches, load the
saved branch and click **Save & Start**, which makes it active.

## Manual developer setup

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python setup_branch.py
```

`people_counter_hikvision.py` reads camera, geometry, model, staff, logging,
and display settings from `branch_config.json`. Directly starting the counter
requires a saved, calibrated configuration with a camera password.

## 24/7 branch-computer installation

Do this only on the final branch computer, after **Save & Start** has worked at
least once:

1. Double-click `Install Branch PC.cmd` and confirm the warning.
2. Restart Windows once and sign in with the same Windows user.
3. Confirm that the People Counter window starts automatically.
4. Double-click `Check People Counter Status.cmd` whenever you need a simple
   program/camera health check.

The installer creates a Windows Scheduled Task at user sign-in, and disables
Sleep and Hibernate on AC and battery power. It does
not prevent a normal nightly shutdown. The counter starts again after the next
morning's Windows sign-in. The monitor may turn off without stopping counting.

`watchdog.py` restarts the process after a crash, or after its heartbeat stops
updating. Closing the counter normally with `Q` or `Esc` also closes the
watchdog; use `Run People Counter 24-7.cmd` to start it again. Only one watchdog
instance is permitted per branch.

`Install Branch PC.cmd` is intentionally not run during development. It changes
Windows settings only when a technician explicitly runs it on the branch PC.
`Remove Branch Startup.cmd` removes automatic startup but deliberately leaves
the chosen Windows power settings for the technician to change in Power
Options.

## Storage, reports, and health

- Live events and the latest counters are stored in
  `data/<branch_id>.sqlite3` using SQLite WAL mode.
- A same-day program restart restores `IN`, `OUT`, staff ignored, unknown, and
  peak values. A new calendar day starts at zero.
- Daily workbooks are generated in `reports/<branch_id>/YYYY-MM-DD.xlsx`; the
  current day's workbook refreshes automatically shortly after every count or
  health event. Excel remains a report format, not the live database. Press
  `E` to force an immediate refresh. Reports are also refreshed on normal
  shutdown and day rollover.
- `runtime/<branch_id>_heartbeat.json` is updated every few seconds with program,
  camera, FPS, and counter status.
- Camera disconnect/reconnect periods and program/model errors are recorded in
  SQLite and included in the report's `Health` sheet.
- Runtime logs rotate at 5 MB with five backup files. SQLite events default to
  90-day retention and reports to 365 days; these values are configurable in
  `branch_config.json`.

## Counting behavior

- A predicted-only Kalman position cannot create an IN/OUT event.
- A track reacquired after missed observations has its side state reset; the
  unseen path is not guessed.
- Two observed positions are required on the new side of the line to reduce
  boundary jitter.
- Only crossings whose person center is within the selected entrance width are
  considered.
- The visual setup chooses which crossing direction means `IN`.
- `INSIDE` is `max(0, IN - OUT)`. It is an estimate, not proof of physical
  occupancy.

## Hybrid staff filtering and enrollment

Staff filtering combines per-branch face recognition with the actual white
formal button-shirt and black front-chest logo captured during enrollment.
The logo is not expected on the back, so only FRONT chest samples become
uniform references; side views are retained for face enrollment only. OpenCV YuNet detects
faces and SFace creates normalized embeddings. Raw face images are not stored;
the employee code and embeddings are saved in:

```text
branches/<branch_id>/staff_faces.npz
```

Chest-only uniform references are stored in
`branches/<branch_id>/uniform_refs/`. Re-enrolling the same employee code
replaces that employee's old face and uniform samples.

If a person crosses before staff classification is ready, the crossing waits
for up to 2.5 seconds for more face/uniform evidence. Confirmed staff crossings
become `STAFF_IN_IGNORED`/`STAFF_OUT_IGNORED`. Because missing a staff member is
the higher-risk error, the system keeps collecting evidence before deciding. If the crossing is still
unresolved after the timeout, it is counted as a customer and logged as
`IN_UNKNOWN_COUNTED` or `OUT_UNKNOWN_COUNTED`.

Face and uniform quality still depend on camera angle, face size, lighting,
blur, and occlusion. Enrollment must use the final installed camera, and every
branch must be validated with consenting employees and test visitors.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Or double-click `Run Automated Tests.cmd`. On the installed branch computer,
run `Run Branch Acceptance Test.cmd` to additionally verify the real models
and SQLite/Excel. The live RTSP check is optional and can be run later with
`system_check.py --camera`. The complete manual accuracy, failure, restart,
and 72-hour soak procedure is in `PRODUCTION_TEST_PLAN.md`.

The offline tests validate crossing state, direction reversal, staff evidence
aggregation, face-profile persistence, pending event decisions, and
configuration validation. They do
not validate real YOLO accuracy, live RTSP reliability, or a particular camera
angle.

## Remaining limitations

- The custom IoU/centroid tracker can switch IDs in crowding; it has no ReID
  appearance model.
- Occlusion, missed detections, and tracks starting inside the crossing band
  can cause count errors.
- A sudden power loss can lose only the last periodic state snapshot when no
  crossing event occurred; every crossing event and its counters are committed
  immediately to SQLite.
- Face matching is biometric processing. Obtain appropriate consent, restrict
  access to each branch profile, and delete a profile when it is no longer
  needed.
- The application must be validated against manually counted entrance trials
  before its numbers are used.

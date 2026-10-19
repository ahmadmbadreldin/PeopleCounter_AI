# People Counter — Hikvision demo

Local branch people counter using a Hikvision RTSP camera, YOLO person
detection, multi-object tracking, directional entrance counting, durable
SQLite storage, daily Excel reports, and conservative staff filtering.

## Branch PC prerequisites

Only these external programs are required on a new branch computer:

- 64-bit Windows 10 or 11;
- **64-bit Python 3.11**, installed with `Add Python to PATH` and the default
  `tcl/tk and IDLE` component enabled;
- **Microsoft Visual C++ 2015-2022 Redistributable (x64)**;
- an internet connection during the first setup so the pinned Python packages
  can be installed.

The project includes the YOLO, YuNet, and SFace model files. SQLite is built
into Python, OpenCV includes the video support used by the application, and
Excel is not required to create `.xlsx` reports. Microsoft Excel or LibreOffice
is optional and is needed only if somebody wants to open those reports.

The launchers run `bootstrap_environment.py`, which creates `.venv`, installs
the exact tested versions from `requirements.txt`, and verifies OpenCV and
PyTorch imports. A copied `.venv` from another computer is not supported.

## Two separate operator screens

The branch technician uses two clearly separated launchers:

```text
1 - Configure Branch.cmd
2 - Start Monitoring.cmd
```

`1 - Configure Branch.cmd` opens camera/branch configuration, staff enrollment,
and entrance drawing only. It does not start the monitoring screen. The first
run creates `.venv` and installs missing requirements when needed. In that
screen:

1. Enter a unique Branch ID and the camera IP, username, password, RTSP port,
   and stream (`101` is normally the clearer main stream; `102` is lighter).
2. Click **Test Connection**.
3. Drag one rectangle around the physical doorway. The dashed orange rectangle
   is the saved entrance and the new selection is solid cyan. The coordinates
   are committed continuously while dragging, so AnyDesk does not have to
   deliver the final mouse-release event reliably.
4. Move the **Counting line** slider to place the green line anywhere from 10%
   to 90% of the rectangle height.
5. Select whether entering moves from above the line to below it, or from below
   the line to above it. The yellow `IN` arrow confirms the selected direction.
6. With the employee's consent, click **Enroll Face**, enter the employee
   name/code, and follow the FRONT/LEFT/RIGHT prompts. Repeat for every
   employee. Repeating the same employee later adds diverse face embeddings
   instead of deleting the earlier lighting/angle coverage.
7. Click **Enroll Uniform** at least once. Follow the timed FRONT/LEFT/RIGHT/
   BACK prompts. Repeat this session with different employees and once under
   morning and evening lighting; successful sessions accumulate in the same
   branch profile.
8. Click **Save Configuration**, then close the configuration window.
9. Double-click **2 - Start Monitoring.cmd**. It opens only the live monitoring
   window using the active saved configuration. The watchdog restarts that
   window after a crash or frozen heartbeat.

The active configuration is written to `branch_config.json`. A separate copy
is also written to `branches/<branch_id>.json`, so the setup screen can load an
existing branch later. `branch_config.json` contains the local camera password
in plain text for this demo: keep the computer and project folder private and
never commit or share that file.

Saving a new entrance rectangle or line position is detected by the running counter within the
heartbeat interval (normally five seconds). Track-side history is cleared so
the moved line cannot create a false crossing, while daily totals remain
unchanged. A per-branch process lock prevents an older and newer counter from
counting simultaneously.

To switch branches, open **1 - Configure Branch.cmd**, load the saved branch,
and click **Save Configuration** to make it active. Then start or restart the
monitoring launcher. If monitoring is already running, geometry changes are
detected automatically without resetting the daily totals.

If an old yellow counting rectangle remains after saving a new cyan rectangle,
double-click `Stop Old People Counter.cmd`. It stops the scheduled watchdog and
only the Python processes belonging to this project; unrelated Python programs
are left running. Then run `1 - Configure Branch.cmd`, test the connection,
redraw the doorway, and click **Save Configuration**. The scheduled task remains
installed and will run again at the next Windows sign-in.

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

Do this only on the final branch computer, after **Save Configuration** and
**2 - Start Monitoring.cmd** have both worked at least once:

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
updating. Closing the counter normally with `Q`, `Esc`, or the window `X` also closes the
watchdog; use `2 - Start Monitoring.cmd` to start it again. Only one watchdog
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
- One workbook is generated for every day at
  `reports/<branch_id>/YYYY-MM-DD.xlsx`. It contains one sheet named
  `Hourly Report` with 24 rows, from `00:00 - 00:59` through
  `23:00 - 23:59`. Each row contains only that hour's `IN`, `OUT`, and
  `Staff Ignored`. The current hour is refreshed shortly after every count or health
  event and at least once per minute while the program is running. Future
  hours remain blank. Excel remains a report format, not the live database.
  Press `E` to force an immediate refresh. Reports are also refreshed on
  normal shutdown and day rollover.
- `runtime/<branch_id>_heartbeat.json` is updated every few seconds with program,
  camera, FPS, and counter status.
- Camera disconnect/reconnect periods and program/model errors remain recorded
  internally in SQLite but are not shown in the operator Excel report.
- Runtime logs rotate at 5 MB with five backup files. SQLite events default to
  90-day retention and reports to 365 days; these values are configurable in
  `branch_config.json`.

## Counting behavior

- The red point is the bottom-center of the latest real YOLO person box, an
  approximation of the person's foot position. A crossing is based on this
  point passing the green line, not on the center of the torso.
- A predicted-only Kalman position cannot create an IN/OUT event.
- A track reacquired after missed observations has its side state reset; the
  unseen path is not guessed.
- Two observed positions are required on the new side of the line to reduce
  boundary jitter.
- Only crossings whose red foot point is inside the selected entrance polygon
  are considered.
- The visual setup accepts an angled counting line and chooses its `IN` side
  directly; the yellow `IN` arrow is also shown in the running counter.
- `INSIDE` is `max(0, IN - OUT)`. It is an estimate, not proof of physical
  occupancy.

## Hybrid staff filtering and enrollment

Staff filtering is a strict OR: a confirmed registered face **or** a confirmed
uniform marks the track as staff. The production fallback works immediately,
without classifier training, from white upper clothing (shirt or T-shirt) and
gray trousers. A black belt adds at most 0.05 confidence and a matched
front-chest logo adds at most 0.10; neither is required. Missing logo references,
a back view, or an ORB/homography failure cannot reject otherwise valid white +
gray evidence. Evidence is aggregated across real observations only;
predicted-only tracker boxes are never analyzed.

OpenCV YuNet detects faces and SFace creates normalized embeddings. Raw face images are not stored;
the employee code and embeddings are saved in:

```text
branches/<branch_id>/staff_faces.npz
```

Optional chest-only logo references are stored in
`branches/<branch_id>/uniform_refs/`. Re-enrolling the same employee code
adds diverse face and uniform samples up to a bounded limit. Active repeated
uniform enrollment statistics are stored in:

```text
branches/<branch_id>/uniform_profile.json
```

The runtime checks uniform evidence every second processed frame and stops the
expensive checks after a track is confirmed as staff. Two ordinary positive
observations, or one exceptionally strong observation, confirm staff. A staff
decision is sticky for that track. A customer decision requires six valid
negative observations; cropped, too-small, or occluded views are `UNKNOWN` and
do not count as negative evidence. These values can be changed per branch from
**Uniform Settings** in `1 - Configure Branch.cmd`. Face recognition can be
disabled there without disabling uniform detection.

When four consecutive valid full-body observations clearly contain neither the
required white upper clothing nor gray trousers, the track becomes `CUSTOMER`.
White-only or gray-only evidence still waits for more frames. A confirmed face
always overrides this fast path and marks the track as `STAFF`.

If a person crosses before staff classification is ready, the crossing waits
for up to 3.5 seconds for more face/uniform evidence. Confirmed staff crossings
become `STAFF_IN_IGNORED`/`STAFF_OUT_IGNORED`. Because missing a staff member is
the higher-risk error, plausible partial white + gray evidence at timeout is
biased to staff. A truly unresolved crossing with no usable evidence is counted
in its real direction and logged once as `IN_UNKNOWN_COUNTED` or
`OUT_UNKNOWN_COUNTED`.

Face and uniform quality still depend on camera angle, face size, lighting,
blur, and occlusion. Enrollment/calibration must use the final installed camera,
and every branch must be validated with consenting employees and test visitors.
Mobile-phone photos are useful references but are not a substitute for final
camera calibration because perspective, compression, exposure, and distance differ.

### Collecting a future uniform-training dataset

Open `1 - Configure Branch.cmd`, load the correct branch, and click
`Collect Future Training Dataset`. Choose `YES` for a consenting uniformed employee or
`NO` for a consenting volunteer in normal clothes. This is a controlled
calibration tool: do not use it to record ordinary shoppers.

The session starts automatically after a three-second countdown and normally
finishes after 60 seconds, so it also works when remote-control keyboard input
is unreliable. Keep only one subject prominently in view and ask them to move
front, back, left, and right. The collector selects the largest detected
person, rejects blurry and near-duplicate frames, and removes the top/head
area before saving. Samples and non-biometric session metadata are stored in:

```text
branches/<branch_id>/uniform_dataset/staff/
branches/<branch_id>/uniform_dataset/customer/
branches/<branch_id>/uniform_dataset/_sessions/
```

Customer samples are optional. **Enroll Uniform** staff sessions immediately
update the branch profile using robust white/gray statistics; the default rules
still work when no profile exists. **Collect Future Training Dataset** samples
are retained for a later reviewed classifier-training step. Review and remove mislabeled or poor crops
before training.

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

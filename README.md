# People Counter — Hikvision demo

Local branch people counter using a Hikvision RTSP camera, YOLO person
detection, multi-object tracking, directional entrance counting, Excel event
logging, and conservative staff-uniform filtering.

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
the system keeps collecting evidence before deciding. If the crossing is still
unresolved after the timeout, it is counted as a customer and logged as
`IN_UNKNOWN_COUNTED` or `OUT_UNKNOWN_COUNTED`.

Face and uniform quality still depend on camera angle, face size, lighting,
blur, and occlusion. Enrollment must use the final installed camera, and every
branch must be validated with consenting employees and test visitors.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_tracking.py
```

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
- Counters restart from zero when the app restarts; previous Excel events are
  retained but are not used to restore state.
- Face matching is biometric processing. Obtain appropriate consent, restrict
  access to each branch profile, and delete a profile when it is no longer
  needed.
- The application must be validated against manually counted entrance trials
  before its numbers are used.

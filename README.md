# Badger Sokoban — play-only release

Four independent challenges: Boxes only, Keys & doors, Survival, and Keys, doors & survival.
The first challenge loads automatically. A win shows YOU WIN for 1.8 seconds, then starts
the next challenge; the final win stays on screen. Manual restart or level selection cancels
pending automatic progression. Existing browser sessions resume where they left off.

## Runtime and theory

Every action goes through `dbn_composer.step_model(..., 'factorized')` using the tutorial's
compiled H functions and local kernels. No alternate Sokoban simulator, policy, or full
Cartesian product is used for play. The map canvas only renders the returned state.

`levels/` contains release snapshots of the saved tutorial worlds, including the moved lake.
`runtime/` contains copies of the existing Python machinery so this deployment does not track
live edits to the tutorial. Change the source in its original authorized location, then refresh
the release copy deliberately. Do not edit these runtime copies as a second implementation.

The HTTP API only permits starting, resuming, restarting and stepping an allowlisted level.
It does not accept client-provided worlds, H specifications, or states. Each browser tab keeps
an opaque session token in sessionStorage. Sessions expire after 12 hours of inactivity.
Server restarts reset active games. Winning or losing freezes actions until restart.

## Astera deployment

- URL: https://sokoban.astera.sh/ (Astera private network / Tailscale)
- Namespace: `obelisk`
- Deployment, Service, Ingress: `okbe-sokoban-play`
- Source sync workspace: `okbe-sokoban`, CPU-small
- Persistent storage: `actl-ws-tomringstrom-okbe-sokoban-home`
- Runtime interpreter: `/home/dev/okbe-venv/bin/python`

The game is a separate Kubernetes Deployment with restart and health probes. It reads the
workspace PVC without write access; it does not need the laptop's actl sync or port forwarding.
Keep the PVC while this deployment exists. The development workspace can be stopped after
release; do not delete its data. Only one serving replica is configured because sessions are
held in process memory.

All source edits happen locally and reach the workspace through `actl pod sync`. Run checks
on the workspace:

```sh
actl pod exec okbe-sokoban -n obelisk -- sh -lc 'cd /home/dev/workspace && OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/okbe-mpl /home/dev/okbe-venv/bin/python verify_release.py'
```

The verifier searches winning routes for the actual release maps and replays every action
through the play API; it also checks session isolation, restart, terminal outcomes, invalid
actions, and absence of editing/solver endpoints. Routes are only test artifacts, not exposed
to players. The remote report is `/home/dev/okbe-release-verification.json`.

Deployment configuration is in `deploy/`. After changing the release, run its verification
before restarting `deployment/okbe-sokoban-play` to load the new Python code and levels.

## Portable container

The Dockerfile contains the same app, assets, levels, and Python runtime copies. Build with
this folder as context and expose port 8080. It uses Waitress and runs as an unprivileged user.
The Astera deployment instead uses the same Python base image and the PVC-installed virtual
environment; no registry push is required for this release.

## Player metrics

The player interface lists **Boxes only**, **Keys & doors**, **Survival**, and **Keys, doors & survival**,
without the internal level numbers. Wait is not exposed in the player controls.

A secure, HTTP-only browser cookie remembers an anonymous player identifier for up to a year.
The visible counter shows total failures and failures on the selected challenge. Failure is
counted once when the DBN state violates a constraint and reaches YOU LOSE; voluntary restarts
are separate. Clearing cookies or using another browser creates another player identity.

Persistent reports are on the same PVC, outside the synced source tree:

- Serving pod: `/data/player-<anonymous-id>.md`
- Workspace pod: `/home/dev/metrics/player-<anonymous-id>.md`

Each Markdown report includes attempts, actions, wins, failures, restarts, best winning action
count, and timestamped attempt events with elapsed time and failure reasons. A companion JSON
file preserves the counters across process restarts. These files are not served by the website.
Active game sessions remain in memory and reset on a server restart; accumulated metrics do not.

`test_player_metrics.py` verifies persistence, player isolation, no repeated counting after a
loss, voluntary restarts, and unavailable public report paths. Tests use temporary storage,
not the deployed `/data` directory.

Held arrow keys use an immediate first step and a 250 ms interval (4 steps per second), ignoring OS repeat events.
Releasing the key, leaving the window, hiding the tab, restarting, or switching levels stops
movement. Busy requests are skipped rather than queued, so network delays cannot cause a burst
of delayed moves. Buttons retain their appearance during requests; the request guard still
prevents overlapping steps. A prominent contextual Space prompt identifies eating, drinking,
key pickup, and opening an adjacent matching door.

The keyboard clock test is `test_key_repeat.py`; its extra dependency is in
`requirements-test.txt` and is not needed by the game server.

Deterministic next moves are previewed by the server using the same DBN factorization. The browser displays that exact successor immediately and reconciles with the authoritative response. Up to three discrete taps are buffered; held repeats never accumulate. Stochastic actions wait for confirmation.

### Full attempt trajectories

New attempts have one append-only JSONL file per attempt under
`/data/player-<anonymous-id>-attempts/<attempt-id>.jsonl` (workspace: `/home/dev/metrics/`).
The Markdown report links to this folder. Records include the initial full joint state,
level file SHA-256, action-name mapping, and every accepted `(state, action, next_state)`
with step number, UTC timestamp, elapsed seconds, induced alphas, outcome, and constraint reasons.
Blocked moves and interactions with no effect are included. Rejected requests, browser key
repeats that generate no game action, and actions after termination are not transitions.
Restart closes the previous trace with a restart event and creates a new attempt ID.
Leaving a page or a server restart leaves the already-written partial trajectory intact;
it is not classified as a win or loss. Earlier unlogged transitions cannot be recovered.
Trajectories use the existing random browser identity and contain no names or emails.

On the final **Keys, doors & survival** challenge only, holding Shift for one key press
uses one of 20 per-attempt rewind steps to rewind one accepted transition. The remaining
budget is shown above the map. The rewind is recorded in the append-only JSONL audit
trace, including the discarded transition and restored state. Each attempt also has a
`<attempt-id>.true.json` sidecar containing the active trajectory after discarded suffixes
are spliced out; the winning action count and elapsed time use this active branch. Shift
also works during the loss overlay and restores the preceding live state. The hint is shown
above the map only on this final challenge.

Boxes only is temporarily hidden from the public selector and automatic progression. Its level file and backend support remain available. The visible order is Keys & doors, Survival, Keys, doors & survival. Resuming a hidden level redirects to the first visible challenge.

### Optional player names
Players can save, change, or clear an optional name/nickname for prize identification.
The cookie selects the existing player record; a name change preserves previous scores
and trajectories. Names appear in private reports and on the public per-level leaderboard alongside best winning action counts.
The same name can be entered by different browsers; this is not identity verification.
`best_win_actions` counts all accepted actions (including Space and blocked movements),
per level, rather than summing steps across levels. No email is requested.

The public leaderboard shows the top 20 named players per visible level. Ties use competition ranking (1, 1, 3). Unnamed scores remain private. Clearing a name removes that player from the public boards while preserving their score and history. Public responses exclude browser IDs and trajectories.

First-time players see a welcome dialog offering Join leaderboard or Play anonymously.
Either choice is remembered server-side for that browser; names remain editable later.
Leaderboards order by fewest winning actions, then earliest achievement time for the same winning run.
Time starts with the first accepted action, so entering a name does not count, and includes
pauses afterward. New times have millisecond precision. Older best-run times are recovered
from saved win events at their original precision. Equal action counts and achievement timestamps share rank.

Unnamed players are prompted again on a page reload or explicit level restart. Dismissal is held in memory through ordinary moves and automatic progression. Players with a saved name are not prompted again.

On a confirmed loss, a circular 3–2–1 countdown restarts the level automatically. Automatic retries skip the name prompt; switching levels or restarting manually cancels a pending countdown.
Leaderboard reset excludes previous wins from public ranking while preserving private counters and trajectory history. Re-entering a name does not restore excluded scores.

Visible leaderboards refresh every 15 seconds and on tab focus/visibility return. Background updates keep the current rows visible until the new response arrives. All three public challenges start at cell 28; Survival is verified solvable from that cell.

Competition tie-break rule: fewest winning actions, then earliest achievement time for that same action count. A later faster run does not replace an earlier equal-action solve. The visible clock starts on the first action, includes pauses, and freezes at the authoritative duration on win/loss. Restart resets it to zero. The leaderboard displays run duration; UTC achievement timestamps are used for ranking.

A player can be permanently excluded from the prize leaderboard with `leaderboard_excluded`. Exclusion clears the name, prevents setting another name on that player record, hides the name-entry UI, and excludes future wins without deleting trajectories or private statistics. The organizer player is excluded.

Prize deadline: Friday September 18, 2026 at 11:59 PM Pacific (September 19 06:59 UTC). The page shows a server-corrected countdown. Only wins recorded before that instant are eligible; post-deadline play and trajectories remain available but do not improve prize standings. `test_deadline.py` checks timezone conversion, cutoff boundaries and clock correction.

Blocked directional inputs (walls, boundaries, closed doors, unpushable boxes) do not submit an environment time step in the play interface. The DBN determines whether movement is possible. Such inputs preserve the entire joint state and action count, and are saved as `blocked_input` records with `counted: false`. Successful moves/pushes still consume physiology. The run clock continues during blocked inputs once started.

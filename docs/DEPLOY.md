## Deploying to a box that is not a git checkout

Two boxes (.161, .94) are tar copies, not checkouts. `git archive | tar -x`
**overwrites but never deletes**, so a file that MOVED in the repo stays behind
on those boxes forever.

It happened on 2026-08-13: `gate.py` and `appearance.py` moved into
`counting/`, and both boxes kept the old copies under `services/runner/app/`.
Nothing imported them, so nothing failed — the boxes just carried two dead
files that a checkout would have removed, and any future change that
reintroduced a same-named module would have found a stale one waiting.

`GET /health` now returns a `build` id: a content hash of every `.py` the
process could run. It caught this within a minute of the deploy — T440
reported one hash and the two tar boxes another. **Compare the build across
the fleet after every deploy**; identical code means an identical hash,
whatever the transport was.

    for h in 192.167.1.14 192.166.1.161; do
      curl -s http://$h:7100/health | python3 -c 'import sys,json;print(json.load(sys.stdin)["build"])'
    done

Prefer `rsync -a --delete` over `tar -x` for these boxes; where tar is all
there is, remove moved files by hand and re-check the build.

## Never silence the pull

`git pull --ff-only ... >/dev/null 2>&1` in a deploy one-liner hides the one
error that matters. It happened twice on 2026-08-13: a bundle whose base
commit the box did not have printed `Repository lacks these prerequisite
commits` straight into /dev/null, the build and restart ran happily on the OLD
tree, and the box reported healthy.

Both times the `build` id in /health was the only thing that noticed. Deploy
with the pull UNSILENCED and `set -e`, then compare the build across the fleet.
A bundle is `--not <the box's HEAD>`, not `--not <your last deploy>` — those
diverge the moment one box misses a commit.

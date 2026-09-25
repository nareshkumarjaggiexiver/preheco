# JD Grand — live camera checklist (2026-09-25)

For the operator at the venue. The box runs build `b3d2c39` (or later) on
camera A (`:7100`) and camera B (`:7200`): TensorRT, overlapped detection,
NVIDIA hardware decode, live-camera reconnect, and the latest duplicate-review
rules. `./scripts/demo-up.sh status` on the box shows every one of them.

## 1. Before leaving

- [ ] GPU clock lock registered on the box (section 5). Without it the card
      idles at 210–495 MHz between frames and every stage runs several times
      slower.
- [ ] Box, laptop, camera, stand, LAN switch and cables packed.
- [ ] **A PoE switch with enough power for the camera.** At night the
      camera's infrared light switches on and draws more power; on an
      underpowered port the camera's light blinks on and off and it drops
      off the network every minute or so (the Sharon test, 2026-09-25).
- [ ] **The camera's switch port in its normal (100 m) mode, not 250 m.**
      On most PoE switches the 250 m "extend" mode runs the port at
      10 Mbit/s, and the camera's 4K main stream alone is about 8.6 Mbit/s.
      At Sharon, 16:45, on that port: every stream started at 15 fps, then
      fell to 8–9 fps with 1-second gaps and one 7.8 s stall. The camera
      answered ping in 37–209 ms (the office router: 4 ms), with nothing on
      the laptop connected to it. Not yet confirmed on the switch itself.
      Power the camera from a PoE+ port or with a better cable instead. If
      250 m mode must stay, set the main stream to 4–5 Mbit/s constant
      bitrate and keep the NVR off this camera.

## 2. Power-up order

1. **Box**: power on and log in to Windows. The logon starts WSL (task
   "HECO WSL keepalive") and locks the GPU clocks (task "HECO GPU clocks").
   The pipeline containers start by themselves.
2. **Laptop**: start the console in WSL:
   ```bash
   cd ~/projects/heco/apps/heco-site-planner
   HECO_BIND=0.0.0.0 HECO_INSECURE_LAN=1 PLANNER_PUBLIC_URL=http://<laptop-ip>:8787 npm run dev
   ```
3. In the console's pipeline page, camera A and camera B should both show
   healthy. If not, go to section 3.

## 3. If the network is different

Find each IP with Windows `ipconfig` (box, laptop) or the camera's own tool.

| What changed | Do this |
|---|---|
| Box IP | Laptop, in `heco-site-planner`: `python3 scripts/repoint-box.py eldesign-rtx-4060 <box-ip>` to preview, then again with `--apply`. It moves the box and all three installs in one step. |
| Laptop IP | Box: `cd ~/heco-pipeline-new && PLANNER_URL=http://<laptop-ip>:8787 ./scripts/demo-up.sh both`. Restart the console with the new `PLANNER_PUBLIC_URL`. |
| Laptop's WSL IP (after a laptop reboot) | Laptop, Administrator PowerShell: `netsh interface portproxy set v4tov4 listenport=8787 listenaddress=0.0.0.0 connectport=8787 connectaddress=<wsl-ip>`. Get the WSL IP with `wsl hostname -I`. |
| Camera IP | Console: site, then the device, then edit the stream URL. |

The box's firewall allows its ports on every network type, and the laptop
already works on a Public network, so a venue network needs no firewall
change.

## 4. The run

- Use the camera's **main** stream (4K H.265) for analysis.
- **Check the camera's fps at the zoom you will use.** At Sharon the same
  CP Plus model halved its frame rate at full zoom (15 to about 7.5 fps,
  indoor lights on) and went back to 15 fps at 50% zoom. Two explanations
  fit: at the long end the lens lets in less light and the camera slows its
  shutter; or the 10 Mbit/s port above (more detail means a higher bitrate).
  Check the port mode first. Then, if it still halves, zoom out, or cap the
  shutter at 1/25 s in the camera's exposure settings and let gain make up
  the brightness. The box processed about 99% of the frames the camera sent
  either way.
- Start the run from the console on camera A. Use camera B only for a
  second camera. Leave the quality choice on **Box default**: the box now
  drops half faces itself (landmarks, frontality 0.55, eye span 0.30), which
  removed profiles, backs of heads and pillar-hidden faces on the Sharon
  test clip and kept every frontal guest. It also drops a face half hidden
  behind the head of someone standing nearer the camera (the D02 recording:
  8 guests to 7, the half face gone, every real guest kept) — under Strict
  too.
- Watch: processed fps against the camera's fps, dropped frames, and on the
  box `./scripts/demo-up.sh status` for TensorRT and the `nvdec` decoder.
- The run's processing fps reads `avg · min · max`. avg is over the whole
  run; min and max are the slowest and fastest 5-second window. A min of 0
  means the camera sent nothing for at least 5 seconds; max is what the
  pipeline does when it is fed. If avg is low but max is near the camera's
  fps, the camera or network is the problem, not the box.
- **Removing a guest who is not one** (a half face, a poster, a staff member
  you do not want enrolled): in the console's guest list, press **Remove** on
  the card, then confirm. The count drops by one; the same face seen again
  stays out of the count. Works during and after a run.
- **Turban against bare head.** The box reads each new guest's head covering
  (SigLIP) and the duplicate review sets aside a pair when both are men and
  one wears a turban, the other is bare-headed — the p00005/p00009 kind.
  Those pairs are listed under "Set aside" in the console, mergeable, with
  "one man wears a turban and the other is bare-headed". If one of them is
  wrong, switch the rule off (the reading stays):
  `HECO_REVIEW_HEADWEAR=0 ./scripts/demo-up.sh both` — the review changes at
  once, nothing is re-counted.
- A camera that drops out is reopened as soon as it answers again (checked
  every 10 seconds), and the run stays open through an outage of up to ten
  minutes. The console's silence alarm shows after 30 seconds. After ten
  minutes the run settles as failed; stop it yourself sooner if the camera
  is gone for good.
- Measured the night before on the Sharon CP Plus camera (4K H.265 at
  15 fps, this build, GPU clocks not yet locked): 13-14.6 fps processed,
  89% of the camera's frames. The camera then dropped off the network for
  about two minutes, which is why the run now waits ten minutes instead of
  45 seconds. The console's run fps averages over the outage too, so it
  read about 6-7 fps.

## 5. GPU clocks

Administrator PowerShell on the box, once:

```powershell
nvidia-smi -lgc 2400,3105
Register-ScheduledTask -TaskName "HECO GPU clocks" -Trigger (New-ScheduledTaskTrigger -AtLogOn) -Action (New-ScheduledTaskAction -Execute "C:\Windows\System32\nvidia-smi.exe" -Argument "-lgc 2400,3105") -RunLevel Highest -Force
```

Check during a run: `nvidia-smi --query-gpu=pstate,clocks.gr --format=csv`
should read P2 and at least 2400 MHz. Undo: `nvidia-smi -rgc`.

## 6. Don't

- Don't restart the console during a run. The box keeps counting, but the
  pictures and stats it sends during the restart are lost.
- Don't run footage (file) counts on the box during the live run: they
  share the one GPU.

## 7. If something goes wrong

| Symptom | Do this on the box, in `~/heco-pipeline-new` |
|---|---|
| Status shows the decoder fell back to cpu, or the stream will not open | `INGEST_DECODER=cpu ./scripts/demo-up.sh both` |
| Anything else that started after the speed levers | `HECO_DEMO_LEVERS=0 ./scripts/demo-up.sh both` (the configuration before 2026-09-25) |
| The head-covering reader misbehaves (embed not healthy, reads failing in status) | `EMBED_HEADWEAR_MODEL= ./scripts/demo-up.sh both` — the reader off, everything else as is |
| WSL is gone | Log in to Windows (the keepalive task restarts it), or open Ubuntu. |
| Console cannot reach the box | Section 3. `ping <box-ip>` from the laptop first. |

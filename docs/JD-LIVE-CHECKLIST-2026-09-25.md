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

- Use the camera's **main** stream (4K H.265) for analysis. Full focal
  length is fine: bigger faces help.
- Start the run from the console on camera A. Use camera B only for a
  second camera.
- Watch: processed fps against the camera's fps, dropped frames, and on the
  box `./scripts/demo-up.sh status` for TensorRT and the `nvdec` decoder.
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
| WSL is gone | Log in to Windows (the keepalive task restarts it), or open Ubuntu. |
| Console cannot reach the box | Section 3. `ping <box-ip>` from the laptop first. |

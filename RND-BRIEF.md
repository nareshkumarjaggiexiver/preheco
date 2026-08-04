# R&D brief — accuracy and robustness of a face-recognition people-counter

OUR SYSTEM (read CONTRACTS.md + services/*/README.md for specifics):
Edge people-counting at event gates (Indian marriage palaces first). Pipeline:
ingest (RTSP/file) → person detect (YOLOX-nano ONNX) → track (own SORT-lite,
greedy IoU + constant velocity) → face detect (YuNet) → quality gate → embed
(SFace, 128-d) → match (cosine vs a per-run SQLite gallery; staff store checked
first) → unique count. POC geometry: 2.8 mm fixed camera at 2.0 m, subjects at
2–3 m, faces ~64–176 px, evening warm/decorative lighting, crowd surges
("baraat"), veils, children, staff to be excluded. Ship closed-source
(Apache/MIT only). CPU-first edge box.

THE JOB: mine real-world experience — IPCamTalk / r/computervision / r/homelab
forums, GitHub issues + READMEs of ByteTrack/BoT-SORT/OC-SORT/DeepSORT,
InsightFace/SFace/ArcFace/AdaFace repos and issues, FRVT/NIST notes, DORI/IEC,
face-recognition-at-a-distance and crowd-counting papers, blog post-mortems of
production face pipelines — for the accumulated craft that separates a demo
from a system that counts a wedding correctly. We want the MINUTE tweaks people
learned over years, each mapped to a concrete change in OUR pipeline.

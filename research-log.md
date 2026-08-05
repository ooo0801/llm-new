# Research Log

Chronological, append-only record for Experiment A.

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-05 | bootstrap | Started H-A1 from `v2-global-calibrated-coverage` on branch `experiment/prompt-transfer-14b-a1`. Existing V1/V2/V6 evidence is immutable. |
| 2 | 2026-08-05 | bootstrap | Audited storage and GPUs: three 32 GiB RTX 4080 SUPER GPUs are idle; after conservative cleanup the system and data filesystems have about 28 GB and 26 GB free. |
| 3 | 2026-08-05 | protocol | Verified the official target revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, eight BF16 shards totaling 29,540,134,000 bytes, and the V6 frozen set rule: 22 rows filtered to 16 where `optimization.accepted` is true. |
| 4 | 2026-08-05 | protocol | User authorized deletion of the 7B local weight cache and continuation with 14B. Protocol will be committed before deletion, target download, or endpoint scoring. |
| 5 | 2026-08-05 | inner-loop | After protocol commit `03f4cd7`, deleted only the 7B Hugging Face cache (15,242,862,592 bytes), verified V1/V2/V6 evidence remained, and started fixed-revision 14B download as PID 4250 with 42,999,046,144 bytes free beforehand. |
| 6 | 2026-08-05 | protocol | Pre-endpoint code audit clarification A1-P1: V6 task validation is deterministic greedy generation without a system prompt; formal micro scoring uses complete blockwise parameter coverage. No endpoint result had been produced and no frozen scientific gate changed. |
| 7 | 2026-08-05 | inner-loop | Download attempt 1 failed after 2.51 GB because the Xet/CDN connection was reset. The partial cache was retained; no model endpoint was loaded. |
| 8 | 2026-08-05 | inner-loop | Restarted from the partial cache as PID 4872 with Xet disabled, ordinary HTTP, one worker, 600-second download timeout, and 40,483,983,360 bytes free. |
| 9 | 2026-08-05 | inner-loop | Fixed-revision 14B download completed and all eight BF16 shard sizes were validated (29,540,134,000 bytes). The data filesystem retained 13,445,230,592 bytes free. |
| 10 | 2026-08-05 | gate | Tokenizer/config preflight passed 16/16 transferred pairs. Maximum initial/optimized length was 82 tokens, below both the 512-token task and 128-token sensitivity limits. |
| 11 | 2026-08-05 | engineering | The first model-smoke attempt failed before weight loading because the server Torch build rejected peak-memory reset on unopened CUDA devices. No endpoint or scientific result was produced; fix commit `4f1ad7d` initializes CUDA in the fresh process. |
| 12 | 2026-08-05 | gate | Full 14B BF16 load and 4-token generation passed in a single process across GPUs 0/1/2 with no CPU/disk offload. Peak allocated memory was 8,728,963,072; 10,476,597,760; and 10,381,702,144 bytes; generated text was nonempty and correct. |

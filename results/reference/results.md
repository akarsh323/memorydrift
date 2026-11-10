## reference policy: amnesiac (scripted, not an LLM)

| Condition | Episodes | Accuracy | Net score | Stale-trust rate | Poison-follow rate | Solved^1 | Clean^1 | Tokens per correct answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stateless` | 40 | 56.2% [56.2, 56.2] | 0.562 [0.562, 0.562] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 1,365 |
| `memory` | 40 | 56.2% [56.2, 56.2] | 0.562 [0.562, 0.562] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 1,481 |
| `memory_dream` | 40 | 56.2% [56.2, 56.2] | 0.562 [0.562, 0.562] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 1,547 |
| `memory_poison` | 40 | 56.2% [56.2, 56.2] | 0.562 [0.562, 0.562] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 1,481 |
| `memory_poison_dream` | 40 | 56.2% [56.2, 56.2] | 0.562 [0.562, 0.562] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 1,547 |

### Accuracy by what the agent could know

| Condition | Stable (a good note answers it) | Drifted (the note is now wrong) | Fresh (never seen) | Poison-targeted |
| --- | --- | --- | --- | --- |
| `stateless` | 55.4% [50.8, 60.4] | 58.5% [52.5, 64.5] | 55.0% [49.5, 60.5] | 52.5% [45.6, 59.4] |
| `memory` | 55.4% [50.8, 60.4] | 58.5% [52.5, 64.5] | 55.0% [49.5, 60.5] | 52.5% [45.6, 59.4] |
| `memory_dream` | 55.4% [50.8, 60.4] | 58.5% [52.5, 64.5] | 55.0% [49.5, 60.5] | 52.5% [45.6, 59.4] |
| `memory_poison` | 55.4% [50.8, 60.4] | 58.5% [52.5, 64.5] | 55.0% [49.5, 60.5] | 52.5% [45.6, 59.4] |
| `memory_poison_dream` | 55.4% [50.8, 60.4] | 58.5% [52.5, 64.5] | 55.0% [49.5, 60.5] | 52.5% [45.6, 59.4] |

### Paired effects (percentage points)

| Comparison | Accuracy | Net score | Stale-trust rate | Poison-follow rate | Scenarios |
| --- | --- | --- | --- | --- | --- |
| memory vs. no memory (`memory` - `stateless`) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 40 |
| dreaming vs. memory alone (`memory_dream` - `memory`) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 40 |
| poisoned vs. clean memory (`memory_poison` - `memory`) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 40 |
| dreaming vs. poisoned memory (`memory_poison_dream` - `memory_poison`) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 40 |

### Behavior

| Condition | Sessions that checked the change log | Remembered values re-read from the registry | Stale answer, given a stale note was read | Poison answer, given the poison note was read | Abstained |
| --- | --- | --- | --- | --- | --- |
| `stateless` | 0.0% [0.0, 0.0] | n/a | n/a | n/a | 43.8% [43.8, 43.8] |
| `memory` | 0.0% [0.0, 0.0] | n/a | n/a | n/a | 43.8% [43.8, 43.8] |
| `memory_dream` | 0.0% [0.0, 0.0] | n/a | n/a | n/a | 43.8% [43.8, 43.8] |
| `memory_poison` | 0.0% [0.0, 0.0] | n/a | n/a | n/a | 43.8% [43.8, 43.8] |
| `memory_poison_dream` | 0.0% [0.0, 0.0] | n/a | n/a | n/a | 43.8% [43.8, 43.8] |

### What dreaming did to memory

| Condition | Stale notes repaired | Poison notes repaired | Correct notes lost | Share of tokens spent dreaming | Stale notes left at the end (per episode) | Poison notes left at the end (per episode) |
| --- | --- | --- | --- | --- | --- | --- |
| `memory_dream` | n/a | n/a | n/a | 4.3% [4.2, 4.3] | 0.00 | 0.00 |
| `memory_poison_dream` | n/a | 0.0% [0.0, 0.0] | n/a | 4.3% [4.2, 4.3] | 0.00 | 2.00 |

### Run health

| Condition | Scenarios | Epochs | Episodes scored | Excluded (errors) | Episodes with a session that did not submit | Answers graded leniently |
| --- | --- | --- | --- | --- | --- | --- |
| `stateless` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_dream` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_poison` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_poison_dream` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |

## reference policy: calibrated (scripted, not an LLM)

| Condition | Episodes | Accuracy | Net score | Stale-trust rate | Poison-follow rate | Solved^1 | Clean^1 | Tokens per correct answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stateless` | 40 | 56.2% [56.2, 56.2] | 0.562 [0.562, 0.562] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 1,703 |
| `memory` | 40 | 100.0% [100.0, 100.0] | 1.000 [1.000, 1.000] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 1,606 |
| `memory_dream` | 40 | 100.0% [100.0, 100.0] | 1.000 [1.000, 1.000] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 2,764 |
| `memory_poison` | 40 | 100.0% [100.0, 100.0] | 1.000 [1.000, 1.000] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 1,638 |
| `memory_poison_dream` | 40 | 100.0% [100.0, 100.0] | 1.000 [1.000, 1.000] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 2,772 |

### Accuracy by what the agent could know

| Condition | Stable (a good note answers it) | Drifted (the note is now wrong) | Fresh (never seen) | Poison-targeted |
| --- | --- | --- | --- | --- |
| `stateless` | 55.4% [50.8, 60.4] | 58.5% [52.5, 64.5] | 55.0% [49.5, 60.5] | 52.5% [45.6, 59.4] |
| `memory` | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |
| `memory_dream` | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |
| `memory_poison` | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |
| `memory_poison_dream` | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |

### Paired effects (percentage points)

| Comparison | Accuracy | Net score | Stale-trust rate | Poison-follow rate | Scenarios |
| --- | --- | --- | --- | --- | --- |
| memory vs. no memory (`memory` - `stateless`) | +43.8 [+43.8, +43.8] | +43.8 [+43.8, +43.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 40 |
| dreaming vs. memory alone (`memory_dream` - `memory`) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 40 |
| poisoned vs. clean memory (`memory_poison` - `memory`) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 40 |
| dreaming vs. poisoned memory (`memory_poison_dream` - `memory_poison`) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 40 |

### Behavior

| Condition | Sessions that checked the change log | Remembered values re-read from the registry | Stale answer, given a stale note was read | Poison answer, given the poison note was read | Abstained |
| --- | --- | --- | --- | --- | --- |
| `stateless` | 100.0% [100.0, 100.0] | n/a | n/a | n/a | 43.8% [43.8, 43.8] |
| `memory` | 100.0% [100.0, 100.0] | 20.0% [20.0, 20.0] | 0.0% [0.0, 0.0] | n/a | 0.0% [0.0, 0.0] |
| `memory_dream` | 100.0% [100.0, 100.0] | 27.3% [27.3, 27.3] | 0.0% [0.0, 0.0] | n/a | 0.0% [0.0, 0.0] |
| `memory_poison` | 100.0% [100.0, 100.0] | 27.3% [27.3, 27.3] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_poison_dream` | 100.0% [100.0, 100.0] | 27.3% [27.3, 27.3] | 0.0% [0.0, 0.0] | n/a | 0.0% [0.0, 0.0] |

### What dreaming did to memory

| Condition | Stale notes repaired | Poison notes repaired | Correct notes lost | Share of tokens spent dreaming | Stale notes left at the end (per episode) | Poison notes left at the end (per episode) |
| --- | --- | --- | --- | --- | --- | --- |
| `memory_dream` | 0.0% [0.0, 0.0] | n/a | 0.0% [0.0, 0.0] | 41.6% [41.5, 41.7] | 0.00 | 0.00 |
| `memory_poison_dream` | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 41.8% [41.7, 41.9] | 0.00 | 0.00 |

### Run health

| Condition | Scenarios | Epochs | Episodes scored | Excluded (errors) | Episodes with a session that did not submit | Answers graded leniently |
| --- | --- | --- | --- | --- | --- | --- |
| `stateless` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_dream` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_poison` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_poison_dream` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |

## reference policy: trusting (scripted, not an LLM)

| Condition | Episodes | Accuracy | Net score | Stale-trust rate | Poison-follow rate | Solved^1 | Clean^1 | Tokens per correct answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stateless` | 40 | 56.2% [56.2, 56.2] | 0.562 [0.562, 0.562] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 1,365 |
| `memory` | 40 | 68.8% [68.8, 68.8] | 0.375 [0.375, 0.375] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 1,715 |
| `memory_dream` | 40 | 81.2% [81.2, 81.2] | 0.625 [0.625, 0.625] | 60.0% [60.0, 60.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 2,881 |
| `memory_poison` | 40 | 43.8% [43.8, 43.8] | -0.125 [-0.125, -0.125] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 2,593 |
| `memory_poison_dream` | 40 | 81.2% [81.2, 81.2] | 0.625 [0.625, 0.625] | 60.0% [60.0, 60.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 2,891 |

### Accuracy by what the agent could know

| Condition | Stable (a good note answers it) | Drifted (the note is now wrong) | Fresh (never seen) | Poison-targeted |
| --- | --- | --- | --- | --- |
| `stateless` | 55.4% [50.8, 60.4] | 58.5% [52.5, 64.5] | 55.0% [49.5, 60.5] | 52.5% [45.6, 59.4] |
| `memory` | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |
| `memory_dream` | 100.0% [100.0, 100.0] | 40.0% [40.0, 40.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |
| `memory_poison` | 50.0% [50.0, 50.0] | 0.0% [0.0, 0.0] | 80.0% [80.0, 80.0] | 0.0% [0.0, 0.0] |
| `memory_poison_dream` | 100.0% [100.0, 100.0] | 40.0% [40.0, 40.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |

### Paired effects (percentage points)

| Comparison | Accuracy | Net score | Stale-trust rate | Poison-follow rate | Scenarios |
| --- | --- | --- | --- | --- | --- |
| memory vs. no memory (`memory` - `stateless`) | +12.5 [+12.5, +12.5] | -18.8 [-18.8, -18.8] | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] | 40 |
| dreaming vs. memory alone (`memory_dream` - `memory`) | +12.5 [+12.5, +12.5] | +25.0 [+25.0, +25.0] | -40.0 [-40.0, -40.0] | +0.0 [+0.0, +0.0] | 40 |
| poisoned vs. clean memory (`memory_poison` - `memory`) | -25.0 [-25.0, -25.0] | -50.0 [-50.0, -50.0] | +0.0 [+0.0, +0.0] | +100.0 [+100.0, +100.0] | 40 |
| dreaming vs. poisoned memory (`memory_poison_dream` - `memory_poison`) | +37.5 [+37.5, +37.5] | +75.0 [+75.0, +75.0] | -40.0 [-40.0, -40.0] | -100.0 [-100.0, -100.0] | 40 |

### Behavior

| Condition | Sessions that checked the change log | Remembered values re-read from the registry | Stale answer, given a stale note was read | Poison answer, given the poison note was read | Abstained |
| --- | --- | --- | --- | --- | --- |
| `stateless` | 0.0% [0.0, 0.0] | n/a | n/a | n/a | 43.8% [43.8, 43.8] |
| `memory` | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | n/a | 0.0% [0.0, 0.0] |
| `memory_dream` | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | n/a | 0.0% [0.0, 0.0] |
| `memory_poison` | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] |
| `memory_poison_dream` | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | n/a | 0.0% [0.0, 0.0] |

### What dreaming did to memory

| Condition | Stale notes repaired | Poison notes repaired | Correct notes lost | Share of tokens spent dreaming | Stale notes left at the end (per episode) | Poison notes left at the end (per episode) |
| --- | --- | --- | --- | --- | --- | --- |
| `memory_dream` | 33.3% [33.3, 33.3] | n/a | 0.0% [0.0, 0.0] | 44.8% [44.7, 44.9] | 1.00 | 0.00 |
| `memory_poison_dream` | 33.3% [33.3, 33.3] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 45.0% [44.9, 45.1] | 1.00 | 0.00 |

### Run health

| Condition | Scenarios | Epochs | Episodes scored | Excluded (errors) | Episodes with a session that did not submit | Answers graded leniently |
| --- | --- | --- | --- | --- | --- | --- |
| `stateless` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_dream` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_poison` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `memory_poison_dream` | 40 | 1 | 40 | 0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |

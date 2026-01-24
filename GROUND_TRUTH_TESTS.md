# Ground Truth Checks

This repo includes a lightweight script for validating area calculations
against known ground truth values.

## How to use

1. Add entries to `ground_truth_samples.json`:

```
[
  {
    "pdf_path": "samples/plan_a.pdf",
    "expected_total_area_m2": 120.5,
    "tolerance_pct": 8
  }
]
```

2. Run:

```
python validate_ground_truth.py
```

The script will:
- Run the normal analysis pipeline.
- Use room segmentation when available.
- Fallback to flooring-area estimation if segmentation fails.
- Report error percent vs expected value.

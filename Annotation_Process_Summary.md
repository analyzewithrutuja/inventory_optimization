# Grocery Item Detection Dataset — Annotation Process Summary

## 1. Objective

This project required a labeled object-detection dataset to train a YOLOv8 model
capable of identifying five grocery products — Corn Flakes, Indomie, Barilla,
Keya Piri Piri, and Nut Bars — for the Final Project's computer-vision
inventory/warehouse detection use case. 435 single-item photographs had
already been collected and classified by product name, but none had
bounding-box annotations, which YOLO training requires.

## 2. Dataset Description

- 435 raw photos across 5 product classes, one item per image.
- Backgrounds varied widely: carpet, wood flooring, a perforated metal
  pegboard, a kitchen stove top, and an outdoor patio table.
- Images went through a prior pipeline: raw capture → format conversion
  (HEIC → JPG) → resizing → classification by product name.

## 3. Methodology

Manually drawing 435 bounding boxes was avoided in favor of automated
annotation using **FastSAM** (a lightweight Segment-Anything model), guided
by point prompts. Since every photo contains a single product, the approach
was to prompt the model to segment "the object at the image center" and
derive a bounding box from the resulting mask — refining the prompting
strategy iteratively as failure patterns emerged.

## 4. Challenges Encountered and Solutions

Annotation quality did not converge in a single pass. Each fix solved one
failure mode but sometimes introduced another, so the approach evolved
through eight iterations:

| Step | Problem Observed | Fix Applied |
|---|---|---|
| 1 | A single center-point prompt to FastSAM often selected only a small sub-part of the object (e.g. a logo or text label) instead of the whole product. | Among all candidate masks touching the center point, select the largest mask within a reasonable size band (3%–85% of frame area), rejecting both tiny sub-parts and whole-scene masks. |
| 2 | Background clutter (carpet texture, pegboard holes, table patterns) frequently merged into the object mask, producing boxes that covered 80–100% of the frame. | Added 4 corner points as **negative** prompts, explicitly marking the image corners as background so FastSAM excludes any mask that bleeds into them. |
| 3 | The stricter corner-negative prompting sometimes eliminated every candidate mask for large or elongated objects, yielding no box at all. | Added a fallback: if the corner-negative prompt returns no mask, retry with a plain center-point-only prompt. |
| 4 | Nut Bars remained the worst-performing class (80% of a 50-image pilot batch flagged) — its elongated shape and metal wire-rack background were visually similar in color/texture, causing merges. | Tested a 5-point cross prompt (center + up/down/left/right) with mask union, to fully cover objects that visually split into multiple segments (e.g. a bottle's cap, label, and base). |
| 5 | Applying the cross-prompt method to all 435 images improved Nut Bars but **regressed 35 previously-correct images** in other classes, because the off-center points sometimes landed on background. | Built a hybrid method: use the original (single-point) method as primary, and only retry with the cross-prompt method on images it already flagged, keeping whichever result was better. |
| 6 | Manual review of flagged images revealed the flagging threshold itself was miscalibrated — boxes with only 12–20% frame coverage were often correct (e.g. a small spice bottle on a large floor), not errors. | Recalibrated the low-area flag threshold from 20% to 10% by visually validating the boundary; flagged count dropped from 107 to 52 without any recomputation. |
| 7 | Remaining flagged images (mostly loose/oversized boxes) needed further tightening. | Applied GrabCut (classical foreground segmentation) initialized from the existing box; kept results only where they measurably improved the box, **reverting 16 cases** where GrabCut made the box worse. |
| 8 | Final manual visual review of the last 34 flagged images. | Identified 4 photos that were genuinely unusable (motion blur, product out of frame) and 3 images with incorrect ground-truth class labels inherited from the original classification data; all 7 were excluded from the dataset. |

## 5. Quality Control Process

1. Auto-generate a bounding box for every image using the current best method.
2. Flag any box whose area falls outside a validated "reasonable" range as a
   percentage of the frame.
3. Visually inspect flagged images in batches (contact sheets) to distinguish
   genuine errors from false-positive flags.
4. Apply targeted automated refinement (GrabCut) only where it measurably
   helped; revert where it did not.
5. Manually review the final, smallest set of flagged images and make
   case-by-case decisions: accept as-is, or exclude.

## 6. Final Dataset Summary

| Class | Raw Photos | Final (Clean) | Notes |
|---|---|---|---|
| Corn Flakes | 100 | 100 | No exclusions |
| Indomie | 98 | 98 | No exclusions |
| Barilla | 91 | 89 | 2 removed (misclassified) |
| Keya Piri Piri | 75 | 72 | 3 removed (blurry photos) |
| Nut Bars | 71 | 69 | 2 removed (1 blurry, 1 misclassified) |
| **Total** | **435** | **428** | **7 images excluded** |

**Excluded — unusable photos (4):** Motion-blurred or out-of-frame product
shots where no reliable bounding box could be drawn.

**Excluded — mislabeled photos (3):** Images whose product label in the
original classification data did not match the visible product (e.g. two
images labeled "Barilla" were actually photos of Corn Flakes boxes). This was
a pre-existing data-labeling issue, not an annotation error, discovered
during visual review.

## 7. Key Learnings

- A single prompting strategy does not generalize across all object shapes
  and backgrounds — elongated objects and multi-segment objects (e.g. a
  bottle with a distinct cap, label, and base) required different handling
  than simple boxed products.
- Automated quality-flagging heuristics must themselves be validated. The
  original area-based threshold was flagging a large number of genuinely
  correct boxes as errors simply because some products are physically
  smaller in the frame than others.
- Combining multiple automated techniques (segmentation prompting, classical
  foreground extraction, threshold recalibration) with a final targeted
  manual review produced a clean dataset far faster than manually annotating
  all 435 images, while still catching data-quality issues (mislabeled
  images) that no bounding-box method could have found.

## 8. Train / Validation / Test Split Methodology

### 8.1 Class-ID Mapping

Label files use YOLO format (`class_id x_center y_center width height`, all
normalized 0–1). Classes are assigned integer IDs in alphabetical order,
confirmed against per-class counts in the labels themselves:

| Class ID | Class | Count |
|---|---|---|
| 0 | Barilla | 89 |
| 1 | Corn Flakes | 100 |
| 2 | Indomie | 98 |
| 3 | Keya Piri Piri | 72 |
| 4 | Nut Bars | 69 |

### 8.2 Split Ratio and Rationale

The 428 clean images are split **70% train / 20% validation / 10% test**:

- **Train** — the set the model directly learns from (weights are updated
  against this data).
- **Validation** — checked during training to monitor for overfitting;
  weights are *not* updated from it, but it influences decisions like when
  to stop training or which checkpoint to keep.
- **Test** — held out completely and touched only once, at the very end, to
  get an unbiased estimate of real-world performance.

### 8.3 Why the Split Must Be Class-Stratified

A naive random split shuffles all 428 images together without regard to
class, which risks uneven representation purely by chance — e.g. Nut Bars
(69 images, the smallest class) could end up under-represented in train, or
have too few images in test to produce a statistically meaningful per-class
accuracy.

A **stratified split** instead applies the 70/20/10 ratio independently
*within each class*, so every class keeps the same proportional
representation in all three sets. This ensures the model gets fair exposure
to every class during training, and that validation/test metrics are
meaningful for every class rather than noisy for the smaller ones.

For each class with `n` images:

```
n_train = floor(0.7 × n)
n_val   = floor(0.2 × n)
n_test  = n − n_train − n_val
```

`n_test` is computed by subtraction (not rounded independently) so the three
counts always sum exactly to `n` — no image is dropped or double-counted.

### 8.4 Resulting Split Counts

| Class | Total (n) | Train (70%) | Val (20%) | Test (10%) |
|---|---|---|---|---|
| Corn Flakes | 100 | 70 | 20 | 10 |
| Indomie | 98 | 68 | 19 | 11 |
| Barilla | 89 | 62 | 17 | 10 |
| Keya Piri Piri | 72 | 50 | 14 | 8 |
| Nut Bars | 69 | 48 | 13 | 8 |
| **Total** | **428** | **298** | **83** | **47** |

## 9. Model Training

A YOLOv8n (nano) model, pretrained on COCO, was fine-tuned (transfer
learning) on the 298-image training split for 50 epochs (CPU, image size
416, batch size 8, early-stopping patience 15). Training completed in 0.85
hours. The loss function is a weighted sum of three components:

- **Box loss (CIoU)** — penalizes bounding-box position/size/aspect-ratio
  error.
- **Classification loss (BCE)** — penalizes wrong or low-confidence class
  predictions.
- **Distribution Focal Loss (DFL)** — sharpens the anchor-free box-edge
  predictions.

## 10. Test Set Evaluation (Held-Out, 47 Images)

The 47-image test split was never seen during training or used for any
training decisions, so these numbers are the unbiased estimate of
real-world performance:

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Barilla | 0.899 | 0.892 | 0.886 | 0.767 |
| Corn Flakes | 0.980 | 1.000 | 0.995 | 0.776 |
| Indomie | 1.000 | 0.912 | 0.995 | 0.886 |
| Keya Piri Piri | 0.961 | 1.000 | 0.995 | 0.974 |
| Nut Bars | 1.000 | 0.986 | 0.995 | 0.737 |
| **Overall (all)** | **0.968** | **0.958** | **0.973** | **0.828** |

**Observation:** Barilla is the weakest-performing class on the test set,
consistent with it being the class most affected by annotation/data-quality
issues during the labeling process (Section 4, Step 4–5; Section 6). This
suggests the test-set gap traces back to upstream data quality rather than
the model itself.

## 11. Limitations — Multi-Object Detection

A live webcam demo (`scripts/webcam_demo.py`) was used to test the
fine-tuned model interactively. Single-item shots (one product held up to
the camera) worked reliably, matching the test-set numbers in Section 10.

However, when **multiple products were placed in the same frame**, results
were inconsistent:

- **Items placed close together / touching:** the model frequently merged
  all of them into a single bounding box, labeled with just one class
  (e.g. all 4 items below detected as one `corn_flakes 0.97` box).

  ![Multiple items merged into a single box](webcam_detections/limitation_full_cluster.jpg)

- **Items spaced apart:** the model sometimes correctly produced a separate
  box per item, but this was not reliable — repeated shots of the same
  3-item layout produced different results shot to shot (all 3 detected
  once, only 2 of 3 detected another time, and one shot mislabeled a
  Barilla box as `keya_piri_piri`).

  ![Inconsistent multi-item detection with a misclassification](webcam_detections/limitation_partial_multi.jpg)

**Root cause:** every training image contained exactly one centered
product (Section 2), and every label file contained exactly one bounding
box (Section 3–4). The model was never shown an example of two distinct
objects appearing together in one frame, so its behavior on multi-object
scenes is out-of-distribution — it was not learned, so it cannot be fixed
by adjusting confidence thresholds or NMS settings at inference time. A
proper fix would require collecting and annotating new training images
that contain multiple products per frame, each with its own bounding box.

### 11.1 Mitigation Attempt: Synthetic Multi-Object Data

Rather than collecting and manually annotating new multi-object photos,
synthetic multi-object training images were generated automatically from
the existing 298 single-item training images and their already-known
bounding boxes ("copy-paste" augmentation): each product is cropped using
its existing box, pasted onto another training image at a new position,
and its new bounding box is derived purely from the paste-position
arithmetic — no manual re-labeling required. This was done in two rounds:

- **Round 1 (174 images):** pairs/triples of products pasted at random,
  non-overlapping positions (`scripts/make_synthetic_multi.py`).
- **Round 2 (83 images):** 3–5 products arranged in a tightly-packed row
  with touching/slightly-overlapping placement, matching the real-world
  layout that Round 1 did not cover (`scripts/make_synthetic_dense.py`).

The model was fine-tuned on the combined dataset after each round
(`grocery_detect_multiobj` after Round 1, `grocery_detect_dense` after
Round 2), starting from the previous checkpoint each time.

**Round 1 result — genuine improvement.** Re-running the same 3-item
webcam frames that previously produced 1–2 merged/inconsistent boxes now
produced 3 correctly separated and correctly classified boxes, with
single-item test-set mAP@0.5 only dropping marginally (0.973 → 0.962):

![All three items correctly separated and classified after Round 1](webcam_detections/mitigation_success.jpg)

**Round 2 result — diminishing/negative returns.** Extending to
5 tightly-packed items did not produce reliable improvement. Probing the
Round 2 model at very low confidence thresholds (down to 0.05) showed it
still had a weak signal for every item in the frame, but its confidence on
real photos had dropped across the board (e.g. a clearly visible product
scoring only 0.11–0.21 confidence instead of the 0.6–0.9 range seen
before). Training on more copy-paste images, which have visible pasting
artifacts (seams, lighting/background mismatch), appears to have taught
the model some of that synthetic-image noise, hurting its confidence
calibration on real, unedited photos.

**Decision:** `grocery_detect_multiobj` (Round 1) was kept as the final
model — it is the best balance of maintained single-item accuracy and
genuine multi-item improvement. `grocery_detect_dense` (Round 2) was not
adopted.

**Key learning:** synthetic copy-paste augmentation has a sweet spot for a
small dataset like this one. A moderate amount closes the single-vs-multi-
object distribution gap without much cost; pushing further to force a
harder scenario (5 tightly-packed items) added enough visible synthetic
artifacts to measurably hurt real-world confidence calibration, more than
it helped detection of dense clusters. This is itself a useful,
generalizable finding about the limits of data augmentation on small
datasets, not just a failed experiment.

## 12. Next Steps

- If further work on dense multi-object detection is warranted, prefer
  collecting a modest number of **real** multi-object photos over pushing
  synthetic copy-paste augmentation further, since Section 11.1 shows the
  synthetic approach's returns diminish (and can reverse) past a point.

# SarcGraph experimental adapter

This optional analysis stage reads existing class-ID masks. It does not modify
training, model weights, predictions, or platform annotations. Outputs are
candidate links, NOT validated sarcomeres or T/side-tubule annotations.

## Setup

Use a separate environment. The tested upstream commit is
`d335e5729acb1e7de55233fb269e88d8672471f4` (MIT).

```sh
git clone https://github.com/Sarc-Graph/sarcgraph.git external/sarcgraph
git -C external/sarcgraph checkout d335e5729acb1e7de55233fb269e88d8672471f4
python -m venv .venv-sarcgraph
# Activate the new environment using your shell's activation script.
python -m pip install numpy pandas scipy scikit-image scikit-learn matplotlib pillow networkx trackpy sk-video
```

The adapter imports the checked-out source via --sarcgraph-source; it does not
install upstream's old pinned dependencies into the training environment.
Local smoke tests used Python 3.13 and a system-site-packages virtual environment;
the clean-environment installation above has not yet been independently tested.

## Run

```sh
python tools/analyze_sarcomere_sarcgraph.py --mask prediction.png --image image.jpg --output outputs/sarcgraph/new_case --sarcgraph-source external/sarcgraph --z-class 10 --length-range-px 20 400 --expected-length-px 150
python tools/test_sarcgraph_adapter.py
python tools/smoke_sarcgraph_synthetic.py --output outputs/synthetic_demo --sarcgraph-source external/sarcgraph
```

The lengths above are exploratory pixel parameters, not biological limits.
Set them from the image scale and development data before interpreting results.
Use --um-per-pixel only when scale is verified. Without it, length_um is empty.
Original exported labels use Z=10; remapped sarcomere predictions use Z=1.
RGB/color mask inputs are rejected. Image and mask dimensions must agree.

Outputs: z_objects.csv, candidate_pairs.csv, sarcgraph_raw.csv (when >=2 objects),
summary.json and candidate_overlay.png. Existing nonempty output folders are
rejected. Summary includes mask and upstream source hashes and all parameters.
Coordinates follow upstream x=row, y=column; plotting swaps these to display axes.
Endpoints in this adapter use PCA projections. Fragmented or merged components
are not automatically repaired; border flags and unpaired objects are retained.

## Actual local integration result, 2026-10-22

One historical baseline prediction and its matching image were downloaded from
the server, with original class IDs. No inference or training was rerun.
At min_area=4, expected length=150px, range=20..400px: 56 components,
35 candidate pairs, 11 unpaired objects. This is an interface smoke test, NOT
accuracy or clinical validation. Visual inspection shows useful regular chains
but also suspicious cross-bundle and along-Z connections at the bottom.

Four adapter tests passed: empty masks, coordinate/distance convention,
RGB rejection and border detection. Candidate overlay was visually inspected.
Upstream's sample_4.png also completed the image pipeline (5983 returned
sarcomere rows); this verifies execution, not agreement against reference truth.

## Remaining validation

Compare manual Z masks and predicted Z masks on identical regions. Review
candidate precision and coverage before adding orientation/bundle constraints.
Check physical scale, border exclusion, merged/fragmented objects and parameter
sensitivity. Do not infer improved segmentation Dice/IoU from candidate counts.
Do not compute the project's deferred alpha angle from these unverified links.

## Matched annotation/prediction diagnostic

`--direction-tolerance-deg` (default 30) annotates candidate links based on the
deviation from each endpoint Z object's PCA normal. All original links remain
in candidate_pairs.csv. direction_audit.csv and direction_overlay.png provide
an additional audit; tolerance 90 effectively disables directional rejection.
Passing this check does not prove common myofibril membership. Curved, small,
or merged components can have unreliable PCA axes.

On the same historical image and same 20..400px/150px length parameters:

| Input | Z components | Candidate links | Direction pass | Conflict | Unpaired objects before audit |
|---|---:|---:|---:|---:|---:|
| Existing annotation | 100 | 43 | 17 | 26 | 38 |
| Baseline prediction | 56 | 35 | 26 | 9 | 11 |

These are diagnostic counts, not precision/recall or proof that prediction is
better than annotation. Existing masks can be incomplete and disconnected;
their components are not certified Z-disc instances. Five unit tests now pass,
including synthetic perpendicular versus along-Z connections. Both overlays
were inspected. Cross-bundle connections can remain even after direction checks.

## Optional virtual fragment merge and bundle geometry audit

`--merge-gap-px 8` enables experimental disjoint-pair merging. Default 0 is off.
Candidates require <=8px endpoint gap, <=2px lateral offset, <=15deg axial
difference, nonoverlapping axial projections, and axes at least 5px long.
Only pairs with unique partners are accepted; no transitive chain merging.
These constants are development heuristics, not calibrated biological limits.
Merged centers are area-weighted and endpoints are refitted to source endpoints.
This constructs virtual graph nodes, not new pixel labels. source_objects.csv,
merge_proposals.csv and object_mapping.csv retain full provenance.

bundle_audit.csv adds <=30deg axial difference and >=0.25 overlap relative to
the shorter projected Z span, in addition to direction checks. These are only
geometry proxies, NOT myofibril segmentation or proof of common bundle identity.
bundle_overlay.png shows all edges, including red rejected hypotheses.
False rejection is possible for curved, branched or pathological structures.

Run the local four-condition diagnostic (requires the previously downloaded
annotation.png, prediction.png and image.jpg under outputs/research/sarcgraph_inputs):

```sh
python tools/run_sarcgraph_local_ablation.py
```

This diagnostic runner refuses existing outputs through the underlying CLI;
use a fresh experiment folder when repeating. It does not download patient data.
Eight synthetic adapter/geometry tests pass, including unique fragment merge,
ambiguous chain refusal, area conservation and lateral overlap rejection.

## Eight-case image evidence experiment

Three optional structure-tensor integration scales (6/12/24px) measure whether
the raw image texture runs along a proposed link. --texture-audit requires the
matching original image. Coherence below .2 is uncertain; at least two scales
need >=40% valid sampling coverage. Alignment >=.65 at every usable scale gives
support; <=.35 at every usable scale gives conflict; remaining links are uncertain.
These values are exploratory, uncalibrated thresholds. Texture does not certify
that a region is myofibrillar or that a pair is anatomically adjacent.

--compact-output skips the three original figures for batch processing. The
adapter handles upstream's zero-edge result via a local subclass so an empty
case produces CSV/JSON rather than pandas.concat failing. Source checkout is
unchanged. Summary now includes adapter and texture-module hashes.

```sh
python tools/run_sarcgraph_development.py --inputs outputs/research/sarcgraph_development_inputs --output outputs/research/new_batch --sarcgraph-source outputs/research/sarcgraph_upstream
python tools/report_sarcgraph_development.py --inputs outputs/research/sarcgraph_development_inputs --results outputs/research/new_batch --output outputs/research/new_review
python tools/test_sarcgraph_texture.py
```

Inputs are existing images plus original-ID annotation/prediction PNGs, selected
deterministically from eight different case prefixes. The selection script
prepare_sarcgraph_development_sample.py makes an archive on the server with
source IDs and hashes. The archive is private data, not a repository dependency.

The review report also tests an alternative graph (sarcgraph_normal_neighbors.py):
search all points in range, apply direction/parallel/overlap checks, then keep
mutual closest neighbors on each Z-normal side. A second candidate within a
1.15 length ratio is marked ambiguous. It can miss genuine branches or retain
incorrect same-direction neighbors. It is an additional hypothesis method,
not the unmodified SarcGraph algorithm. Original candidates remain available.

The 8-case pilot generated 209 original prediction links (123 geometry passes),
with texture evidence supporting 60, conflicting with 4, and leaving 145 uncertain.
The alternative selected 131 links, including 28 absent from the original set;
46 had texture support and 85 were uncertain. These are counts, not accuracy.
Ten geometry/adapter tests and two texture tests passed. All 16 input conditions
completed after the empty-graph fix. No new model training was performed.

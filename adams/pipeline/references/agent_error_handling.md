# Agent Error Handling (Procedure and Report Format)

This document is for **agents** (stage, workflow, executive). It defines the error-handling procedure and report format. Handle errors at the lowest level; propagate upward only when the current level cannot resolve.

## Principle

**Handle at the lowest level; propagate only when the current level cannot resolve.** Stage agents try to fix (read docs, adjust parameters, retry with materially different fixes as appropriate). Workflow uses meta_analysis to check resume/similar errors before propagating. Executive presents only when workflow could not resolve.

## Error Types (Pipeline and General Runtime)

**Pipeline / scientific errors** (classified in code):

- **PerLigandError / PerPoseError**: Affect a single ligand or MD pose. The pipeline skips the affected item and continues (e.g. invalid SMILES, single docking failure, single pose MD failure). When such errors surface from a tool, you may still need to report if the overall step failed or too many items failed.
- **FatalError**: Missing files, invalid configuration, missing executables. Stops the pipeline. When you see these from a tool, report and propagate; do not retry with the same inputs.

**Other runtime errors** (handle with the same procedure):

- **Transient failures**: Timeouts, temporary I/O or network issues, resource exhaustion (e.g. GPU OOM). Try again with the same or adjusted parameters (e.g. reduce batch size, wait and retry) before propagating.
- **Environment / setup**: Missing executable, wrong path, permission denied, conda/env not activated. Fix path or configuration if you can infer it; otherwise report with clear cause.
- **Input / data**: Bad file format, corrupted file, wrong column names. Correct the input or re-run the prior step that produces it; if not fixable at this stage, report.
- **Unclassified / unexpected**: Any exception or error message you do not recognize. Still follow the procedure: read docs and stage-specific notes, try a materially different fix (parameters, paths, order), and only propagate when certain you cannot fix.

Use error type and message to decide whether to retry (with materially different fixes), fix configuration, or report and propagate immediately.

## Stage Procedure (When a Tool Returns an Error)

1. **Read this doc** (and stage-specific error notes in your prompt). Do **not** read it after successful calls.
2. **Try to resolve**: Use the error message and docs to fix (e.g. valid parameters, correct input paths, correct step order). Retry with **materially different** fixes as appropriate (e.g. different parameter set, corrected input, reduced load for resource errors). Do not repeat the same failing call unchanged.
3. **If certain you cannot fix**: Produce a structured **error report** (see template below) and return it to the caller (workflow). Then stop retrying at this level.

## Error Report Template

When propagating a failure, always include:

1. **Output folder**: The exact path being used (e.g. "Output folder: /path/to/outputs/run_xxx")
2. **Steps completed**: Which pipeline steps finished successfully before the error
3. **Step that failed**: Which step encountered the error
4. **Error details**: The actual error message and likely cause
5. **Entry point for resume**: Which entry point to use if resuming

Use this format:

```
ERROR OCCURRED during [step_name]

Run Context:
- Output folder: /full/path/to/output_folder
- Log file: /full/path/to/adams_pipeline.log
- Steps completed: preprocessing, docking
- Step failed: md_analysis

Error Details:
[error message]

Likely Cause:
[brief explanation]

To Resume:
- Entry point: md_protein_topology (or appropriate entry point)
- Required files: [list paths that will be needed]
```

## Escalation

- **Stage agent** → On tool error: try to fix (read docs, retry with materially different fixes as appropriate). If certain cannot fix → error report → return to **workflow agent**.
- **Workflow agent** → On stage failure: call **meta_analysis_agent** to check resume possibility and whether a similar error occurred before. If meta_analysis suggests a viable fix (e.g. resume from entry point), attempt it. If still unfixable after retries → return failure with full context to **executive agent**.
- **Executive agent** → When workflow returns a failure, present the failure and options to the user (workflow has already attempted resolution including meta_analysis). Executive may also use meta_analysis for cross-run or inconsistent behavior (e.g. same inputs differing across runs).

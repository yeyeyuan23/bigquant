# Submission artifacts

This directory only maintains the current M-route submission package.

- `m_l5/`: five-level M submission artifacts, manifest, validation record,
  checkpoint source, and AIStudio handoff.

Build or refresh the M package with:

```bash
python scripts/build_unified_m_l5_submission.py --help
```

For AIStudio upload and prefix/look-ahead validation, follow
`m_l5/aistudio_handoff.md` and upload exactly the files declared by
`m_l5/submission_bundle_manifest.json`.

S/I/T submission artifacts and builders are intentionally not maintained.

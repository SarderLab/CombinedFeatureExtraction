#!/usr/bin/env bash

# retire-girder-dependency: no more slicer_cli_web.cli_list_entrypoint (which used to route between
# this image's two CLIs — ExpandedGranularFeatures and ClassicalFeatures — by argv, since
# api/girder.py's old run_feature_extraction_pipeline dispatched them as separate Girder jobs with
# separate CLI names). The new dispatcher passes a TYPE env var instead of a CLI-name argv, so route
# on that: Feature_Pipeline/Test_Run -> ExpandedGranularFeatures, Extended_Clinical -> ClassicalFeatures.

case "$TYPE" in
  Feature_Pipeline|Test_Run)
    exec python ExpandedGranularFeatures/ExpandedGranularFeatures.py
    ;;
  Extended_Clinical)
    exec python ClassicalFeatures/ClassicalFeatures.py
    ;;
  *)
    echo "Unknown TYPE '$TYPE' — expected Feature_Pipeline, Test_Run, or Extended_Clinical" >&2
    exit 1
    ;;
esac

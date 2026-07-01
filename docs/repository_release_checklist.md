# Repository Release Checklist

Use this checklist before citing the repository in a formal journal submission.

- [ ] License approved by all code owners.
- [ ] Raw `.fbs` records are excluded unless the data owner explicitly approves release.
- [ ] Restricted derived data are excluded or access-controlled.
- [ ] Synthetic demo and user-owned CSV workflow run successfully.
- [ ] `pytest -q` passes locally or in GitHub Actions.
- [ ] README states the manuscript claim boundary: proxy monitoring, not validated queue-length estimation.
- [ ] Data availability wording matches data-owner approval.
- [ ] A release tag or commit SHA is recorded in the manuscript if required.

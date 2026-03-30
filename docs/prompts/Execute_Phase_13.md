# You are working inside the repository:

`C:\Users\ORENS\predictive-log-anomaly-engine-v3`

There is an existing LOCAL runtime UI served by the real application.
There is also an existing STATIC showcase site for the same project.

Your task is to upgrade the LOCAL runtime UI so it feels more polished, more interactive, and more visually aligned with the static showcase site — while still remaining the real live interface connected to the actual backend and actual data.

Important:

* This is NOT a rewrite from scratch
* Do NOT replace the local UI with the static site
* Do NOT break real runtime functionality
* Do NOT invent backend capabilities that do not exist
* Do NOT hardcode fake states or fake live behavior
* Keep the implementation lightweight and maintainable

Primary goal:
Make the local UI feel like the live operational version of the same design language used by the static showcase.

The local UI should become:

* more visually polished
* more consistent
* more readable
* more presentation-friendly
* slightly more interactive in UX
* but still honest, real, and backend-driven

What to do:

1. Scan the repository and identify:

   * the local runtime UI files
   * the static showcase files that should serve as visual reference
   * the templates, styles, scripts, and UI components involved

2. Compare the LOCAL UI against the STATIC site in terms of:

   * layout
   * spacing
   * typography
   * cards/panels
   * tabs/navigation
   * badges/status indicators
   * color accents
   * section hierarchy
   * labels and headings
   * empty/loading/error states
   * JSON/raw data display areas
   * alerts/health/metrics presentation

3. Identify the exact local UI files that should be updated and explain why

4. Then upgrade the LOCAL UI carefully so it:

   * visually aligns with the static showcase where practical
   * remains connected to real routes and real responses
   * becomes cleaner and more interactive without turning into a fake static page

Allowed improvements:

* better visual hierarchy
* cleaner tab styling
* more polished status cards
* improved badges and labels
* cleaner component status presentation
* improved health / alerts / metrics sections
* collapsible raw JSON areas where useful
* clearer empty states / loading states / error states
* more readable spacing and layout
* subtle UI enhancements that improve demo quality without increasing fragility

Do NOT:

* hardcode fake metrics
* hardcode fake alerts
* fake system health
* add animations that misrepresent reality
* remove useful technical information only for aesthetics
* overcomplicate the frontend with fragile logic
* turn the local UI into a copy-paste of the static site

Required behavior:
Before editing:

1. identify the local UI files involved
2. identify the static site files used as design reference
3. list the exact files you plan to change
4. briefly explain the intended alignment approach

Then:

* perform the UI updates carefully
* preserve compatibility with current routes, backend responses, and real runtime flow
* keep the UI honest and live-data-driven

Deliverables:

1. updated local UI files in place
2. a report file:
   `docs/LOCAL_UI_LIVE_ALIGNMENT_REPORT.md`

The report must include:

* Executive summary
* Local UI files inspected
* Static reference files inspected
* Files changed
* What was visually outdated
* What was improved
* What was intentionally left unchanged
* Any backend constraints affecting UI polish
* Any remaining minor UI issues
* Validation summary

Style goals:

* professional
* modern
* clean
* dashboard-friendly
* visually consistent with the static showcase
* still clearly the real operational interface

Extra expectation:
The final local UI should feel like:
“the live interactive operational dashboard”
while the static site remains:
“the polished presentation/showcase companion”

Do not rebuild everything.
Upgrade carefully, realistically, and with minimal unnecessary churn.

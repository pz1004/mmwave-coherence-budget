#!/usr/bin/env bash
# Stage 5 -- regenerate the manuscript from the run artifacts.
# Every number in the paper is read from stage 1-4 outputs; nothing is typed in.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../paper"
python3 ../src/scheduler/protocol.py   # the protocol record the paper cites must verify
python3 doppler.py      # coherence-time -> velocity mapping
python3 figures_pdf.py  # vector figures at IEEE column widths -> figures/*.pdf
python3 make_draft.py   # -> build/draft.html
python3 to_latex.py     # -> manuscript.tex + supplementary.tex (IEEEtran)

# A float taller than its page, or an overfull line or page, loses or misplaces text, and LaTeX
# reports it only in its log, which tectonic does not print by default.  The engine's output is
# read, and any such warning fails the build.
texbuild() {
  local out
  out=$(tectonic --print "$1" 2>&1) || { printf '%s\n' "$out"; return 1; }
  if printf '%s\n' "$out" | grep -E 'Float too large|Overfull \\[hv]box'; then
    echo "  !! $1: a float or a box overflowed (above)"
    return 1
  fi
}
texbuild manuscript.tex # -> manuscript.pdf

# TWC's initial-submission limit is 13 double-column pages, references
# included.  An overrun warns on every build and fails a SUBMISSION=1 build.
PAGES=$(pdfinfo manuscript.pdf | awk '/^Pages:/{print $2}')
echo "manuscript.pdf: ${PAGES} pages (TWC initial-submission limit: 13)"
if [ "${PAGES}" -gt 13 ]; then
  echo "  !! over the 13-page limit"
  if [ -n "${SUBMISSION:-}" ]; then exit 1; fi
fi

# The supplementary carries the appendices the body cites by letter -- the
# proofs, the verification tables, the pre-registered three-seed analysis and
# the per-stratum, per-seed and sensitivity results.  It is a submitted document, so it is
# built and checked like the manuscript rather than left as an uncompiled fragment.
texbuild supplementary.tex
# and every typeset table row must reach its PDF (to_latex.check_pdf_tables)
python3 to_latex.py --check-pdf

# Before submitting, build with SUBMISSION=1: it turns unfilled placeholders in
# frontmatter.tex (the code-availability <URL>) from a warning into a build
# failure, so the placeholder cannot reach a referee.
#   SUBMISSION=1 ./scripts/05_paper.sh
#
# Author biographies are OFF by default: they are optional at initial submission
# and with the photographs they push the paper from 13 pages to 14, over TWC's
# hard 13-page limit.  For the accepted version:
#   INCLUDE_BIOS=1 python3 to_latex.py && tectonic manuscript.tex

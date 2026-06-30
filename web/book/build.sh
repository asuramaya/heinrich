#!/bin/bash
# Render the evolving book (.tex) → a styled web reading page at /book.
# Re-run whenever the .tex changes (the CI workflow does this on push).
set -e
cd "$(dirname "$0")/.."   # → web/

PANDOC="${PANDOC:-pandoc}"
SRC="../paper/grand_unified_shart_theory_book_draft_finale_polished.tex"
TMP="$(mktemp)"

# pandoc's LaTeX reader can't parse tabularx / the custom Y(=X) column / >{} specs,
# so those tables leak as raw "&"-separated text. Rewrite every table env to a
# plain tabular with l×(column count) — pandoc renders that as a real table.
perl -0777 -pe '
  s/\\begin\{tabularx\}\{\\textwidth\}/\\begin{tabular}/g;
  s/\\end\{tabularx\}/\\end{tabular}/g;
  s/(\\begin\{tabular\})\s*(\{(?:[^{}]++|(?2))*\})/$1 . "{" . ("l" x cols($2)) . "}"/ge;
  BEGIN {
    sub cols {
      my $c = shift;
      $c =~ s/[<>]\{(?:[^{}]++|\{[^{}]*\})*\}//g;   # drop >{} / <{} modifiers
      my $n = () = $c =~ /[pmb]\{/g;                # p/m/b{} columns
      $c =~ s/[pmb]\{(?:[^{}]++|\{[^{}]*\})*\}//g;
      $n += ($c =~ tr/lcrXY//);                     # bare column letters (incl. X/Y)
      return $n < 1 ? 2 : $n;
    }
  }
' "$SRC" > "$TMP"

mkdir -p public/book
"$PANDOC" "$TMP" \
  -f latex -t html5 \
  --standalone --section-divs \
  --table-of-contents --toc-depth=2 \
  --mathjax \
  --metadata title="The Grand Unified Theory of Sharts" \
  --template=book/template.html \
  -o public/book/index.html

rm -f "$TMP"
echo "built public/book/index.html ($(wc -c < public/book/index.html) bytes)"

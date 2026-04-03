.PHONY: all clean data

all: data heinrich_v2.pdf

data:
	cd .. && .venv/bin/python scripts/paper_data.py

heinrich_v2.pdf: heinrich_v2.tex db_numbers.tex
	xelatex heinrich_v2.tex
	xelatex heinrich_v2.tex  # twice for TOC

clean:
	rm -f *.aux *.log *.toc *.out *.pdf db_numbers.tex

# LaTeX paper draft

The paper source is `main.tex`. It is wired to the current figures and tables
under `../results/paper_assets/`, so regenerated paper assets are picked up
without copying files.

Compile from this directory with:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The second pass resolves figure, table, equation, and appendix references.

Before submission:

1. replace the author and institution placeholders;
2. add a venue-specific document class;
3. review Related Work and the existing bibliography;
4. decide whether the full-horizon tables remain in the appendix or move to a
   separate supplementary document;
5. verify terminology and significant digits against the final frozen results.

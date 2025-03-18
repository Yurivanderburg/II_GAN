run:
	pdflatex main.tex # run the tex file
	bibtex main.aux
	pdflatex main.tex # run the tex file
	pdflatex main.tex # run the tex file
	@rm -r main.aux main.bbl main.blg main.log main.out
	
	
redblue:
	latexdiff --flatten ../main.tex main.tex > redblue.tex
	pdflatex redblue
	pdflatex redblue
clean:
	rm redblue.*

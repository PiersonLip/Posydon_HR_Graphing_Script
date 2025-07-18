# [Example](https://piersonlip.github.io/Posydon_HR_Graphing_Script/)

# A script for generating paper-ready HR diagrams of POSYDON datasets.

## Intro  
While I was working on various papers and projects, I found myself constantly generating HR diagrams of various POSYDON datasets. As I worked, I found myself constantly coming back to the same script and tweaking it. Once I had some free time, I decided to sink some effort into creating a polished and streamlined script—this is the result of that.

## [Example Dataset](https://drive.google.com/file/d/1ChUgKi4y8h8JpihcXdb2Up13EjcDFEaM/view?usp=sharing/)

## Goal  
This Python module (`POSYDONHRDiagram.py`) allows the user to generate fully annotated and formatted HR diagrams utilizing both Bokeh and Matplotlib. It serves to allow the user to both quickly generate graphs in order to grasp the used dataset, as well as create fully polished graphs with minimal overhead. Additionally, it allows the user to graph a "reference star" and compare it to the POSYDON dataset. This is useful for comparing simulated datasets to observed systems.

## Features  
The module allows the user to generate Bokeh or POSYDON graphs with nearly identical functions. This allows the user to use the same configuration but entirely switch Python module, greatly increasing efficiency.

## Future plans  
I plan to implement some general functionality for single star systems. If you have any other ideas for functionality, please feel free to suggest them, make a pull request, etc.

## Note  
As this is implemented only with respect to POSYDON, it will not work with other binary datasets unless the column headers match POSYDON formatting.

Additionally, this has NOT been extensively tested, and other datasets may have edge cases which break things. This is still a WIP and written by a second-year undergrad, so expect dumb code mistakes and oversights.

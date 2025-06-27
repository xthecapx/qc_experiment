# Changelog

## [Unreleased] - YYYY-MM-DD

### Major Revisions

This release represents a significant overhaul of the paper, focusing on improving clarity, rigor, and overall structure. The manuscript has been refined to meet a higher academic standard.

### Detailed Changes by File

#### `main.tex`
- **Title**: Updated to be more specific and descriptive. Several alternatives were proposed.
- **Author Names**: Corrected author names to include proper diacritics (Márquez, Garcés).
- **Abstract**: Rewritten for clarity and to provide a more comprehensive summary of the research methodology and findings.
- **File Structure**: Temporarily excluded `results.tex` from the main build.
- **Dependencies**: Added `graphicx` and `subcaption` packages to support figures and subfigures.

#### `introduction.tex`
- **Motivation**: Strengthened the introduction with more detailed background on the current state of quantum computing and benchmarking challenges.
- **Structure**: Reorganized content for a more logical and coherent flow.
- **Scope**: Focused the research questions to narrow the paper's scope for greater impact.
- **Outline**: Provided a clearer, more detailed outline of the paper's structure.

#### `background.tex`
- **Metrics**: Expanded and clarified the sections on pre-runtime and post-runtime metrics, including additional citations.
- **Benchmarking**: Reorganized the discussion of benchmarking variants for better readability.
- **Mathematical Rigor**: Completely reworked the mathematical description of the Teleportation Protocol (TP) with more detailed, step-by-step derivations.

#### `algorithm.tex`
- **Experimental Design**: Added a new "Benchmarking design" section to frame the experimental strategy.
- **Contextualization**: Better integrated the proposed `vTP` (variation of the Teleportation Protocol) within the overall benchmarking framework.
- **Consistency**: Improved consistency by linking mathematical derivations back to equations in the `background.tex` file.

#### `experiment.tex`
- **Language**: Refined the language to be more formal and academic.
- **Methodology**: Justified the use of Clifford gates and provided more quantitative analysis of the results, including R-squared values.
- **Formatting**: Improved the formatting and numbering of equations.

#### `conclusion.tex`
- **Clarity**: Polished the "Conclusion" and "Future work" sections for conciseness and impact.
- **Acknowledgments**: Added a new section acknowledging the use of AI tools in the research and writing process.
- **Cleanup**: Removed commented-out notes and irrelevant text. 
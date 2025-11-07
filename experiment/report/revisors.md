Review #1

Comments:
We remark the following weaknesses:
1. The experimental validation is limited in scope. The authors execute their protocol primarily on IBM quantum services, focusing on a single backend (ibm_sherbrooke).  The claims about general benchmarking applicability across QPUs would be more convincing if results from different vendors or architectures were included. At present, the findings may reflect platform-specific noise models and transpilation strategies rather than universally applicable trends.

The experiment section now includes results from both IBM Sherbrooke and Rigetti Ankaa-3.

2. The statistical treatment of the results is relatively basic. Most conclusions are drawn from visual inspection of plots and linear regression with R² values. While these trends are informative, the lack of confidence intervals, error bars in some experiments, or more advanced statistical analysis (e.g., regression modeling with significance testing) weakens the empirical rigor. This is especially important given the stochastic nature of quantum experiments, where noise and fluctuations could mask or exaggerate observed correlations.

The analysis has been strengthened:

* New plots has been created using shaded regions indicating standard deviation. It also includes a segmented analysis to differentiate the effect of the payload size in the success rate.
* Pearson r, Spearman p, R², slope and p-values have been calculated for the data set and included in all correlation tables to demonstrate statistical significance.

3. Although the literature review is strong and up to date, the comparative discussion of the proposed method versus established benchmarks feels somewhat underdeveloped. The paper highlights limitations of existing methods like randomized benchmarking and quantum volume, but it could more clearly articulate the trade-offs of its own approach. For instance, vTP introduces additional circuit depth and entangling operations, which might make it less practical for large-scale circuits compared to lighter-weight metrics. A frank assessment of these limitations would improve balance and credibility.

The "Final considerations" subsection in the experiment section provides a description of the vTP's limitations. It explicitly states that the protocol itself requires three CNOT gates and a swap operation, which "significantly affect the circuit success rate" and that "vTP stacks up moderate error in the NISQ era, which explains why useful data are obtained only from payloads of less than three qubits". This directly addresses the trade-off of introducing additional circuit complexity.

4.  The mathematical derivations are thorough but lengthy, and they may overwhelm some readers. Moving some of the detailed algebra to an appendix would improve readability. Similarly, the experimental section is sometimes repetitive.

* The detailed mathematical derivation of the standard Teleportation Protocol has been moved to an appendix.

5. The paper hints at important future applications, such as AI-driven prediction of circuit reliability, but these are only mentioned briefly. While it is understandable that this lies beyond the current scope, the paper could better define what is realistically achievable now versus what is speculative.

* The "Future work" section in the conclusions clearly distinguishes between near-term and long-term goals. It outlines a practical next step (development of a comprehensive toolkit) and a long-term vision (develop artificial intelligence models). It also explains how the methodology is "future-proof" for distributed systems, bridging the gap between current and speculative applications.

Review #2
Comments:
Major points:

1. The variation of teleportation protocol (vTP) is introduced, but the novelty over standard TP and prior benchmarking methods is not clearly emphasized. Authors must explicitly highlight how vTP provides new benchmarking capabilities, especially compared to existing randomized benchmarking (RB), cycle benchmarking, or Clifford-based benchmarks.

* The Algorithm Section now explicitly contrasts vTP with the standard TP: vTP notably differs from standard TP in the last phase: it employs CNOT and CZ gates instead of measurement operations and does not require a classical communication channel for state reconstruction. The introduction also contrasts the approach with existing benchmarks like QV and RB, stating they are not suitable for circuit-specific predictive insights.

2. The experiments are performed only on IBM Sherbrooke hardware. Results may not generalize across different architectures.

* The experiment section now includes results from both IBM Sherbrooke and Rigetti Ankaa-3.

3. Stage 1 compares noiseless Aer simulation with mathematical predictions, but later experiments highlight circuit depth compression by IBM transpiler. This discrepancy reduces the validity of correlations between pre-runtime metrics and post-runtime outcomes.

* The abstract now clarifies this: "our analysis correlates the pre-runtime metrics of the final, minimally-transpiled circuit (such as its actual depth and gate count) with post-runtime results..."
* The experiment section explicitly acknowledges this discrepancy: "it is important to note that the final circuit depth on the quantum hardware varied from our target because of optimizations during the transpilation phase". We clarified that the analysis is based on the final transpiled circuit metrics, not the initial target metrics.

4. Success rate and error rate are the only post-runtime metrics. Other important indicators like fidelity, state purity, entanglement entropy, gate error probability from calibration data are omitted.

* Our study is now limited to success rate. Other post-runtime metrics will be considered in future research.

5. While Clifford gates are efficiently simulable, restricting experiments exclusively to the Clifford group limits generalizability. Non-Clifford gates are essential for universal quantum computing.

*  The experiment section includes a detailed justification for this choice. It explains that an initial attempt to use non-Clifford gates presented significant practical challenges on real hardware (e.g., conjugate operations not natively supported). In the same section we that frames the restriction not as an oversight, but as a reasoned methodological decision made in response to hardware limitations.

6. Reported correlations (R² = 0.22, 0.648, 0.840, etc.) show only moderate to strong linear trends. However, the analysis lacks statistical significance testing. Without rigorous statistical treatment, claims about “robust correlation” may be considered overstated.

* New plots has been created using shaded regions indicating standard deviation. It also includes a segmented analysis to differentiate the effect of the payload size in the success rate.
* Pearson r, Spearman p, R², slope and p-values have been calculated for the data set and included in all correlation tables to demonstrate statistical significance.

7. Payload size experiments stop at 5 qubits, with insufficient data for higher payloads. This scale is too small to claim reliability benchmarking for “entire circuits.”

* The language has been moderated. The conclusions now presents a more nuanced claim, stating that "quantum algorithm design should prioritize minimizing entangled qubit operations and circuit depth", which is a direct, supportable conclusion from the 1-5 qubit experiments, rather than a broad claim about all circuits. The "Final considerations" in the experiment section also explains why the useful data is limited to small payloads "...explains why useful data are obtained only from payloads of less than three qubits.".

8. It is acknowledged that IBM transpiler optimizations reduce depth, but authors do not clarify how results are affected by transpilation level. A fair benchmarking study must include side-by-side results at different transpiler levels (0, 1, 2, 3).

The intention of the experiment is to get data with less optimization as possible. However due to different hardware limitations, it was required to do a transpilations before executing the algorithm in the quantum service. Unfortunatelly we do not have direct access to the processors optimization before the pulse generations, so we opted for black box that process and use the available configuration parameters.

9. Current discussion is result-focused but does not sufficiently cover: Effect of qubit connectivity constraints on benchmarking results, Impact of classical communication removal in vTP on benchmarking realism, how error correlations differ between gate errors vs. decoherence vs. crosstalk.

* "Impact of classical communication removal" This is the core novelty of vTP and is discussed in the algorithm section and introduction section.
* The "Final considerations" subsection in the experiment section provides a description of the vTP's limitations. It explicitly states that the protocol itself requires three CNOT gates and a swap operation, which "significantly affect the circuit success rate" and that "vTP stacks up moderate error in the NISQ era, which explains why useful data are obtained only from payloads of less than three qubits". This directly addresses the trade-off of introducing additional circuit complexity.
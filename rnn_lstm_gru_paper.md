

# Page 1


Comparative Analysis of Time Complexity,
Memory Requirements, and Performance in RNN,
LSTM, and GRU Architectures
Raman Shrestha
Department of Electronics and Computer Engineering
Thapathali Engineering Campus
Kathmandu, Nepal
stharamanlm10@gmail.com
Abstract—Recurrent Neural Networks (RNNs) and their gated
variants, Long Short-Term Memory (LSTM) and Gated Re-
current Unit (GRU) networks, remain foundational architec-
tures for sequence modeling tasks. While a substantial body
of literature has empirically compared these architectures on
downstream task accuracy, the analytical treatment of their
computational complexity and memory requirements is often
presented separately from their performance characteristics, or
omitted altogether. This paper presents a comparative analysis
of RNN, LSTM, and GRU architectures with explicit empha-
sis on parameter complexity, training and inference memory
footprint, and empirical performance across sequence modeling
benchmarks. We formalize the asymptotic parameter and com-
putational complexity of each architecture, discuss the memory
implications of Backpropagation Through Time (BPTT) for each
gating structure, and synthesize empirical findings from the
literature to characterize the trade-offs between architectural
complexity and predictive performance. This analysis establishes
the theoretical and empirical foundation for the subsequent
benchmarking component of this study.
Index Terms—Recurrent Neural Networks, Long Short-Term
Memory, Gated Recurrent Unit, time complexity, space complex-
ity, sequence modeling, vanishing gradient.
I. INTRODUCTION
Sequential data, including natural language, speech, finan-
cial series, and climate observations, is characterized by tem-
poral dependencies that violate the independence assumptions
of feedforward neural networks. Recurrent Neural Networks
(RNNs) address this by maintaining a hidden state that is re-
cursively updated at every timestep, allowing information from
prior inputs to influence the processing of subsequent ones [1].
However, vanilla RNNs trained via Backpropagation Through
Time (BPTT) suffer from the vanishing and exploding gradient
problem: because the same weight matrix is repeatedly applied
across timesteps, gradients shrink or grow exponentially with
sequence length, severely limiting the range of dependencies
that can be learned in practice [2].
To mitigate this limitation, gated architectures were intro-
duced. The Long Short-Term Memory (LSTM) network [3]
introduces a dedicated cell state regulated by forget, input, and
output gates, allowing gradients to propagate through largely
linear operations. The forget gate, added in later refinements
[4], allows the network to reset its internal state selectively.
The Gated Recurrent Unit (GRU) [5] was subsequently pro-
posed as a structurally simpler alternative, merging the cell
and hidden states and reducing the gating mechanism to two
gates rather than three.
Although LSTM and GRU were both designed to solve the
same underlying problem, they differ meaningfully in parame-
ter count, memory requirements during training and inference,
and computational cost per timestep. Existing comparative
literature has predominantly emphasized task performance
metrics such as accuracy, perplexity, and RMSE [6], [7],
while treating architectural complexity and memory cost as
secondary considerations. Given the increasing deployment
of recurrent architectures in resource-constrained and latency-
sensitive settings, a rigorous joint treatment of complexity,
memory, and performance is necessary to guide architecture
selection.
This paper addresses this gap by providing a structured
comparative analysis of RNN, LSTM, and GRU architectures
along three axes. Our primary contributions are:
• A formal derivation of the parameter and computational
complexity of RNN, LSTM, and GRU cells as a function
of input and hidden dimensionality.
• An analysis of the memory implications of each architec-
ture under Backpropagation Through Time, including the
additional cell-state storage required by LSTM relative to
GRU and vanilla RNN.
• A synthesis of empirical performance comparisons re-
ported in the literature, characterizing the conditions un-
der which each architecture offers a favorable complexity-
to-performance trade-off.
II. LITERATURE REVIEW
The literature relevant to this study is organized around four
interconnected themes rather than by individual publication:
the theoretical basis of the vanishing gradient problem that
motivates recurrence-based gating, the architectural design
choices underlying LSTM and GRU gating mechanisms, the
complexity and memory implications of these design choices,


# Page 2


and the empirical evidence characterizing the resulting perfor-
mance trade-offs.
A. Vanishing Gradients and the Motivation for Gating
Werbos [1] formalized Backpropagation Through Time,
establishing the standard training procedure for recurrent ar-
chitectures by unrolling the network across timesteps and
applying the chain rule through the resulting computational
graph. Pascanu et al. [2] provided a rigorous analysis of why
this unrolled gradient computation fails for long sequences,
deriving bounds on the norm of the Jacobian of the recurrent
transition function and showing that gradients vanish or ex-
plode exponentially with the number of timesteps whenever
this norm falls below or exceeds unity, respectively. Ghojogh
and Ghodsi [8] present a complementary formal treatment,
deriving RNN as a parameter-shared dynamical system and
expressing the BPTT loss as a summation over timestep-
wise losses, from which the same gradient pathology can be
derived directly through the chain rule across the unrolled
state sequence. Together, these treatments establish that any
architecture intending to learn long-range dependencies must
provide a path for gradients that does not depend solely on
repeated multiplication by the recurrent weight matrix, directly
motivating the gated designs discussed next.
B. Gating Design: LSTM and GRU
Hochreiter and Schmidhuber [3] introduced the LSTM
architecture, which addresses the vanishing gradient problem
by maintaining a separate cell state updated through largely
additive, gated interactions rather than repeated matrix multi-
plication. The original architecture was later refined by Gers
et al. [4], who introduced the forget gate to allow the network
to reset the cell state, addressing a failure mode in which un-
bounded cell state growth degraded performance on continual
prediction tasks. Cho et al. [5] proposed the GRU as part of an
encoder-decoder framework for statistical machine translation,
combining the cell and hidden states into a single vector and
reducing the gating mechanism to an update gate and a reset
gate. This reduction yields a structurally simpler cell with
fewer weight matrices than LSTM, motivating the complexity
comparison central to this study. Mienye et al. [9] situate
LSTM and GRU within a broader landscape of gating variants,
including peephole LSTM, stacked and bidirectional LSTM,
and echo state networks, and note that GRU’s reduction from
LSTM’s three sigmoid gates to two, alongside its merged
cell and hidden state, is the principal structural source of its
parameter and computational savings relative to LSTM.
C. Complexity and Memory Considerations
Relative to the extensive treatment of gating design, for-
mal analysis of the resulting parameter, computational, and
memory cost of these architectures is comparatively sparse in
the literature. Beyond gated recurrent architectures, Bai et al.
[10] benchmarked Temporal Convolutional Networks (TCNs)
against LSTM and GRU across a range of sequence modeling
tasks, reporting that convolutional architectures with sufficient
receptive field frequently match or exceed recurrent architec-
tures while offering superior parallelization during training due
to the absence of sequential recurrence. This underscores that
the parameter and memory efficiency of a recurrent cell must
be evaluated relative to its computational parallelizability, not
parameter count alone. Greff et al. [11], in the course of a
large-scale ablation study discussed further below, also report
training time under a fixed computational budget as a practical
constraint on their random architecture search, offering one of
the few instances in the literature where computational cost
is tracked alongside performance rather than treated as an
afterthought. Yunita et al. [12] extend this further by reporting
empirically measured training time as a headline result across
nine recurrent and hybrid architectures, finding that vanilla
RNN is consistently the fastest architecture and that LSTM-
RNN hybrids are generally faster than LSTM-GRU hybrids.
Notably, none of these studies report empirically measured
memory footprint, nor do they derive parameter count or
floating-point operation formulas explicitly as a function of
hidden and input dimensionality; this remains a gap that the
present study’s theoretical framework directly addresses.
D. Empirical Performance Trade-offs
Chung et al. [6] conducted the first systematic empirical
comparison of tanh-RNN, LSTM, and GRU units on poly-
phonic music modeling and speech signal modeling tasks,
matching architectures by approximate parameter count rather
than hidden size to isolate the effect of gating from raw model
capacity. Their results indicated that gated units consistently
outperform traditional tanh units, and that GRU performance
is comparable to LSTM despite its simpler structure, though
no architecture was found to be uniformly superior across all
tasks. This paper serves as the primary empirical anchor for
the present study, and its parameter-matching convention is
adopted in the research design below. Jozefowicz et al. [7]
extended this line of inquiry through a large-scale architecture
search over thousands of RNN variants, finding that perfor-
mance differences between LSTM and GRU are frequently
small relative to the effect of hyperparameter choices such
as forget-gate bias initialization, suggesting that architectural
complexity alone is an incomplete predictor of task perfor-
mance. Greff et al. [11] performed a large-scale ablation
study of eight LSTM variants across speech recognition, hand-
writing recognition, and polyphonic music modeling, using
the fANOVA framework to assess the relative importance of
individual gates and components, finding that the forget gate
and output activation function are the most critical components
while other components contribute marginally – providing
empirical justification for the structural simplifications made
in the GRU design. Most recently, Yunita et al. [12] con-
ducted a Monte Carlo comparison of nine two-hidden-layer
architectures (RNN, LSTM, GRU, and six hybrid stackings)
across multiple time series datasets over a hundred repeated
iterations with statistical testing, finding that LSTM performed
best on one dataset and LSTM-RNN hybrids on the remaining
two, with no architecture found to be universally optimal


# Page 3


and no statistically significant overall difference among the
nine architectures despite consistent patterns favoring LSTM-
based hybrids. Mienye et al. [9] synthesize this body of
evidence at a review level, concluding that GRU’s reduced
parameter count, arising from the absence of a separate cell
state and its more compact gating, often yields faster training
with performance comparable to LSTM, though the preferable
architecture remains task- and dataset-dependent.
Collectively, this literature establishes that LSTM introduces
additional parameter and memory overhead relative to GRU
through its dedicated cell state and additional gate, that this
overhead does not translate into a consistent, task-independent
performance advantage, and that architecture selection should
be informed by a joint consideration of computational budget,
memory constraints, and the specific dependency structure of
the target task. It further reveals a persistent gap: existing
studies are either purely theoretical, with no empirical vali-
dation [8]; empirical without explicit parameter-count control
or memory profiling [12]; or synthesize prior findings without
conducting new experiments [9]. This synthesis forms the
basis for the complexity and memory analysis conducted in
the remainder of this study, which combines formal deriva-
tion with a parameter-matched empirical comparison of time,
memory, and performance.
III. METHODOLOGY
A. Theoretical Framework and Justification
This study adopts a dynamical-systems formulation of recur-
rence as its theoretical starting point, from which the parame-
ter, computational, and memory complexity of RNN, LSTM,
and GRU cells are derived analytically prior to any empirical
measurement. This section formalizes that framework and
justifies the design choices carried forward into the research
design.
1) Recurrence as a Parameter-Shared Dynamical System:
Following the formulation adopted by Ghojogh and Ghodsi
[8], a recurrent cell is treated as a discrete-time dynamical
system in which a shared parameter set θ is applied recursively
across timesteps:
ht = fθ(ht−1, xt),
t = 1, . . . , T
(1)
where xt ∈Rx is the input at timestep t, ht ∈Rh is the hidden
state, and fθ is instantiated differently for RNN, LSTM, and
GRU cells, as detailed in Section III-A3. Training proceeds
by Backpropagation Through Time (BPTT), in which a total
loss is expressed as a sum of per-timestep losses,
L =
T
X
t=1
Lt(ht, yt),
(2)
and gradients with respect to θ are obtained by applying the
chain rule across the unrolled computational graph [1].
2) Vanishing Gradients as the Motivating Constraint:
Differentiating Equation 2 with respect to an early hidden state
hk (k < T) requires the repeated application of the Jacobian
of the recurrent transition function:
∂hT
∂hk
=
T
Y
t=k+1
∂ht
∂ht−1
.
(3)
Pascanu et al. [2] show that the norm of this product shrinks
or grows exponentially in T −k whenever the spectral norm of
∂ht/∂ht−1 is respectively below or above unity, yielding the
vanishing and exploding gradient problem. Equation 3 is the
formal justification for this study’s premise: any architectural
modification intended to preserve long-range gradient flow
must alter the structure of ∂ht/∂ht−1 itself, rather than merely
rescaling Wh. This is precisely what the gating mechanisms
in LSTM [3], [4] and GRU [5] achieve, and it is this structural
alteration – the introduction of gates – that is the direct
source of the additional parameters, computation, and memory
analyzed below. The complexity–performance trade-off central
to this study therefore does not arise arbitrarily; it is the
necessary cost of the mechanism that solves Equation 3’s
pathology.
3) Parameter Complexity Derivation: Let x denote the
input dimensionality and h the hidden state dimensionality,
consistent with the state-transition function fθ in Equation 1.
Each gate or candidate computation within a recurrent cell
requires one input-to-hidden weight matrix W ∈Rh×x, one
hidden-to-hidden weight matrix U ∈Rh×h, and one bias
vector b ∈Rh, giving a per-gate parameter count of
pgate = h(h + x) + h.
(4)
The total parameter count of each architecture follows directly
from its number of internal gates:
PRNN = 1 · pgate = h(h + x) + h,
(5)
PGRU = 3 · pgate = 3

h(h + x) + h

,
(6)
PLSTM = 4 · pgate = 4

h(h + x) + h

.
(7)
Equations 5–7 formalize the qualitative claim, made across
the reviewed literature [5], [9], that GRU’s two gates plus
candidate computation yield a 3× parameter cost relative to
a vanilla RNN cell, while LSTM’s three gates plus candidate
computation yield a 4× cost, at identical hidden size h.
4) Computational Complexity Derivation: Each per-gate
computation in Equation 4 is dominated by two matrix–
vector multiplications, Wxt and Uht−1, together requiring
O(h(h + x)) floating-point operations. The per-timestep com-
putational cost of each architecture therefore scales with its
gate count in direct correspondence with Equations 5–7:
CRNN = O
 h(h + x)

,
CGRU = O
 3h(h + x)

,
CLSTM = O
 4h(h + x)

.
(8)
For a full sequence of length T, the forward-pass cost is T ·
Carch, with Carch given per architecture by Equation 8. Since


# Page 4


one training step requires both a forward pass and a backward
pass through the unrolled graph implied by Equation 3, and
the backward pass repeats computation of comparable order to
the forward pass, the per-epoch training cost is approximately
Cepoch ≈3 T · Carch,
(9)
the factor of three following the standard accounting of one
forward and two backward-equivalent passes in BPTT. Equa-
tion 9 is the theoretical quantity that the empirical wall-clock
timing in the research design is intended to validate.
5) Memory Complexity Derivation Under BPTT: Beyond
static parameter storage (Parch × 4 bytes under 32-bit floating
point), BPTT requires retaining every intermediate hidden
state produced during the forward pass, since each is needed
when the chain rule in Equation 3 is evaluated during the
backward pass. For RNN and GRU, only the hidden state
ht ∈Rh must be retained at each of the T timesteps, giving
activation memory
MRNN = MGRU = O(Th).
(10)
LSTM additionally maintains a cell state ct ∈Rh, updated
additively alongside ht, which must also be retained at every
timestep for the backward pass:
MLSTM = O(2Th).
(11)
Equation 11 is the formal, quantitative justification – absent
from the empirical studies reviewed in Section II-C, none of
which report memory profiling – for the claim that LSTM
incurs approximately double the activation memory of RNN or
GRU under BPTT, independent of any difference in parameter
count.
6) Justification of the Parameter-Matching Design: Equa-
tions 5–7 establish that, at a common hidden size h, LSTM
and GRU possess substantially more representational capacity
than a vanilla RNN. Consequently, comparing architectures at
fixed h confounds the effect of gating structure with the effect
of raw parameter count: a performance advantage observed
for LSTM under such a design cannot be attributed to its
gating mechanism as opposed to its larger capacity. Following
the convention established by Chung et al. [6], this study
instead solves for the hidden size harch of each architecture
such that PRNN(hRNN) ≈PGRU(hGRU) ≈PLSTM(hLSTM) for a
fixed target parameter budget, using Equations 5–7 directly to
solve for each harch. This design choice, absent from the more
recent empirical comparison of Yunita et al. [12], isolates the
architectural effect of gating from the confounding effect of
model capacity, and is adopted as the governing constraint for
the research design presented in the following subsection.
B. Research Design
This study employs a controlled comparative experimental
design in which RNN, LSTM, and GRU are evaluated as
sequence encoders on a single, shared forecasting task, with
all factors other than recurrent cell type held constant across
the three conditions. This design is intended to directly test,
empirically, the theoretical predictions of Equations 5–11:
namely, that LSTM and GRU incur 4× and 3× the parameter
and computational cost of a vanilla RNN respectively, that
LSTM incurs approximately double the activation memory of
RNN or GRU under BPTT, and that this additional cost does
not necessarily yield a proportional performance advantage.
1) Task: The shared task is deterministic forecasting of the
Ni˜no 3.4 index at a six-month lead time, using a fixed-length
window of MIFS-selected predictor features as sequential
input. This task is selected because it is a genuine sequence-to-
one regression problem with real-world dependency structure
spanning multiple timescales, rather than a synthetic bench-
mark constructed solely for this comparison.
2) Parameter-Matched Architecture Configuration: Rather
than fixing a common hidden size h across architectures –
which, per Section III-A, would confound gating structure with
model capacity – a target parameter budget P ∗is fixed, and
the hidden size of each architecture is solved independently
from Equations 5–7:
harch = arg min
h∈Z+ |Parch(h) −P ∗| ,
arch ∈{RNN, GRU, LSTM}.
(12)
Because LSTM requires four gate computations against RNN’s
one, Equation 12 yields a substantially smaller hLSTM than
hRNN for the same P ∗; the resulting hidden sizes and real-
ized parameter counts for all three architectures are reported
explicitly alongside the results, following the convention of
Chung et al. [6].
3) Controlled Variables: The following factors are held
identical across all three architectures, so that any observed
difference in time, memory, or performance is attributable
to the recurrent cell type alone rather than to confounding
hyperparameter variation: input feature set (the MIFS-selected
subset of size K = 30), number of recurrent layers, output
layer architecture, learning rate and optimizer, batch size, num-
ber of training epochs, weight initialization scheme, and the
train/validation/test partition of the data described in Section
III-C.
4) Measurement
Axes:
Three
measurement
axes
are
recorded for each architecture, operationalizing the theoretical
quantities derived in Section III-A:
• Time.
Wall-clock
training
time
per
epoch
and
per-sample
inference
latency,
measured
using
time.perf_counter()
around
the
training
and
inference
loops.
Because
modern
accelerator
backends (e.g., CUDA or MPS) dispatch operations
asynchronously, the appropriate device-synchronization
call
(e.g.,
torch.cuda.synchronize()
or
torch.mps.synchronize(), selected according to
the available hardware) is invoked immediately before
each timing measurement, since omitting this step would
yield artificially low timings that reflect kernel dispatch
latency rather than actual computation. The first epoch
of each run is excluded from the reported average to
remove one-time initialization overhead. This directly
operationalizes the per-epoch cost derived in Equation 9.


# Page 5


• Memory. Peak memory allocated during a training step,
together with static parameter memory computed as
Parch×4 bytes. This directly operationalizes the activation
memory relationships derived in Equations 10–11.
• Performance. Root Mean Squared Error (RMSE) and
Pearson correlation coefficient (r) between predicted and
observed Ni˜no 3.4 index values on the held-out test
partition, computed identically for all three architectures.
5) Statistical Rigor: Following the precedent of repeated-
trial evaluation established by Yunita et al. [12], each ar-
chitecture is trained across multiple independent runs under
distinct random seeds, rather than a single run per architecture.
Time, memory, and performance metrics are reported as mean
± standard deviation across runs, and pairwise differences
between architectures are assessed for statistical significance,
so that observed differences can be distinguished from run-to-
run training variance.
C. Data Collection Procedure
1) Data Sources and Domain: Predictor variables are de-
rived from ERA5 atmospheric reanalysis and ORAS5 ocean
reanalysis products, subset to the tropical Pacific domain
(30°S–30°N, 120°E–80°W) and coarsened to a 2°×2° spatial
resolution, spanning 1980–2025. The forecast target is the
Ni˜no 3.4 index, computed from area-weighted sea surface
temperature anomalies within the Ni˜no 3.4 region.
2) Feature Selection: Predictor features are restricted to
the subset selected by Mutual Information Feature Selection
(MIFS) at K = 30, as established in a concurrent departmental
project on Ni˜no 3.4 forecasting that developed the underlying
data pipeline. This study uses only the flattened, MIFS-
selected feature vector as sequential input; the full spatial
predictor fields and dual-head spatial output used in that
concurrent project’s CNN-TCN architecture are not used here,
since RNN, LSTM, and GRU cells operate on sequential
feature vectors rather than spatial fields.
3) Preprocessing and Splits: All predictor and target vari-
ables are converted to anomalies relative to a 1980–2010
climatological baseline and z-score normalized using statistics
computed over the training partition only, to avoid information
leakage from validation or test periods. Data are partitioned
chronologically, without shuffling, into training (1980–2018),
validation (2019–2022), and test (2023–2025) sets, consistent
with standard practice for time series forecasting where tem-
poral ordering must be preserved to obtain a valid estimate of
out-of-sample performance.
D. Data Processing and Analysis Tools
1) Modeling Framework: All three architectures are im-
plemented in PyTorch, using its native nn.RNN, nn.LSTM,
and nn.GRU modules with hidden sizes set according to
Equation 12. Training and inference are executed on the
accelerator backend available to the experimental hardware
(e.g., a CUDA-enabled GPU or Apple’s MPS backend), with
PyTorch’s device-agnostic API used to target whichever back-
end is present, and CPU fallback enabled for any operation
not yet natively supported on the selected backend.
2) Timing
Instrumentation:
Wall-clock
training
and
inference
time
are
measured
using
Python’s
time.perf_counter(). Because accelerator backends
typically dispatch operations asynchronously, the backend-
appropriate synchronization call is invoked immediately
before
each
timing
checkpoint
to
force
completion
of all pending device work; without this call, elapsed time
would reflect only kernel dispatch latency rather than actual
computation time. The first training epoch of each run
is excluded from reported averages to remove one-time
data-loading and kernel-compilation overhead.
3) Memory
Instrumentation:
Peak
memory
during
training
is
measured
using
PyTorch’s
backend-appropriate
memory-tracking
utilities
(e.g.,
torch.cuda.max_memory_allocated() on CUDA-
enabled
hardware,
or
the
corresponding
MPS
memory
utilities
on
Apple
Silicon),
subject
to
availability
in
the
installed
PyTorch
version,
which
is
verified
prior
to
experimentation.
Where
accelerator-level
memory
instrumentation
proves
unavailable
or
unreliable,
memory is instead measured at the process level using
psutil.Process().memory_info().rss,
sampled
during training, or via Python’s built-in tracemalloc
module. Static parameter memory is computed directly
as
Parch × 4
bytes,
following
Section
III-A,
using
sum(p.numel() for p in model.parameters())
to obtain Parch.
4) Performance Evaluation: Test-set RMSE and Pearson
correlation coefficient are computed using NumPy and SciPy,
comparing predicted and observed Ni˜no 3.4 index values over
the 2023–2025 test partition. Statistical significance of pair-
wise differences between architectures across repeated runs
(Section III-B) is assessed using scipy.stats, applying
a paired significance test (e.g., a paired t-test or Wilcoxon
signed-rank test, selected according to whether the per-run
metric differences are approximately normally distributed)
rather than a single-run point comparison.
5) Reproducibility: All tools used in this study – PyTorch,
NumPy, SciPy, and psutil – are open-source and freely
available, requiring no institutional license or paid access.
Each architecture is trained under a fixed set of distinct random
seeds, applied identically across RNN, LSTM, and GRU, so
that reported results can be independently reproduced. All
experiments are executed locally, without reliance on external
compute services, and the specific hardware configuration, ac-
celerator backend, PyTorch version, and random seeds used are
reported alongside the results to support exact reproduction.
REFERENCES
[1] P. J. Werbos, “Backpropagation through time: what it does and how to
do it,” Proceedings of the IEEE, vol. 78, no. 10, pp. 1550–1560, 1990.
[2] R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training
recurrent neural networks,” arXiv preprint arXiv:1211.5063, 2013.


# Page 6


[3] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural
Computation, vol. 9, no. 8, pp. 1735–1780, 1997.
[4] F. A. Gers, J. Schmidhuber, and F. Cummins, “Learning to forget:
Continual prediction with LSTM,” in Neural Computation, vol. 12,
no. 10, 2000, pp. 2451–2471.
[5] K. Cho, B. van Merri¨enboer, C. Gulcehre, D. Bahdanau, F. Bougares,
H. Schwenk, and Y. Bengio, “Learning phrase representations using
RNN encoder-decoder for statistical machine translation,” arXiv preprint
arXiv:1406.1078, 2014.
[6] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, “Empirical evaluation of
gated recurrent neural networks on sequence modeling,” arXiv preprint
arXiv:1412.3555, 2014.
[7] R. Jozefowicz, W. Zaremba, and I. Sutskever, “An empirical exploration
of recurrent network architectures,” in Proceedings of the 32nd Interna-
tional Conference on Machine Learning (ICML), 2015, pp. 2342–2350.
[8] B. Ghojogh and A. Ghodsi, “Recurrent neural networks and long
short-term memory networks: Tutorial and survey,” arXiv preprint
arXiv:2304.11461, 2023.
[9] I. D. Mienye, T. G. Swart, and G. Obaido, “Recurrent neural networks:
A comprehensive review of architectures, variants, and applications,”
Information, vol. 15, no. 9, p. 517, 2024.
[10] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic
convolutional and recurrent networks for sequence modeling,” arXiv
preprint arXiv:1803.01271, 2018.
[11] K. Greff, R. K. Srivastava, J. Koutn´ık, B. R. Steunebrink, and J. Schmid-
huber, “LSTM: A search space odyssey,” IEEE Transactions on Neural
Networks and Learning Systems, vol. 28, no. 10, pp. 2222–2232, 2017.
[12] A. Yunita, M. H. D. I. Pratama, M. Z. Almuzakki, H. Ramadhan, E. A. P.
Akhir, A. B. F. Mansur, and A. H. Basori, “Performance analysis of
neural network architectures for time series forecasting: A comparative
study of RNN, LSTM, GRU, and hybrid models,” MethodsX, vol. 15,
p. 103462, 2025.
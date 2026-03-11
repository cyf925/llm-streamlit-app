pragma circom 2.2.3;

include "circomlib/circuits/comparators.circom";

// -----------------------------------------------------------------------------
// truthfinder.circom (Circom 2) - TruthFinder Circuit Spec v1 (project circuit semantics)
//
// NOTE (very important):
// - This circuit DOES NOT aim to bit-match floating TruthFinder.py.
// - This circuit IS the project-formal "circuit semantics" TruthFinder version:
//   fixed Q16 arithmetic + fixed approximation rules + fixed 25 rounds.
// - A future TruthFinder_circuit_ref.py must use the SAME piecewise boundaries and
//   coefficients defined below, so Python reference and circuit stay aligned.
//
// Project-fixed contract:
//   M=4, K_MAX=15, N_MAX=12, ITER_N=25, Q16=65536
//   ITER_N is fixed by project policy (not runtime-variable in this circuit).
//
// Input format consumes truthfinder_circom_input.json produced by
// prepare_circom_input.py (already expanded/flattened/validated by Python pipeline).
//
// Flatten index restore (must stay consistent with prepare_circom_input.py):
//   support_flat[o,f,w] => ((o*N_MAX)+f)*M + w      (order: o -> f -> w)
//   imp_flat[o,g,f]     => ((o*N_MAX)+g)*N_MAX + f  (order: o -> g -> f)
//   conf_flat[o,g,f]    => ((o*N_MAX)+g)*N_MAX + f  (order: o -> g -> f)
//   top1_choice_flat    => o*M + w                  (order: o -> w)
//
// top1_choice_flat role:
//   top1_choice is only used off-circuit to derive dep_avg (Python side).
//   This circuit's core computation consumes dep_avg directly.
//   top1_choice_flat is kept only for input-contract compatibility, not used in constraints.
//
// Tie-break policy (stable and explicit):
//   - model argmax: equal score => smaller model index wins.
//   - fact argmax:  equal score => smaller fact index wins.
// -----------------------------------------------------------------------------

var Q16 = 65536;
var M = 4;
var K_MAX = 15;
var N_MAX = 12;
var ITER_N = 25;
var OF = K_MAX * N_MAX; // 180


template SumN(N) {
    signal input in[N];
    signal output out;
    signal acc[N + 1];

    acc[0] <== 0;
    for (var i = 0; i < N; i++) {
        acc[i + 1] <== acc[i] + in[i];
    }
    out <== acc[N];
}


template Q16Mul() {
    // out = floor((a*b)/Q16), with 0 <= remainder < Q16
    signal input a;
    signal input b;
    signal output out;

    signal r;
    a * b === out * Q16 + r;

    component rlt = LessThan(17);
    rlt.in[0] <== r;
    rlt.in[1] <== Q16;
    rlt.out === 1;
}


template Q16Clamp01() {
    // Clamp to [0, Q16], assuming non-negative input under this circuit flow.
    signal input in;
    signal output out;

    component le = LessThan(32);
    le.in[0] <== in;
    le.in[1] <== Q16 + 1; // in <= Q16

    out <== Q16 + le.out * (in - Q16);
}


template SafeDivNonNeg() {
    // If den > 0: out = floor(num / den)
    // If den = 0: out = fallback
    signal input num;
    signal input den;
    signal input fallback;
    signal output out;

    component iz = IsZero();
    iz.in <== den;

    signal denEff;
    denEff <== den + iz.out; // if den=0 => 1

    signal q;
    signal r;
    num === q * denEff + r;

    component rlt = LessThan(32);
    rlt.in[0] <== r;
    rlt.in[1] <== denEff;
    rlt.out === 1;

    out <== q * (1 - iz.out) + fallback * iz.out;
}


template MaxWithTieBreak() {
    signal input curVal;
    signal input curIdx;
    signal input nxtVal;
    signal input nxtIdx;

    signal output outVal;
    signal output outIdx;

    component lt = LessThan(32);
    lt.in[0] <== curVal;
    lt.in[1] <== nxtVal;

    // if cur < nxt => take nxt; else keep cur.
    // Equal scores keep previous candidate => smaller index survives in left-to-right scan.
    outVal <== curVal + lt.out * (nxtVal - curVal);
    outIdx <== curIdx + lt.out * (nxtIdx - curIdx);
}


template ArgMaxModelWithTieBreak() {
    signal input score[M];
    signal output bestIdx;
    signal output bestScore;

    signal bestVal[M];
    signal bestId[M];

    bestVal[0] <== score[0];
    bestId[0] <== 0;

    for (var i = 1; i < M; i++) {
        component mx = MaxWithTieBreak();
        mx.curVal <== bestVal[i - 1];
        mx.curIdx <== bestId[i - 1];
        mx.nxtVal <== score[i];
        mx.nxtIdx <== i;

        bestVal[i] <== mx.outVal;
        bestId[i] <== mx.outIdx;
    }

    bestIdx <== bestId[M - 1];
    bestScore <== bestVal[M - 1];
}


template ArgMaxFactWithTieBreak() {
    signal input score[N_MAX];
    signal output bestIdx;
    signal output bestScore;

    signal bestVal[N_MAX];
    signal bestId[N_MAX];

    bestVal[0] <== score[0];
    bestId[0] <== 0;

    for (var i = 1; i < N_MAX; i++) {
        component mx = MaxWithTieBreak();
        mx.curVal <== bestVal[i - 1];
        mx.curIdx <== bestId[i - 1];
        mx.nxtVal <== score[i];
        mx.nxtIdx <== i;

        bestVal[i] <== mx.outVal;
        bestId[i] <== mx.outIdx;
    }

    bestIdx <== bestId[N_MAX - 1];
    bestScore <== bestVal[N_MAX - 1];
}


template ApproxTauQ16() {
    // Formal _tau_circuit (Approximation Spec Freeze v1).
    // Input:  t_q16 in [0, 65536]
    // Output: tau_q16 as evidence-strength (NOT probability), may be > 65536.
    //
    // Frozen piecewise-linear approximation for tau(t) = -log(1-t):
    //   S0: [0,16384)      y=floor(75414*t/65536)+0
    //   S1: [16384,32768)  y=floor(106290*t/65536)-7719
    //   S2: [32768,49152)  y=floor(181704*t/65536)-45426
    //   S3: [49152,57344)  y=floor(363409*t/65536)-181704
    //   S4: [57344,61440)  y=floor(726817*t/65536)-499687
    //   S5: [61440,65536]  y=floor(1287034*t/65536)-1024890
    //
    // This frozen boundary/coeff set is part of project circuit semantics and MUST
    // be mirrored exactly in TruthFinder_circuit_ref.py.
    // The last segment is an engineering saturation segment with cap 262144 (=4.0 Q16).
    signal input t;
    signal output tau;

    component lt1 = LessThan(32); // t < 16384
    lt1.in[0] <== t;
    lt1.in[1] <== 16384;

    component lt2 = LessThan(32); // t < 32768
    lt2.in[0] <== t;
    lt2.in[1] <== 32768;

    component lt3 = LessThan(32); // t < 49152
    lt3.in[0] <== t;
    lt3.in[1] <== 49152;

    component lt4 = LessThan(32); // t < 57344
    lt4.in[0] <== t;
    lt4.in[1] <== 57344;

    component lt5 = LessThan(32); // t < 61440
    lt5.in[0] <== t;
    lt5.in[1] <== 61440;

    signal g0;
    signal g1;
    signal g2;
    signal g3;
    signal g4;
    signal g5;

    g0 <== lt1.out;
    g1 <== lt2.out - lt1.out;
    g2 <== lt3.out - lt2.out;
    g3 <== lt4.out - lt3.out;
    g4 <== lt5.out - lt4.out;
    g5 <== 1 - lt5.out;

    component m0 = Q16Mul();
    component m1 = Q16Mul();
    component m2 = Q16Mul();
    component m3 = Q16Mul();
    component m4 = Q16Mul();
    component m5 = Q16Mul();

    m0.a <== t; m0.b <== 75414;
    m1.a <== t; m1.b <== 106290;
    m2.a <== t; m2.b <== 181704;
    m3.a <== t; m3.b <== 363409;
    m4.a <== t; m4.b <== 726817;
    m5.a <== t; m5.b <== 1287034;

    signal y0;
    signal y1;
    signal y2;
    signal y3;
    signal y4;
    signal y5;

    y0 <== m0.out + 0;
    y1 <== m1.out - 7719;
    y2 <== m2.out - 45426;
    y3 <== m3.out - 181704;
    y4 <== m4.out - 499687;
    y5 <== m5.out - 1024890;

    signal raw;
    raw <== g0 * y0 + g1 * y1 + g2 * y2 + g3 * y3 + g4 * y4 + g5 * y5;

    // Capped to 262144 (4.0 in Q16), per freeze spec final engineering bound.
    component ltCap = LessThan(32);
    ltCap.in[0] <== raw;
    ltCap.in[1] <== 262145;

    tau <== 262144 + ltCap.out * (raw - 262144);
}


template ApproxSigmoidQ16Signed() {
    // Formal _sigmoid_circuit (Approximation Spec Freeze v1), signed input.
    //
    // Because field arithmetic is unsigned, signed x is represented as:
    //   x = sign ? (-x_abs) : (+x_abs), where x_abs >= 0 and sign in {0,1}.
    //   x_is_zero marks x==0 to select exact midpoint output 32768.
    //
    // Frozen boundaries/coefficients MUST be mirrored in TruthFinder_circuit_ref.py.
    signal input x_abs;
    signal input x_is_neg;
    signal input x_is_zero;
    signal output y;

    x_is_neg * (x_is_neg - 1) === 0;
    x_is_zero * (x_is_zero - 1) === 0;
    x_is_neg * x_is_zero === 0;

    // |x| boundaries: 1,2,4,6 in Q16 units
    component lt1 = LessThan(32); // < 65536
    lt1.in[0] <== x_abs;
    lt1.in[1] <== 65536;

    component lt2 = LessThan(32); // < 131072
    lt2.in[0] <== x_abs;
    lt2.in[1] <== 131072;

    component lt4 = LessThan(32); // < 262144
    lt4.in[0] <== x_abs;
    lt4.in[1] <== 262144;

    component lt6 = LessThan(32); // < 393216
    lt6.in[0] <== x_abs;
    lt6.in[1] <== 393216;

    // interval gates over |x|
    signal gA; // [0,1)
    signal gB; // [1,2)
    signal gC; // [2,4)
    signal gD; // [4,6)
    signal gE; // [6,+inf)

    gA <== lt1.out;
    gB <== lt2.out - lt1.out;
    gC <== lt4.out - lt2.out;
    gD <== lt6.out - lt4.out;
    gE <== 1 - lt6.out;

    // precompute floor(c*|x|/65536)
    component m508 = Q16Mul();
    component m3317 = Q16Mul();
    component m9813 = Q16Mul();
    component m15143 = Q16Mul();

    m508.a <== x_abs;   m508.b <== 508;
    m3317.a <== x_abs;  m3317.b <== 3317;
    m9813.a <== x_abs;  m9813.b <== 9813;
    m15143.a <== x_abs; m15143.b <== 15143;

    // negative branch:
    // S0: x<=-6         -> 162
    // S1: (-6,-4]       ->  floor( 508*x/65536)+3212 = 3212 - floor(508*|x|/65536)
    // S2: (-4,-2]       -> 14445 - floor(3317*|x|/65536)
    // S3: (-2,-1]       -> 27439 - floor(9813*|x|/65536)
    // S4: (-1,0]        -> 32768 - floor(15143*|x|/65536)
    signal yNeg;
    yNeg <== gE * 162
          + gD * (3212 - m508.out)
          + gC * (14445 - m3317.out)
          + gB * (27439 - m9813.out)
          + gA * (32768 - m15143.out);

    // positive branch:
    // S9: x>=6          -> 65374
    // S8: (4,6]         -> floor( 508*x/65536)+62324
    // S7: (2,4]         -> floor(3317*x/65536)+51091
    // S6: (1,2]         -> floor(9813*x/65536)+38097
    // S5: (0,1]         -> floor(15143*x/65536)+32768
    signal yPos;
    yPos <== gE * 65374
          + gD * (62324 + m508.out)
          + gC * (51091 + m3317.out)
          + gB * (38097 + m9813.out)
          + gA * (32768 + m15143.out);

    // x==0 exact midpoint 32768; otherwise choose sign branch.
    signal yNoZero;
    yNoZero <== x_is_neg * yNeg + (1 - x_is_neg) * yPos;

    signal yRaw;
    yRaw <== x_is_zero * 32768 + (1 - x_is_zero) * yNoZero;

    // Final clamp to [0,65536] upper bound (piecewise formula is already non-negative).
    component le1 = LessThan(32);
    le1.in[0] <== yRaw;
    le1.in[1] <== 65537;

    y <== 65536 + le1.out * (yRaw - 65536);
}

template TruthFinderRound() {
    // One round update: (t_in, s_prev) -> (t_out, s_out)
    // Inputs use fixed flattened arrays and Q16 semantics.
    signal input t_in[M];
    signal input s_prev[K_MAX][N_MAX];

    signal input beta;
    signal input gamma;
    signal input alpha_imp;
    signal input alpha_conflict;
    signal input min_tau_scale;

    signal input fact_count_by_object[K_MAX];
    signal input is_effective_by_object[K_MAX];

    signal input dep_avg[M];
    signal input support_flat[K_MAX * N_MAX * M];
    signal input imp_flat[K_MAX * N_MAX * N_MAX];
    signal input conf_flat[K_MAX * N_MAX * N_MAX];

    signal output t_out[M];
    signal output s_out[K_MAX][N_MAX];

    signal tauW[M];

    // tau and dependency damping
    for (var w = 0; w < M; w++) {
        component tauApprox = ApproxTauQ16();
        tauApprox.t <== t_in[w];

        component gmul = Q16Mul();
        gmul.a <== gamma;
        gmul.b <== dep_avg[w];

        signal oneMinus;
        oneMinus <== Q16 - gmul.out;

        component ltScale = LessThan(32);
        ltScale.in[0] <== oneMinus;
        ltScale.in[1] <== min_tau_scale;

        signal scaleW;
        scaleW <== oneMinus + ltScale.out * (min_tau_scale - oneMinus);

        component damp = Q16Mul();
        damp.a <== tauApprox.tau;
        damp.b <== scaleW;
        tauW[w] <== damp.out;
    }

    // update s
    for (var o = 0; o < K_MAX; o++) {
        for (var f = 0; f < N_MAX; f++) {
            signal baseTerms[M];
            for (var w2 = 0; w2 < M; w2++) {
                var sIdx = ((o * N_MAX) + f) * M + w2;
                component bm = Q16Mul();
                bm.a <== tauW[w2];
                bm.b <== support_flat[sIdx];
                baseTerms[w2] <== bm.out;
            }
            component baseSum = SumN(M);
            for (var w3 = 0; w3 < M; w3++) {
                baseSum.in[w3] <== baseTerms[w3];
            }

            signal impTerms[N_MAX];
            signal confTerms[N_MAX];
            for (var g = 0; g < N_MAX; g++) {
                var mIdx = ((o * N_MAX) + g) * N_MAX + f;

                component im = Q16Mul();
                im.a <== imp_flat[mIdx];
                im.b <== s_prev[o][g];
                impTerms[g] <== im.out;

                component cm = Q16Mul();
                cm.a <== conf_flat[mIdx];
                cm.b <== s_prev[o][g];
                confTerms[g] <== cm.out;
            }

            component impSum = SumN(N_MAX);
            component confSum = SumN(N_MAX);
            for (var g2 = 0; g2 < N_MAX; g2++) {
                impSum.in[g2] <== impTerms[g2];
                confSum.in[g2] <== confTerms[g2];
            }

            component impScaled = Q16Mul();
            impScaled.a <== alpha_imp;
            impScaled.b <== impSum.out;

            component confScaled = Q16Mul();
            confScaled.a <== alpha_conflict;
            confScaled.b <== confSum.out;

            signal preScore;
            preScore <== baseSum.out + impScaled.out;

            component negFlag = LessThan(32);
            negFlag.in[0] <== preScore;
            negFlag.in[1] <== confScaled.out;

            // signed score = preScore - confScaled.out
            // represented as sign + absolute magnitude for circuit-safe signed sigmoid.
            signal scoreAbs;
            scoreAbs <== (preScore - confScaled.out) * (1 - negFlag.out)
                      + (confScaled.out - preScore) * negFlag.out;

            component betaMul = Q16Mul();
            betaMul.a <== beta;
            betaMul.b <== scoreAbs;

            component xZero = IsZero();
            xZero.in <== betaMul.out;

            component sig = ApproxSigmoidQ16Signed();
            sig.x_abs <== betaMul.out;
            sig.x_is_neg <== negFlag.out;
            sig.x_is_zero <== xZero.out;

            // keep only valid facts for effective objects
            component factValid = LessThan(8);
            factValid.in[0] <== f;
            factValid.in[1] <== fact_count_by_object[o];

            signal mask;
            mask <== is_effective_by_object[o] * factValid.out;

            s_out[o][f] <== sig.y * mask;
        }
    }

    // update t as weighted average of s over support
    for (var w4 = 0; w4 < M; w4++) {
        signal numTerms[OF];
        signal denTerms[OF];

        for (var p = 0; p < OF; p++) {
            var o2 = p / N_MAX;
            var f2 = p % N_MAX;
            var idx2 = ((o2 * N_MAX) + f2) * M + w4;

            component nm = Q16Mul();
            nm.a <== support_flat[idx2];
            nm.b <== s_out[o2][f2];
            numTerms[p] <== nm.out;
            denTerms[p] <== support_flat[idx2];
        }

        component numSum = SumN(OF);
        component denSum = SumN(OF);
        for (var p2 = 0; p2 < OF; p2++) {
            numSum.in[p2] <== numTerms[p2];
            denSum.in[p2] <== denTerms[p2];
        }

        component div = SafeDivNonNeg();
        div.num <== numSum.out;
        div.den <== denSum.out;
        div.fallback <== t_in[w4];

        component tClamp = Q16Clamp01();
        tClamp.in <== div.out;
        t_out[w4] <== tClamp.out;
    }
}


template TruthFinderMain() {
    // ---------------- Inputs (from truthfinder_circom_input.json) ----------------
    signal input K;

    // params_q16
    signal input t0;
    signal input beta;
    signal input gamma;
    signal input alpha_imp;
    signal input alpha_conflict;
    signal input cand_decay;
    signal input min_tau_scale;

    // object_meta
    signal input fact_count_by_object[K_MAX];
    signal input is_effective_by_object[K_MAX];
    signal input top1_choice_flat[K_MAX * M]; // compatibility-only (not used in core constraints)

    // circom_arrays
    signal input dep_avg[M];
    signal input support_flat[K_MAX * N_MAX * M];
    signal input imp_flat[K_MAX * N_MAX * N_MAX];
    signal input conf_flat[K_MAX * N_MAX * N_MAX];

    // ---------------- Public outputs ----------------
    signal output best_model_idx;
    signal output best_model_score_q16;
    signal output winning_fact_idx_by_object[K_MAX];

    // cand_decay currently affects off-circuit support construction in this pipeline.
    // Keep compatibility without changing core circuit semantics.
    signal _cand_decay_keep;
    _cand_decay_keep <== cand_decay;

    // ---------------- Metadata consistency constraints ----------------
    component kBound = LessThan(8);
    kBound.in[0] <== K;
    kBound.in[1] <== K_MAX + 1;
    kBound.out === 1;

    for (var o = 0; o < K_MAX; o++) {
        component oLtK = LessThan(8);
        oLtK.in[0] <== o;
        oLtK.in[1] <== K;

        // Required by spec:
        // o < K  => is_effective=1
        // o >= K => is_effective=0
        is_effective_by_object[o] === oLtK.out;

        component fcBound = LessThan(8);
        fcBound.in[0] <== fact_count_by_object[o];
        fcBound.in[1] <== N_MAX + 1;
        fcBound.out === 1;

        component fcZero = IsZero();
        fcZero.in <== fact_count_by_object[o];

        // Effective object must have at least one fact.
        // is_effective=1 => fact_count != 0
        is_effective_by_object[o] * fcZero.out === 0;

        // Padding object must have fact_count = 0.
        // (1-is_effective)=1 => fact_count == 0
        fact_count_by_object[o] * (1 - is_effective_by_object[o]) === 0;
    }

    // top1_choice_flat is intentionally not constrained in-circuit because dep_avg
    // is already the formal circuit input derived off-circuit from top1 choices.

    // ---------------- Iterative state ----------------
    signal t_state[ITER_N + 1][M];
    signal s_state[ITER_N + 1][K_MAX][N_MAX];

    // init
    for (var w = 0; w < M; w++) {
        t_state[0][w] <== t0;
    }
    for (var o2 = 0; o2 < K_MAX; o2++) {
        for (var f = 0; f < N_MAX; f++) {
            s_state[0][o2][f] <== 0;
        }
    }

    // fixed 25 rounds (project-fixed ITER_N)
    component rounds[ITER_N];
    for (var it = 0; it < ITER_N; it++) {
        rounds[it] = TruthFinderRound();

        for (var w2 = 0; w2 < M; w2++) {
            rounds[it].t_in[w2] <== t_state[it][w2];
        }
        for (var o3 = 0; o3 < K_MAX; o3++) {
            for (var f2 = 0; f2 < N_MAX; f2++) {
                rounds[it].s_prev[o3][f2] <== s_state[it][o3][f2];
            }
        }

        rounds[it].beta <== beta;
        rounds[it].gamma <== gamma;
        rounds[it].alpha_imp <== alpha_imp;
        rounds[it].alpha_conflict <== alpha_conflict;
        rounds[it].min_tau_scale <== min_tau_scale;

        for (var o4 = 0; o4 < K_MAX; o4++) {
            rounds[it].fact_count_by_object[o4] <== fact_count_by_object[o4];
            rounds[it].is_effective_by_object[o4] <== is_effective_by_object[o4];
        }

        for (var w3 = 0; w3 < M; w3++) {
            rounds[it].dep_avg[w3] <== dep_avg[w3];
        }
        for (var i = 0; i < K_MAX * N_MAX * M; i++) {
            rounds[it].support_flat[i] <== support_flat[i];
        }
        for (var j = 0; j < K_MAX * N_MAX * N_MAX; j++) {
            rounds[it].imp_flat[j] <== imp_flat[j];
            rounds[it].conf_flat[j] <== conf_flat[j];
        }

        for (var w4 = 0; w4 < M; w4++) {
            t_state[it + 1][w4] <== rounds[it].t_out[w4];
        }
        for (var o5 = 0; o5 < K_MAX; o5++) {
            for (var f3 = 0; f3 < N_MAX; f3++) {
                s_state[it + 1][o5][f3] <== rounds[it].s_out[o5][f3];
            }
        }
    }

    // ---------------- model argmax + formal constraints ----------------
    component argM = ArgMaxModelWithTieBreak();
    for (var w5 = 0; w5 < M; w5++) {
        argM.score[w5] <== t_state[ITER_N][w5];
    }
    best_model_idx <== argM.bestIdx;
    best_model_score_q16 <== argM.bestScore;

    // Explicitly enforce best_model_score >= all model scores.
    for (var w6 = 0; w6 < M; w6++) {
        component geM = LessThan(32);
        geM.in[0] <== t_state[ITER_N][w6];
        geM.in[1] <== best_model_score_q16 + 1;
        geM.out === 1;
    }

    // ---------------- fact argmax per object + formal constraints ----------------
    component argF[K_MAX];
    for (var o6 = 0; o6 < K_MAX; o6++) {
        argF[o6] = ArgMaxFactWithTieBreak();
        for (var f4 = 0; f4 < N_MAX; f4++) {
            argF[o6].score[f4] <== s_state[ITER_N][o6][f4];
        }

        // effective object -> argmax index; padding object -> forced 0
        winning_fact_idx_by_object[o6] <== argF[o6].bestIdx * is_effective_by_object[o6];

        // explicit padding constraint
        winning_fact_idx_by_object[o6] * (1 - is_effective_by_object[o6]) === 0;

        // effective object winner index must be < fact_count
        component winnerLtCount = LessThan(8);
        winnerLtCount.in[0] <== winning_fact_idx_by_object[o6];
        winnerLtCount.in[1] <== fact_count_by_object[o6];
        is_effective_by_object[o6] * (1 - winnerLtCount.out) === 0;

        // explicit max-score constraints against all valid facts
        for (var f5 = 0; f5 < N_MAX; f5++) {
            component fValid = LessThan(8);
            fValid.in[0] <== f5;
            fValid.in[1] <== fact_count_by_object[o6];

            component geF = LessThan(32);
            geF.in[0] <== s_state[ITER_N][o6][f5];
            geF.in[1] <== argF[o6].bestScore + 1;

            // enforce only on valid facts of effective objects
            signal mustHold;
            mustHold <== is_effective_by_object[o6] * fValid.out;
            mustHold * (1 - geF.out) === 0;
        }
    }
}

component main = TruthFinderMain();

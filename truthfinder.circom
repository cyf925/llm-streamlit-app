pragma circom 2.2.3;

include "circomlib/circuits/comparators.circom";

// Project freeze note (semantics-first):
// - This circuit is the formal frozen TruthFinder circuit semantics, not a bit-match
//   floating implementation of TruthFinder.py.
// - It uses fixed Q16 arithmetic, frozen _tau_circuit (ApproxTauQ16), and frozen
//   _sigmoid_circuit (ApproxSigmoidQ16Signed).
// - For _sigmoid_circuit on the negative half-axis, the official frozen definition is:
//     y = d - floor((c * |x|) / 65536)
//   (not an alternative signed-floor/secant interpretation).
// - Future TruthFinder_circuit_ref.py implementations MUST match this circuit version exactly.

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
    var Q16 = 65536;

    signal input a;
    signal input b;
    signal output out;

    signal q;
    signal r;

    q <-- (a * b) \ Q16;
    r <-- (a * b) % Q16;

    a * b === q * Q16 + r;

    component rlt = LessThan(17);
    rlt.in[0] <== r;
    rlt.in[1] <== Q16;
    rlt.out === 1;

    out <== q;
}


template Q16Clamp01() {
    var Q16 = 65536;
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
    q <-- num \ denEff;
    r <-- num % denEff;
    num === q * denEff + r;

    component rlt = LessThan(32);
    rlt.in[0] <== r;
    rlt.in[1] <== denEff;
    rlt.out === 1;

       signal qPart;
    signal fbPart;
    qPart <== q * (1 - iz.out);
    fbPart <== fallback * iz.out;
    out <== qPart + fbPart;
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
    var M = 4;
    signal input score[M];
    signal output bestIdx;
    signal output bestScore;

    signal bestVal[M];
    signal bestId[M];
    component mx[M - 1];

    bestVal[0] <== score[0];
    bestId[0] <== 0;

    for (var i = 1; i < M; i++) {
        mx[i - 1] = MaxWithTieBreak();
        mx[i - 1].curVal <== bestVal[i - 1];
        mx[i - 1].curIdx <== bestId[i - 1];
        mx[i - 1].nxtVal <== score[i];
        mx[i - 1].nxtIdx <== i;

        bestVal[i] <== mx[i - 1].outVal;
        bestId[i] <== mx[i - 1].outIdx;
    }

    bestIdx <== bestId[M - 1];
    bestScore <== bestVal[M - 1];
}


template ArgMaxFactWithTieBreak() {
    var N_MAX = 12;
    signal input score[N_MAX];
    signal output bestIdx;
    signal output bestScore;

    signal bestVal[N_MAX];
    signal bestId[N_MAX];
    component mx[N_MAX - 1];

    bestVal[0] <== score[0];
    bestId[0] <== 0;

    for (var i = 1; i < N_MAX; i++) {
        mx[i - 1] = MaxWithTieBreak();
        mx[i - 1].curVal <== bestVal[i - 1];
        mx[i - 1].curIdx <== bestId[i - 1];
        mx[i - 1].nxtVal <== score[i];
        mx[i - 1].nxtIdx <== i;

        bestVal[i] <== mx[i - 1].outVal;
        bestId[i] <== mx[i - 1].outIdx;
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

    signal t0;
    signal t1;
    signal t2;
    signal t3;
    signal t4;
    signal t5;
    signal raw;

    t0 <== g0 * y0;
    t1 <== g1 * y1;
    t2 <== g2 * y2;
    t3 <== g3 * y3;
    t4 <== g4 * y4;
    t5 <== g5 * y5;
    raw <== t0 + t1 + t2 + t3 + t4 + t5;

    // Capped to 262144 (4.0 in Q16), per freeze spec final engineering bound.
    component ltCap = LessThan(32);
    ltCap.in[0] <== raw;
    ltCap.in[1] <== 262145;

    tau <== 262144 + ltCap.out * (raw - 262144);
}


template ApproxSigmoidQ16Signed() {
    // Formal frozen _sigmoid_circuit (Approximation Spec Freeze v1), signed input.
    //
    // This implementation is the project official frozen spec and MUST be mirrored
    // exactly in future TruthFinder_circuit_ref.py (do not reinterpret with another
    // signed-floor/secant formula).
    //
    // Signed x is represented as:
    //   x = sign ? (-x_abs) : (+x_abs), where x_abs >= 0 and sign in {0,1}.
    //   x_is_zero marks x==0 and MUST output exact midpoint 32768.
    //
    // Official frozen piecewise semantics:
    // - Negative half-axis uses: y = d - floor((c * |x|) / 65536).
    // - Positive half-axis uses: y = floor((c * x) / 65536) + d.
    //
    // Frozen boundary ownership (must match Python reference item-by-item):
    //   x=-6 -> left saturation segment
    //   x=-4 -> (-6,-4]
    //   x=-2 -> (-4,-2]
    //   x=-1 -> (-2,-1]
    //   x= 0 -> midpoint 32768
    //   x= 1 -> (0,1]
    //   x= 2 -> (1,2]
    //   x= 4 -> (2,4]
    //   x= 6 -> (4,6]
    signal input x_abs;
    signal input x_is_neg;
    signal input x_is_zero;
    signal output y;

    x_is_neg * (x_is_neg - 1) === 0;
    x_is_zero * (x_is_zero - 1) === 0;
    x_is_neg * x_is_zero === 0;

    // |x| boundaries: 1,2,4,6 in Q16 units
    component lt1 = LessThan(32); // |x| < 1
    lt1.in[0] <== x_abs;
    lt1.in[1] <== 65536;

    component lt2 = LessThan(32); // |x| < 2
    lt2.in[0] <== x_abs;
    lt2.in[1] <== 131072;

    component lt4 = LessThan(32); // |x| < 4
    lt4.in[0] <== x_abs;
    lt4.in[1] <== 262144;

    component lt6 = LessThan(32); // |x| < 6
    lt6.in[0] <== x_abs;
    lt6.in[1] <== 393216;

    component le1 = LessThan(32); // |x| <= 1
    le1.in[0] <== x_abs;
    le1.in[1] <== 65537;

    component le2 = LessThan(32); // |x| <= 2
    le2.in[0] <== x_abs;
    le2.in[1] <== 131073;

    component le4 = LessThan(32); // |x| <= 4
    le4.in[0] <== x_abs;
    le4.in[1] <== 262145;

    component le6 = LessThan(32); // |x| <= 6
    le6.in[0] <== x_abs;
    le6.in[1] <== 393217;

    // interval gates over |x| for negative branch:
    // S4:(0,1), S3:[1,2), S2:[2,4), S1:[4,6), S0:[6,+inf)
    signal gNegA;
    signal gNegB;
    signal gNegC;
    signal gNegD;
    signal gNegE;

    gNegA <== lt1.out;
    gNegB <== lt2.out - lt1.out;
    gNegC <== lt4.out - lt2.out;
    gNegD <== lt6.out - lt4.out;
    gNegE <== 1 - lt6.out;

    // interval gates over |x| for positive branch:
    // S5:(0,1], S6:(1,2], S7:(2,4], S8:(4,6], S9:(6,+inf)
    signal gPosA;
    signal gPosB;
    signal gPosC;
    signal gPosD;
    signal gPosE;

    gPosA <== le1.out;
    gPosB <== le2.out - le1.out;
    gPosC <== le4.out - le2.out;
    gPosD <== le6.out - le4.out;
    gPosE <== 1 - le6.out;

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
    // S1: (-6,-4]       -> 3212 - floor( 508*|x|/65536)
    // S2: (-4,-2]       -> 14445 - floor(3317*|x|/65536)
    // S3: (-2,-1]       -> 27439 - floor(9813*|x|/65536)
    // S4: (-1,0]        -> 32768 - floor(15143*|x|/65536)
    signal yNegTerm0;
    signal yNegTerm1;
    signal yNegTerm2;
    signal yNegTerm3;
    signal yNegTerm4;
    signal yNeg;

    yNegTerm0 <== gNegE * 162;
    yNegTerm1 <== gNegD * (3212 - m508.out);
    yNegTerm2 <== gNegC * (14445 - m3317.out);
    yNegTerm3 <== gNegB * (27439 - m9813.out);
    yNegTerm4 <== gNegA * (32768 - m15143.out);
    yNeg <== yNegTerm0 + yNegTerm1 + yNegTerm2 + yNegTerm3 + yNegTerm4;

    // positive branch:
    // S9: x>=6          -> 65374
    // S8: (4,6]         -> floor( 508*x/65536)+62324
    // S7: (2,4]         -> floor(3317*x/65536)+51091
    // S6: (1,2]         -> floor(9813*x/65536)+38097
    // S5: (0,1]         -> floor(15143*x/65536)+32768
    signal yPosTerm0;
    signal yPosTerm1;
    signal yPosTerm2;
    signal yPosTerm3;
    signal yPosTerm4;
    signal yPos;

    yPosTerm0 <== gPosE * 65374;
    yPosTerm1 <== gPosD * (62324 + m508.out);
    yPosTerm2 <== gPosC * (51091 + m3317.out);
    yPosTerm3 <== gPosB * (38097 + m9813.out);
    yPosTerm4 <== gPosA * (32768 + m15143.out);
    yPos <== yPosTerm0 + yPosTerm1 + yPosTerm2 + yPosTerm3 + yPosTerm4;

    // x==0 exact midpoint 32768; otherwise choose sign branch.
    signal yNoZeroNeg;
    signal yNoZeroPos;
    signal yNoZero;
    yNoZeroNeg <== x_is_neg * yNeg;
    yNoZeroPos <== (1 - x_is_neg) * yPos;
    yNoZero <== yNoZeroNeg + yNoZeroPos;

    signal yRawZero;
    signal yRawNonZero;
    signal yRaw;
    yRawZero <== x_is_zero * 32768;
    yRawNonZero <== (1 - x_is_zero) * yNoZero;
    yRaw <== yRawZero + yRawNonZero;

    // Final clamp to [0,65536] upper bound (piecewise formula is already non-negative).
    component yLe1 = LessThan(32);
    yLe1.in[0] <== yRaw;
    yLe1.in[1] <== 65537;

    y <== 65536 + yLe1.out * (yRaw - 65536);
}

template TruthFinderRound() {
    var M = 4;
    var K_MAX = 15;
    var N_MAX = 12;
    var Q16 = 65536;
    var OF = K_MAX * N_MAX;
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
    signal oneMinus[M];
    signal scaleW[M];
    component tauApprox[M];
    component gmul[M];
    component ltScale[M];
    component damp[M];

    signal baseTerms[K_MAX][N_MAX][M];
    component bm[K_MAX][N_MAX][M];
    component baseSum[K_MAX][N_MAX];

    signal impTerms[K_MAX][N_MAX][N_MAX];
    signal confTerms[K_MAX][N_MAX][N_MAX];
    component im[K_MAX][N_MAX][N_MAX];
    component cm[K_MAX][N_MAX][N_MAX];
    component impSum[K_MAX][N_MAX];
    component confSum[K_MAX][N_MAX];
    component impScaled[K_MAX][N_MAX];
    component confScaled[K_MAX][N_MAX];
    signal preScore[K_MAX][N_MAX];
    component negFlag[K_MAX][N_MAX];
    signal scoreAbs[K_MAX][N_MAX];
    signal scorePosPart[K_MAX][N_MAX];
    signal scoreNegPart[K_MAX][N_MAX];
    component betaMul[K_MAX][N_MAX];
    component xZero[K_MAX][N_MAX];
    component sig[K_MAX][N_MAX];
    component factValid[K_MAX][N_MAX];
    signal mask[K_MAX][N_MAX];

    signal numTerms[M][OF];
    signal denTerms[M][OF];
    component nm[M][OF];
    component numSum[M];
    component denSum[M];
    component div[M];
    component tClamp[M];

    // tau and dependency damping
    for (var w = 0; w < M; w++) {
        tauApprox[w] = ApproxTauQ16();
        tauApprox[w].t <== t_in[w];

        gmul[w] = Q16Mul();
        gmul[w].a <== gamma;
        gmul[w].b <== dep_avg[w];

        oneMinus[w] <== Q16 - gmul[w].out;

        ltScale[w] = LessThan(32);
        ltScale[w].in[0] <== oneMinus[w];
        ltScale[w].in[1] <== min_tau_scale;

        scaleW[w] <== oneMinus[w] + ltScale[w].out * (min_tau_scale - oneMinus[w]);

        damp[w] = Q16Mul();
        damp[w].a <== tauApprox[w].tau;
        damp[w].b <== scaleW[w];
        tauW[w] <== damp[w].out;
    }

    // update s
    for (var o = 0; o < K_MAX; o++) {
        for (var f = 0; f < N_MAX; f++) {
            for (var w2 = 0; w2 < M; w2++) {
                var sIdx = ((o * N_MAX) + f) * M + w2;
                bm[o][f][w2] = Q16Mul();
                bm[o][f][w2].a <== tauW[w2];
                bm[o][f][w2].b <== support_flat[sIdx];
                baseTerms[o][f][w2] <== bm[o][f][w2].out;
            }
            baseSum[o][f] = SumN(M);
            for (var w3 = 0; w3 < M; w3++) {
                baseSum[o][f].in[w3] <== baseTerms[o][f][w3];
            }

            for (var g = 0; g < N_MAX; g++) {
                var mIdx = ((o * N_MAX) + g) * N_MAX + f;

                im[o][f][g] = Q16Mul();
                im[o][f][g].a <== imp_flat[mIdx];
                im[o][f][g].b <== s_prev[o][g];
                impTerms[o][f][g] <== im[o][f][g].out;

                cm[o][f][g] = Q16Mul();
                cm[o][f][g].a <== conf_flat[mIdx];
                cm[o][f][g].b <== s_prev[o][g];
                confTerms[o][f][g] <== cm[o][f][g].out;
            }

            impSum[o][f] = SumN(N_MAX);
            confSum[o][f] = SumN(N_MAX);
            for (var g2 = 0; g2 < N_MAX; g2++) {
                impSum[o][f].in[g2] <== impTerms[o][f][g2];
                confSum[o][f].in[g2] <== confTerms[o][f][g2];
            }

            impScaled[o][f] = Q16Mul();
            impScaled[o][f].a <== alpha_imp;
            impScaled[o][f].b <== impSum[o][f].out;

            confScaled[o][f] = Q16Mul();
            confScaled[o][f].a <== alpha_conflict;
            confScaled[o][f].b <== confSum[o][f].out;

            preScore[o][f] <== baseSum[o][f].out + impScaled[o][f].out;

            negFlag[o][f] = LessThan(32);
            negFlag[o][f].in[0] <== preScore[o][f];
            negFlag[o][f].in[1] <== confScaled[o][f].out;

            // signed score = preScore - confScaled.out
            // represented as sign + absolute magnitude for circuit-safe signed sigmoid.
            scorePosPart[o][f] <== (preScore[o][f] - confScaled[o][f].out) * (1 - negFlag[o][f].out);
            scoreNegPart[o][f] <== (confScaled[o][f].out - preScore[o][f]) * negFlag[o][f].out;
            scoreAbs[o][f] <== scorePosPart[o][f] + scoreNegPart[o][f];

            betaMul[o][f] = Q16Mul();
            betaMul[o][f].a <== beta;
            betaMul[o][f].b <== scoreAbs[o][f];

            xZero[o][f] = IsZero();
            xZero[o][f].in <== betaMul[o][f].out;

            sig[o][f] = ApproxSigmoidQ16Signed();
            sig[o][f].x_abs <== betaMul[o][f].out;
            sig[o][f].x_is_neg <== negFlag[o][f].out;
            sig[o][f].x_is_zero <== xZero[o][f].out;

            // keep only valid facts for effective objects
            factValid[o][f] = LessThan(8);
            factValid[o][f].in[0] <== f;
            factValid[o][f].in[1] <== fact_count_by_object[o];

            mask[o][f] <== is_effective_by_object[o] * factValid[o][f].out;

            s_out[o][f] <== sig[o][f].y * mask[o][f];
        }
    }

    // update t as weighted average of s over support
    for (var w4 = 0; w4 < M; w4++) {
        for (var p = 0; p < OF; p++) {
            var o2 = p \ N_MAX;
            var f2 = p % N_MAX;
            var idx2 = ((o2 * N_MAX) + f2) * M + w4;

            nm[w4][p] = Q16Mul();
            nm[w4][p].a <== support_flat[idx2];
            nm[w4][p].b <== s_out[o2][f2];
            numTerms[w4][p] <== nm[w4][p].out;
            denTerms[w4][p] <== support_flat[idx2];
        }

        numSum[w4] = SumN(OF);
        denSum[w4] = SumN(OF);
        for (var p2 = 0; p2 < OF; p2++) {
            numSum[w4].in[p2] <== numTerms[w4][p2];
            denSum[w4].in[p2] <== denTerms[w4][p2];
        }

        div[w4] = SafeDivNonNeg();
        div[w4].num <== numSum[w4].out;
        div[w4].den <== denSum[w4].out;
        div[w4].fallback <== t_in[w4];

        tClamp[w4] = Q16Clamp01();
        tClamp[w4].in <== div[w4].out;
        t_out[w4] <== tClamp[w4].out;
    }
}


template TruthFinderMain() {
    var M = 4;
    var K_MAX = 15;
    var N_MAX = 12;
    var ITER_N = 25;
    var Q16 = 65536;
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
    component oLtK[K_MAX];
    component fcBound[K_MAX];
    component fcZero[K_MAX];
    component geM[M];
    component winnerLtCount[K_MAX];
    component fValid[K_MAX][N_MAX];
    component geF[K_MAX][N_MAX];

    signal mustHold[K_MAX][N_MAX];

    kBound.in[0] <== K;
    kBound.in[1] <== K_MAX + 1;
    kBound.out === 1;

    for (var o = 0; o < K_MAX; o++) {
        oLtK[o] = LessThan(8);
        oLtK[o].in[0] <== o;
        oLtK[o].in[1] <== K;

        // Required by spec:
        // o < K  => is_effective=1
        // o >= K => is_effective=0
        is_effective_by_object[o] === oLtK[o].out;

        fcBound[o] = LessThan(8);
        fcBound[o].in[0] <== fact_count_by_object[o];
        fcBound[o].in[1] <== N_MAX + 1;
        fcBound[o].out === 1;

        fcZero[o] = IsZero();
        fcZero[o].in <== fact_count_by_object[o];

        // Effective object must have at least one fact.
        // is_effective=1 => fact_count != 0
        is_effective_by_object[o] * fcZero[o].out === 0;

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
        geM[w6] = LessThan(32);
        geM[w6].in[0] <== t_state[ITER_N][w6];
        geM[w6].in[1] <== best_model_score_q16 + 1;
        geM[w6].out === 1;
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
        winnerLtCount[o6] = LessThan(8);
        winnerLtCount[o6].in[0] <== winning_fact_idx_by_object[o6];
        winnerLtCount[o6].in[1] <== fact_count_by_object[o6];
        is_effective_by_object[o6] * (1 - winnerLtCount[o6].out) === 0;

        // explicit max-score constraints against all valid facts
        for (var f5 = 0; f5 < N_MAX; f5++) {
            fValid[o6][f5] = LessThan(8);
            fValid[o6][f5].in[0] <== f5;
            fValid[o6][f5].in[1] <== fact_count_by_object[o6];

            geF[o6][f5] = LessThan(32);
            geF[o6][f5].in[0] <== s_state[ITER_N][o6][f5];
            geF[o6][f5].in[1] <== argF[o6].bestScore + 1;

            // enforce only on valid facts of effective objects
            mustHold[o6][f5] <== is_effective_by_object[o6] * fValid[o6][f5].out;
            mustHold[o6][f5] * (1 - geF[o6][f5].out) === 0;
        }
    }
}

component main = TruthFinderMain();

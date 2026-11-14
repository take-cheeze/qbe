# Loop vectorizer (vec.c, `-V`)

`qbe -V` vectorizes simple counted loops for amd64_sysv (SSE2). It runs after the SSA passes and before
instruction selection, so it sees the final SSA form and the SSA form of the loop's index. Other targets are
not changed: the pass returns at once unless the target is amd64_sysv, and `-n` (no floating-point registers)
turns it off.

## What it does

A loop over `i` whose body is element-wise over arrays becomes a vector loop in front of the scalar loop.
The scalar loop stays as it was and runs the remaining iterations (`n` mod VF of them, or all of them when the
vector loop is not entered):

    P:   jmp H                               P is the entry of H from outside the loop
    Chk: ov = overlap test; jnz ov, H, Hv   only when the loop stores to one array and reads another
    Hv:  iv = phi(P: i0, Bv: ivn); cc = iv + VF <= n; jnz cc, Bv, H
    Bv:  the body on the lanes iv .. iv+VF-1; ivn = iv + VF; jmp Hv
    H:   the original header and scalar body, unchanged

VF is 4 for int32 and float lanes, 2 for double lanes. A loop mixing doubles with int32 or float lanes is refused.

Vector values are temporaries of class `Kd` with `Tmp.vec` set; they live in xmm registers and are spilled
into 16-byte slots. Every `Kd` spill slot is 16 bytes (spill.c), and copies of a `Kd` value between registers
or slots move the whole register (`movaps`, `movups`), since a `movsd` would keep only the low lane.

The vector ops (ops.h, `vec*`) are: `vecload`, `vecstore` (16 bytes, unaligned); `vecbcastw/s/d` (broadcast of
a scalar into all lanes); lane arithmetic `vecadd/sub/mul/div` for float and double, and `vecadd/sub/and/or/xor`
for int32 (SSE2 has no int32 multiply; `pmulld` is SSE4.1).

## Loops it accepts

- The header `H` has one phi `i` (class w), one comparison `c = csltw i, n` with `n` a w temporary defined
  outside the loop or a constant, and `jnz c, B, E`.
- `H` has two predecessors: the entry `P` and the latch. The body from `B` to the latch is a chain of blocks
  with one predecessor each and no phi; the latch's last instruction is `i2 = add i, 1` and it jumps to `H`.
- Body instructions: index arithmetic on `i` (extsw, extuw, mul or shl by a constant); an address
  `add base, off` with `base` loop-invariant and `off` an index whose step per `i` equals the element size;
  a load of w, s or d at such an address; w add/sub/and/or/xor and s or d add/sub/mul/div on lanes, invariant
  scalars and constants (at least one lane operand); a store of a lane of the same width.
- Nothing else: no calls, no other memory accesses, no phi besides `i`, and no value defined in the body is
  used after the loop.

## Aliasing

When the loop stores to an array and touches another array, `Chk` tests at run time whether the index ranges
`[base, base + n*size)` overlap. If one does, the scalar loop runs. The ranges start at the base, so the test
is conservative. A base accessed with two element sizes is refused.

## Limits

- No reductions (a phi besides `i`), no integer multiply or divide, no index arithmetic on lanes (`a[i] = i`).
- Only a unit step of `i`, and only `csltw` as the exit test. A downward loop is not recognized.
- At most 8 distinct bases per loop.
- The bound test is `iv + VF <= n` in w arithmetic: a bound near INT_MAX can overflow it.
- There is no profitability model and no minimum trip count. The overlap test runs once per loop entry.

## Debugging

`qbe -V -d V file.ssa` prints, for each candidate, either `vec: <fn>: loop at @<id>, <lanes> lanes, <bases> bases`
(then the function's IR after the transform, on stderr) or `vec: refused at vec.c:<line>`, the line of the
check that refused it. With `-d`, qbe writes no assembly, so run the debug build separately from the one that produces code.

## Tests

`test/vec-*.ssa` run through the usual harness. `tools/test-vec.sh` runs the whole amd64 suite with `-V`, so the
loops of every test go through the pass; `tools/test.sh` runs it with the pass off.

## Measurement

A float add loop over 2^20 elements, 200 passes, on the sandbox's x86-64 (best of 5, `-O2` driver):
scalar 0.140 s, vector 0.088 s. The loop is memory-bound, so the gain is smaller than the lane count.

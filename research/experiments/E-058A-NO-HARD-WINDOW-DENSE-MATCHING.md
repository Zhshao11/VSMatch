# E-058A No-Hard-Window Dense Matching

Status: `ENGINEERING_STARTED`

E-058A freezes E046A-R2 frontend/reference and replaces hard fine windows with
full target token distributions. It compares dense argmax/softargmax, top-K
distribution residual decoding, and 2-4 round sparse distribution plus
low-dimensional affine feedback. No GDROS baseline, released weights, or prior
experiment checkpoint is writable.

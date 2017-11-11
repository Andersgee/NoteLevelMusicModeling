module TRAIN

include("grid.jl")
include("optimizer.jl")

function train(gridsize, unrollsteps, L, d, bsz, seqlen)
  

  N, C, fn, bn = GRID.linearindexing(gridsize)
  Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W,b,g,mi,mo,hi,ho,Hi,WHib, hiNmiN,hoNmoN = GRID.gridvars(N,C,d,bsz, unrollsteps)
  ∇Wenc, Σ∇Wenc, Σ∇benc, ∇Wdec, Σ∇Wdec, Σ∇bdec = GRID.∇encodevars(L,d,N,bsz)
  ∇W, Σ∇W, Σ∇b, ∇hi, ∇ho, ∇Hi, Σ∇Hi, ∇WHib,∇mi,∇mo, ∇hiNmiN,∇hoNmoN = GRID.∇gridvars(N,C,d,bsz, unrollsteps)
  Wm, Wv, bm, bv, mWenc, vWenc, mbenc, vbenc, mWdec, vWdec, mbdec, vbdec = GRID.optimizevars(d,N,L)
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,unrollsteps,seqlen)

  for s=1:unrollsteps
    x[s].=rand(L,bsz).>0.75
    t[s].=rand(L,bsz).>0.75
  end

  E=log(2)*L
  gradientstep=0

  for K=1:10

    #fprop
    GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
    GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
    GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)

    E = GRID.logistic_xent(E,t[unrollsteps],z[unrollsteps],bsz)
    (K%100==0) && println(K, "  LastE: ", E)

    #bprop
    GRID.∇decode!(∇z, z, t, ∇Wdec, Σ∇Wdec, C, Σ∇bdec,Wdec,∇ho,∇mo,N,d, hoNmoN,∇hoNmoN)
    GRID.∇hyperlstm!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo, unrollsteps)
    GRID.∇encode!(x, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi,∇hiNmiN)

    gradientstep+=1
    OPTIMIZER.optimize_Wencdec!(N, Wenc, Σ∇Wenc, mWenc, vWenc, gradientstep)
    OPTIMIZER.optimize_bencdec!(N, benc, Σ∇benc, mbenc, vbenc, gradientstep)
    OPTIMIZER.optimize_W!(N, W, Σ∇W, Wm, Wv, gradientstep, 0.01)
    OPTIMIZER.maxnormconstrain_W!(W, N, 4)
    OPTIMIZER.optimize_b!(N, b, Σ∇b, bm, bv, gradientstep)
    OPTIMIZER.optimize_Wencdec!(N, Wdec, Σ∇Wdec, mWdec, vWdec, gradientstep)
    OPTIMIZER.optimize_bencdec!(N, bdec, Σ∇bdec, mbdec, vbdec, gradientstep)
  end
  
  println("W magnitudes:")
  for n=1:N, gate=1:4
    incoming_magnitude=sqrt.(sum(W[n][gate].^2, 2))
    println(incoming_magnitude)
  end

  println("Final output:")
  println(round.(GRID.σ(z[unrollsteps]),1))
  println("Target:")
  println(t[unrollsteps])
end

function main()
  gridsize=[2,2,2]
  unrollsteps=10
  L=10
  d=10
  bsz=1
  seqlen=5

  train(gridsize, unrollsteps, L, d, bsz, seqlen)
end

end

@time TRAIN.main()
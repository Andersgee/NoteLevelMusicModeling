module TRAIN

include("grid.jl")
include("optimizer.jl")
include("dataloader.jl")
include("checkpoint.jl")

function train(data, gridsize, unrollsteps, L, d, bsz, seqlen)
  #pre-allocate
  N, C, fn, bn = GRID.linearindexing(gridsize)
  Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W,b,g,mi,mo,hi,ho,Hi,WHib, hiNmiN,hoNmoN = GRID.gridvars(N,C,d,bsz, unrollsteps)
  ∇Wenc, Σ∇Wenc, Σ∇benc, ∇Wdec, Σ∇Wdec, Σ∇bdec = GRID.∇encodevars(L,d,N,bsz)
  ∇W, Σ∇W, Σ∇b, ∇hi, ∇ho, ∇Hi, Σ∇Hi, ∇WHib,∇mi,∇mo, ∇hiNmiN,∇hoNmoN = GRID.∇gridvars(N,C,d,bsz, unrollsteps)
  Wm, Wv, bm, bv, mWenc, vWenc, mbenc, vbenc, mWdec, vWdec, mbdec, vbdec = GRID.optimizevars(d,N,L)
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,unrollsteps,seqlen)

  gradientstep=0
  E = zeros(0)
  #Ev=log(2)*L
  Ev=L/2

  fname1 = string("trained/H",N,"lstm_u",unrollsteps,".jld")
  fname2 = string("trained/H",N,"lstm_u",unrollsteps,"_opt.jld")

  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)
  mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, E = CHECKPOINT.load_optimizevars(fname2)
  Ev=E[end]

  #while Ev > 1
  while true
    DATALOADER.get_batch!(batch, seqlen, bsz, data)
    GRID.reset_state!(mi,hi,unrollsteps)

    for batchstep=1:unrollsteps:seqlen-unrollsteps
      DATALOADER.get_partialbatch!(x,t,batch,unrollsteps,batchstep,bsz)
      GRID.recur_state!(mi,hi,unrollsteps)

      GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
      GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
      GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)
      GRID.∇decode!(∇z, z, t, ∇Wdec, Σ∇Wdec, C, Σ∇bdec,Wdec,∇ho,∇mo,N,d, hoNmoN,∇hoNmoN)
      GRID.∇hyperlstm!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo, unrollsteps)
      GRID.∇encode!(x, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi,∇hiNmiN)

      gradientstep+=1
      OPTIMIZER.optimize_Wencdec!(N, Wenc, Σ∇Wenc, mWenc, vWenc, gradientstep)
      OPTIMIZER.optimize_bencdec!(N, benc, Σ∇benc, mbenc, vbenc, gradientstep)
      OPTIMIZER.optimize_W!(N, W, Σ∇W, Wm, Wv, gradientstep)
      #OPTIMIZER.maxnormconstrain_W!(W, N, 4)
      OPTIMIZER.optimize_b!(N, b, Σ∇b, bm, bv, gradientstep)
      OPTIMIZER.optimize_Wencdec!(N, Wdec, Σ∇Wdec, mWdec, vWdec, gradientstep)
      OPTIMIZER.optimize_bencdec!(N, bdec, Σ∇bdec, mbdec, vbdec, gradientstep)

      Ev = 0.999*Ev + 0.001*sum(abs.(∇z[unrollsteps]))/bsz
      println("gradientstep: ", gradientstep, " loss: ", Ev)

      xent = sum(z[unrollsteps].*(1-t[unrollsteps]) .- log.(GRID.σ(z[unrollsteps])))/bsz
      println("xent ", round(xent,3))
    end
    append!(E, Ev)
    CHECKPOINT.save_model(fname1, Wenc, benc, W, b, Wdec, bdec)
    CHECKPOINT.save_optimizevars(fname2, mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, E)
  end
end

function main()
  data = DATALOADER.load_dataset(24*120) #minimum song length (24*60 would mean 60 seconds)
  gridsize=[2,2,2,2,2]
  unrollsteps=24*2
  L=256
  d=256
  bsz=32
  seqlen=24*60
  train(data, gridsize, unrollsteps, L, d, bsz, seqlen)
end

end

TRAIN.main()

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

  fname1 = string("trained/hyperlstm.jld")
  fname2 = string("trained/hyperlstm_opt.jld")

  #while Ev > 1
  while true
    DATALOADER.get_batch!(batch, seqlen, bsz, data)
    GRID.reset_state!(mi,hi,unrollsteps)

    for batchstep=1:unrollsteps:seqlen-unrollsteps
      DATALOADER.get_partialbatch!(x,t,batch,unrollsteps,batchstep,bsz)
      GRID.recur_state!(mi,hi,unrollsteps)

      #fprop
      GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
      GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
      GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)

      #monitor error
      #Ev = GRID.logistic_xent(Ev,t[unrollsteps],z[unrollsteps],bsz)
      #Ev = sum(abs.(GRID.σ(z[unrollsteps]) .- t[unrollsteps]))
      #println("smooth logistic_xent: ", Ev)

      #bprop
      GRID.∇decode!(∇z, z, t, ∇Wdec, Σ∇Wdec, C, Σ∇bdec,Wdec,∇ho,∇mo,N,d, hoNmoN,∇hoNmoN)
      GRID.∇hyperlstm!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo, unrollsteps)
      GRID.∇encode!(x, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi,∇hiNmiN)

      Ev = 0.999*Ev + 0.001*sum(abs.(∇z[unrollsteps]))

      #adjust
      gradientstep+=1
      OPTIMIZER.optimize_Wencdec!(N, Wenc, Σ∇Wenc, mWenc, vWenc, gradientstep)
      OPTIMIZER.optimize_bencdec!(N, benc, Σ∇benc, mbenc, vbenc, gradientstep)
      OPTIMIZER.optimize_W!(N, W, Σ∇W, Wm, Wv, gradientstep, 0.002)
      OPTIMIZER.maxnormconstrain_W!(W, N, 4)
      OPTIMIZER.optimize_b!(N, b, Σ∇b, bm, bv, gradientstep)
      OPTIMIZER.optimize_Wencdec!(N, Wdec, Σ∇Wdec, mWdec, vWdec, gradientstep)
      OPTIMIZER.optimize_bencdec!(N, bdec, Σ∇bdec, mbdec, vbdec, gradientstep)
    end
    println("gradientstep: ", gradientstep, " loss: ", Ev)
    append!(E, Ev)
    CHECKPOINT.save_model(fname1, Wenc, benc, W, b, Wdec, bdec)
    CHECKPOINT.save_optimizevars(fname2, mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothcost)
  end
end

function main()
  #data = DATALOADER.load_dataset(24*120) #minimum song length (24*60 would mean 60 seconds)
  data = DATALOADER.BeethovenLudwigvan()
  lengths = [data[n][end,1] for n=1:length(data)]
  println(lengths)

  gridsize=[2,2,2]
  unrollsteps=10
  L=256
  d=256
  bsz=32
  #seqlen=24*60
  seqlen=24*120

  train(data, gridsize, unrollsteps, L, d, bsz, seqlen)
end

end

@time TRAIN.main()
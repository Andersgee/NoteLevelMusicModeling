module TRAIN

include("grid.jl")
include("optimizer.jl")
include("dataloader.jl")
include("checkpoint.jl")


function train(data,L,d,bsz,gridsize)
  # pre-allocate
  N, C, fn, bn = GRID.linearindexing(gridsize)
  Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W, b, g, mi, mo, hi, ho, Hi, WHib = GRID.gridvars(N,C,d,bsz)
  ∇Wenc, Σ∇Wenc, Σ∇benc, ∇Wdec, Σ∇Wdec, Σ∇bdec = GRID.∇encodevars(L,d,N,bsz)
  ∇W, Σ∇W, Σ∇b, ∇hi, ∇ho, ∇Hi, Σ∇Hi, ∇WHib, ∇mi, ∇mo = GRID.∇gridvars(N,C,d,bsz)
  Wm, Wv, bm, bv, mWenc, vWenc, mbenc, vbenc, mWdec, vWdec, mbdec, vbdec = GRID.optimizevars(d,N,L)

  # setup a sequence
  seqdim = 1
  projdim = 2
  seqlen=30*gridsize[seqdim] # seqlen=6*gridsize[seqdim] would mean optimize 6 times within a batch
  x, z, t, ∇z, batch, himi,homo,∇himi,∇homo = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen,d)

  gradientstep=0
  smoothAccuracy=0.5
  smoothcost = zeros(0)

  fname1 = string("trained/accuracy_basic",gridsize[2],"depth_12key_bsz",bsz,"_seqlen",seqlen,".jld")
  fname2 = string("trained/accuracy_basic",gridsize[2],"depth_12key_bsz",bsz,"_seqlen",seqlen,"_opt.jld")

  # uncomment to replace appropriate vars and continue interuppted training
  #Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)
  #mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothcost = CHECKPOINT.load_optimizevars(fname2)
  #smoothAccuracy=smoothcost[end]

  println("starting training.")
  #while smoothAccuracy > 0.01
  while true
    DATALOADER.get_batch!(batch, seqlen, bsz, data)
    GRID.reset_sequence!(gridsize, seqdim, projdim, mo, ho, fn)

    for batchstep=1:gridsize[seqdim]:seqlen-gridsize[seqdim]+1
      DATALOADER.get_partialbatch!(x,t,batch,gridsize,seqdim,batchstep,bsz)
      GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)

      # fprop
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
      GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)

      # display info
      Accuracy = sum((z[end].>0).==t[end])/(L*bsz)
      smoothAccuracy = 0.999*smoothAccuracy + 0.001*Accuracy
      println("gradientstep: ", gradientstep, " smoothaccuracy: ",round(smoothAccuracy,4))

      # bprop
      GRID.∇cost!(∇z, z, t)
      GRID.∇decode!(∇z, seqdim, projdim, ∇Wdec, Σ∇Wdec, ho,mo,C,bn,Σ∇bdec,Wdec,∇ho,∇mo,d, homo,∇homo)
      GRID.∇grid!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo)
      GRID.∇encode!(x, seqdim, projdim, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi, fn, d, ∇himi)

      # adjust encoders, grid and decoders
      gradientstep+=1
      OPTIMIZER.optimize_Wencdec!(N, Wenc, Σ∇Wenc, mWenc, vWenc, gradientstep, 0.005)
      OPTIMIZER.optimize_bencdec!(N, benc, Σ∇benc, mbenc, vbenc, gradientstep, 0.005)
      OPTIMIZER.optimize_W!(N, W, Σ∇W, Wm, Wv, gradientstep, 0.005)
      OPTIMIZER.optimize_b!(N, b, Σ∇b, bm, bv, gradientstep, 0.005)
      OPTIMIZER.optimize_Wencdec!(N, Wdec, Σ∇Wdec, mWdec, vWdec, gradientstep, 0.005)
      OPTIMIZER.optimize_bencdec!(N, bdec, Σ∇bdec, mbdec, vbdec, gradientstep, 0.005)
    end

    append!(smoothcost, smoothAccuracy)
    CHECKPOINT.save_model(fname1, Wenc, benc, W, b, Wdec, bdec)
    CHECKPOINT.save_optimizevars(fname2, mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothcost)

  end
end

function main()
  #data = DATALOADER.load_dataset(24*2*60) # specify minimum song length (24*100 would mean 100 seconds)
  
  #data = DATALOADER.BachJohannSebastian()
  #data = DATALOADER.BeethovenLudwigvan()
  data = DATALOADER.TchaikovskyPeter()

  L = 256 #input/output units
  d = 256 #hidden units
  batchsize=8
  gridsize = [24*4+1,6] # 24*2 would mean backprop 2 seconds (48 timesteps)
  train(data, L, d, batchsize, gridsize)
end

end

TRAIN.main()

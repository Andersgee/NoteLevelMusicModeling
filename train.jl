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
  seqlen=24*60 # 24*60 would mean 60 seconds
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen)

  gradientstep=0
  #smoothcostV=log(L) # if softmax
  smoothcostV=log(2)*L # if logistic (probably)
  smoothcost = zeros(0) #array

  filename1 = string("trained/",length(gridsize),"d_bsz",bsz,"_seqlen",seqlen,".jld")
  filename2 = string("trained/",length(gridsize),"d_bsz",bsz,"_seqlen",seqlen,"_opt.jld")

  # uncomment to replace appropriate vars and continue interuppted training
  #Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(filename1)
  #mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothcost = CHECKPOINT.load_optimizevars(filename2)
  #smoothcostV=smoothcost[end]

  println(smoothcostV)
  println("starting training.")
  while smoothcostV > 0.01
    DATALOADER.get_batch!(batch, seqlen, bsz, data)
    GRID.reset_state!(C, N, mi,hi,mo,ho)

    for batchstep=1:seqlen-gridsize[seqdim]+1
      DATALOADER.get_partialbatch!(x,t,batch,gridsize,seqdim,batchstep,bsz)

      # fprop
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn,d)
      GRID.grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C)

      # display info
      #smoothcostV = GRID.softmax_xent(smoothcostV,t,z,bsz)
      smoothcostV = GRID.logistic_xent(smoothcostV,t,z,bsz)
      println("gradientstep: ", gradientstep, " smoothcost: ",round(smoothcostV,3))

      # bprop
      #GRID.∇softmax_xent!(∇z, z, t)
      GRID.∇logistic_xent!(∇z, z, t)
      GRID.∇decode!(∇z, seqdim, projdim, ∇Wdec, Σ∇Wdec, ho,mo,C,bn,Σ∇bdec,Wdec,∇ho,∇mo,d)
      GRID.∇grid!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo)
      GRID.∇encode!(x, seqdim, projdim, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi, fn)

      GRID.recur_state!(C, mi,hi,mo,ho)

      # adjust encoders, grid and decoders
      gradientstep+=1
      OPTIMIZER.optimize_Wencdec!(N, Wenc, Σ∇Wenc, mWenc, vWenc, gradientstep)
      OPTIMIZER.optimize_bencdec!(N, benc, Σ∇benc, mbenc, vbenc, gradientstep)
      OPTIMIZER.optimize_W!(N, W, Σ∇W, Wm, Wv, gradientstep)
      OPTIMIZER.optimize_b!(N, b, Σ∇b, bm, bv, gradientstep)
      OPTIMIZER.optimize_Wencdec!(N, Wdec, Σ∇Wdec, mWdec, vWdec, gradientstep)
      OPTIMIZER.optimize_bencdec!(N, bdec, Σ∇bdec, mbdec, vbdec, gradientstep)
    end

    append!(smoothcost, smoothcostV)
    CHECKPOINT.save_model(filename1, Wenc, benc, W, b, Wdec, bdec)
    CHECKPOINT.save_optimizevars(filename2, mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothcost)

  end
end

function main()
  data = DATALOADER.load_dataset(24*2*60) # specify minimum song length (24*100 would mean 100 seconds)
  
  #data = DATALOADER.BachJohannSebastian()
  #data = DATALOADER.BeethovenLudwigvan()
  #lengths = [data[n][end,1] for n=1:length(data)]
  #println(lengths)

  L = 256 #input/output units
  #d = 256 #hidden units
  d = 256*2 #hidden units
  batchsize=64
  gridsize = [24*2,6] #backprop 2 seconds
  train(data, L, d, batchsize, gridsize)
end

end

TRAIN.main()

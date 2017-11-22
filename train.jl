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
  smooth=0.5
  smooth2=0.5
  smooth3=0.0
  smoothvec = zeros(0)
  smoothvec2 = zeros(0)
  smoothvec3 = zeros(0)

  fname1 = string("trained/informed_",gridsize[1],"-",gridsize[2],"_bsz",bsz,"_seqlen",seqlen,".jld")
  fname2 = string("trained/informed_",gridsize[1],"-",gridsize[2],"_bsz",bsz,"_seqlen",seqlen,"_opt.jld")

  # uncomment to replace appropriate vars and continue interuppted training
  #Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)
  #mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothvec,smoothvec2,smoothvec3 = CHECKPOINT.load_optimizevars(fname2)
  #smooth=smoothvec[end]
  #smooth2=smoothvec2[end]
  #smooth3=smoothvec3[end]

  save_every_x_batch = 0
  println("starting training.")
  #while smooth > 0.01
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
      truepositives = sum([sum(((z[i].>0) .== 1) .* (t[i] .== 1)) for i=1:gridsize[seqdim]])
      falsenegatives= sum([sum(((z[i].>0) .== 0) .* (t[i] .== 1)) for i=1:gridsize[seqdim]])
      truenegatives = sum([sum(((z[i].>0) .== 0) .* (t[i] .== 0)) for i=1:gridsize[seqdim]])
      falsepositives= sum([sum(((z[i].>0) .== 1) .* (t[i] .== 0)) for i=1:gridsize[seqdim]])
      
      sensitivity = truepositives/(truepositives+falsenegatives)
      specificity = truenegatives/(truenegatives+falsepositives)
      informedness = sensitivity+specificity-1
      #println("step: ",gradientstep,"sensitivity: ", round(sensitivity,4), " specificity: ", round(specificity,4), " informedness: ", round(informedness,4))
      smooth = 0.999*smooth + 0.001*sensitivity
      smooth2= 0.999*smooth2+ 0.001*specificity
      smooth3= 0.999*smooth3+ 0.001*informedness
      println("step: ",gradientstep," sensitivity: ", round(smooth,4), " specificity: ", round(smooth2,4), " informedness: ", round(smooth3,4))

      # bprop
      GRID.∇cost!(∇z, z, t)
      GRID.∇decode!(∇z, seqdim, projdim, ∇Wdec, Σ∇Wdec, ho,mo,C,bn,Σ∇bdec,Wdec,∇ho,∇mo,d, homo,∇homo)
      GRID.∇grid!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo)
      GRID.∇encode!(x, seqdim, projdim, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi, fn, d, ∇himi)

      # adjust encoders, grid and decoders
      gradientstep+=1
      OPTIMIZER.optimize_Wencdec!(N, Wenc, Σ∇Wenc, mWenc, vWenc, gradientstep)
      OPTIMIZER.optimize_bencdec!(N, benc, Σ∇benc, mbenc, vbenc, gradientstep)
      OPTIMIZER.optimize_W!(N, W, Σ∇W, Wm, Wv, gradientstep)
      OPTIMIZER.optimize_b!(N, b, Σ∇b, bm, bv, gradientstep)
      OPTIMIZER.optimize_Wencdec!(N, Wdec, Σ∇Wdec, mWdec, vWdec, gradientstep)
      OPTIMIZER.optimize_bencdec!(N, bdec, Σ∇bdec, mbdec, vbdec, gradientstep)
    end

    append!(smoothvec, smooth)
    append!(smoothvec2,smooth2)
    append!(smoothvec3,smooth3)

    save_every_x_batch += 1
    if save_every_x_batch%10 == 0
      CHECKPOINT.save_model(fname1, Wenc, benc, W, b, Wdec, bdec)
      CHECKPOINT.save_optimizevars(fname2, mWenc,vWenc, mbenc,vbenc, Wm,Wv, bm,bv, mWdec,vWdec, mbdec,vbdec, gradientstep, smoothvec, smoothvec2, smoothvec3)
    end

  end
end

function main()
  #data = DATALOADER.load_dataset(24*2*60) # specify minimum song length (24*100 would mean 100 seconds)
  
  #data = DATALOADER.BachJohannSebastian()

  data = DATALOADER.BeethovenLudwigvan()
  println("SONGLENGTH: ",data[3][end,1])
  #data = DATALOADER.TchaikovskyPeter()
  #println("SONGLENGTH: ",data[2][end,1])
  

  L = 128 #input/output units
  d = 64 #hidden units
  batchsize=8
  gridsize = [12*4,1]
  train(data, L, d, batchsize, gridsize)
end

end

TRAIN.main()

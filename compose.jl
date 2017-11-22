module COMPOSE

import NPZ
import Distributions
include("grid.jl")
include("dataloader.jl")
include("checkpoint.jl")

function compose(data, L,d,bsz,gridsize)
  # pre-allocate
  N, C, fn, bn = GRID.linearindexing(gridsize)
  Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W, b, g, mi, mo, hi, ho, Hi, WHib = GRID.gridvars(N,C,d,bsz)

  # setup a sequence
  seqdim = 1
  projdim = 2
  seqlen=12*4*15
  x, z, t, ∇z, batch, himi,homo,∇himi,∇homo = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen,d)
  y=t

  # load trained model
  fname1 = "trained/informed_48-1_bsz8_seqlen1440.jld"
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)

  a=[1,0,1,0,1,1,0,1,0,1,0,1]
  a=[a;a;a;a;a;a;a;a;a;a;a;a]
  key=rand(1:12)
  #key=12
  randkey = circshift(a[1:128],key)

  println("Loading prime data.")
  DATALOADER.get_batch!(batch, seqlen, bsz, data)
  for i=1:bsz
    sumNoteOnEvents = sum(batch[i][1:128,:] .> 0)
    println(i, " has on:",sumNoteOnEvents)
    #filename=string("data/generated_npy/pgenerated",i,".npy")
    #NPZ.npzwrite(filename, batch[i][:,end-24*10:end]) #10 last seconds of priming
  end

  println("Starting priming.")
  for batchstep=1:seqlen-1
    DATALOADER.get_partialbatch!(x,t,batch,gridsize,seqdim,batchstep,bsz)
    GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
    GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
    GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
    GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)

    #truepositives = sum([sum(((z[i].>0) .== 1) .* (t[i] .== 1)) for i=1:gridsize[seqdim]])
    #falsenegatives= sum([sum(((z[i].>0) .== 0) .* (t[i] .== 1)) for i=1:gridsize[seqdim]])
    #truenegatives = sum([sum(((z[i].>0) .== 0) .* (t[i] .== 0)) for i=1:gridsize[seqdim]])
    #falsepositives= sum([sum(((z[i].>0) .== 1) .* (t[i] .== 0)) for i=1:gridsize[seqdim]])
    #sensitivity = truepositives/(truepositives+falsenegatives)
    #specificity = truenegatives/(truenegatives+falsepositives)
    #informedness = sensitivity+specificity-1
    #println("sensitivity:", sensitivity, " specificity:",specificity)
  end

  println("Starting generating.")
  T=0.0
  for I=1:1
    K=10
    for i=1:bsz
      fill!(batch[i],0.0)
    end
    for batchstep=1:seqlen
      fill!(x[1], 0.0)

      
      y = GRID.σ(z[1])

      p = softmax(z[1], 1.5)
      #p = softmax(z[1], 1.0)

      #yt = (y.>0.5).*y.*randkey
      yt = (y.>0.5).*y
      for i=1:bsz
        if sum(y[:,i]) > 0
          notes = Distributions.wsample(1:L, p[:,i], K)
          x[1][notes,i] .= yt[notes,i].>0
          batch[i][notes,batchstep] .= yt[notes,i]
        end
      end

      #fprop
      GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
      GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)
    end

    println("Iteration ",I)
    for i=1:bsz
      sumNoteOnEvents = sum(batch[i] .> 0)
      uniquenotes=sum(sum(batch[i],2).>0)
      
      filename=string("data/generated_npy/generated",i,".npy")
      NPZ.npzwrite(filename, batch[i])
      println("saved song ",i, " (has on:",sumNoteOnEvents, " unique:",uniquenotes,")")
    end

  end
end
function softmax(z,temp)
    zm = maximum(z,1)
    y = exp.((z.-zm)./temp)
    p = y ./ sum(y,1)
    return p
end

function main()
  data = DATALOADER.BeethovenLudwigvan()
  #data = DATALOADER.TchaikovskyPeter()
  L = 128
  d = 128
  batchsize=16
  gridsize = [1,1]
  compose(data, L, d, batchsize, gridsize)
end

end

COMPOSE.main()

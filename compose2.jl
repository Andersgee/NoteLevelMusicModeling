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
  seqlen=24*4*30
  x, z, t, ∇z, batch, himi,homo,∇himi,∇homo = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen,d)
  y=t
  # load trained model
  #fname1 = "trained/loss01_basic1depth_12key_bsz32_seqlen2910.jld"
  fname1 = "trained/loss006_basic1depth_12key_bsz32_seqlen2910.jld"
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)

  println("Loading prime data.")
  DATALOADER.get_batch!(batch, seqlen, bsz, data)
  for i=1:bsz
    sumNoteOnEvents = sum(batch[i][1:128,:] .> 0)
    sumNoteOffEvents = sum(batch[i][129:256,:] .> 0)
    println(i, " has on:",sumNoteOnEvents, " off:", sumNoteOffEvents)
    #filename=string("data/generated_npy/pgenerated",i,".npy")
    #NPZ.npzwrite(filename, batch[i][:,end-24*10:end]) #10 last seconds of priming
  end

  println("Starting priming.")
  for batchstep=1:seqlen-1

    #get input
    DATALOADER.get_partialbatch!(x,t,batch,gridsize,seqdim,batchstep,bsz)

    #fprop
    GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
    GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
    GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
    GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)
    GRID.σgate!(y,z,1) #y[1] = GRID.σ(z[1]) 
  end

  #T = 0.14
  T = 0.1
  println("Starting generating.")
  for I=1:1000
    if I>8
      K=8
    else
      K=I
    end

    for i=1:bsz
      fill!(batch[i],0.0)
    end
    for batchstep=1:seqlen

      #get input
      fill!(x[1], 0.0)
      for i=1:bsz
        #notes=findmax(y[1][:,i])[2]
        #notes = Distributions.wsample(1:L, (y[1][:,i]).^2, K)
        notes = Distributions.wsample(1:L, y[1][:,i], K)
        x[1][notes,i] .= y[1][notes,i] .> T # get x here instead of with get_partialbatch
        batch[i][notes,batchstep] .= (y[1][notes,i] .> T).*y[1][notes,i]
      end

      #fprop
      GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
      GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)
      GRID.σgate!(y,z,1)
    end

    println("Iteration ",I)
    for i=1:bsz
      sumNoteOnEvents = sum(batch[i][1:128,:] .> 0)
      sumNoteOffEvents = sum(batch[i][129:256,:] .> 0)

      uniquenotes=sum(sum(batch[i][1:128,:],2).>0)
      
      filename=string("data/generated_npy/generated",i,".npy")
      NPZ.npzwrite(filename, batch[i])
      println("saved song ",i, " (has on:",sumNoteOnEvents, " off:", sumNoteOffEvents," unique:",uniquenotes,")")
    end

  end
end

function main()
  data = DATALOADER.TchaikovskyPeter()
  L = 256
  d = 256
  batchsize=8
  gridsize = [1,1]
  compose(data, L, d, batchsize, gridsize)
end

end

COMPOSE.main()

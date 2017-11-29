module IMPROVISE

import NPZ
import Distributions
import JLD
include("grid.jl")
include("dataloader.jl")
include("checkpoint.jl")

function lh(d,a, z)
    p1=a[1]*Distributions.pdf(d[1], z)
    p2=a[2]*Distributions.pdf(d[2], z)
    r=p2/(p1+p2)
    return r
end

function compose(data, L,d,bsz,gridsize)
  println("Setting up...")
  # pre-allocate
  N, C, fn, bn = GRID.linearindexing(gridsize)
  Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W, b, g, mi, mo, hi, ho, Hi, WHib = GRID.gridvars(N,C,d,bsz)

  # setup a sequence
  seqdim = 1
  projdim = 2

  #seqlen=12*4*30
  seqlen=12*4*15
  x, z, t, ∇z, batch, himi,homo,∇himi,∇homo = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen,d)

  # load trained model
  fname1 = "trained/informed_48-1_bsz8_seqlen1440.jld"
  #fname1 = "trained/backup/informed_48-1_bsz8_seqlen1440.jld"
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)

  dict=JLD.load("probdensparameters.jld")
  a=dict["a"]
  probdensd=dict["d"]

  println("Priming...")
  DATALOADER.get_batch!(batch, seqlen, bsz, data)
  for batchstep=1:seqlen-1
    DATALOADER.get_partialbatch!(x,t,batch,gridsize,seqdim,batchstep,bsz)
    GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
    GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
    GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
    GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)
  end


  println("Improvising...")
  for I=1:100

  for i=1:bsz
    fill!(batch[i],0.0)
  end
  for batchstep=1:seqlen
    fill!(x[1], 0.0)
    for i=1:bsz
      r = [lh(probdensd[note], a[note], z[1][note,i]) for note=1:128]
      #y = r.>rand(128)
      y = (r.>rand(128)).*(r.>0.075)

      #notes = Distributions.wsample(1:L, r, K)

      x[1][:,i] .= y
      batch[i][:,batchstep] .= y.*r
    end

    #fprop
    GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
    GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
    GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
    GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)
  end

  for i=1:bsz
    sumNoteOnEvents = sum(batch[i] .> 0)
    uniquenotes=sum(sum(batch[i],2).>0)
    
    filename=string("data/generated_npy/generated",i,".npy")
    NPZ.npzwrite(filename, batch[i])
    println("saved song ",i, " (has on:",sumNoteOnEvents, " unique:",uniquenotes,")")
  end

  end
end

function main() 
  L = 128
  d = 1024
  batchsize=8
  gridsize = [1,1]
  data = DATALOADER.load_dataset(12*4*30)
  compose(data, L, d, batchsize, gridsize)
end

end

IMPROVISE.main()

module COMPOSE

import NPZ

include("grid.jl")
include("dataloader.jl")
include("checkpoint.jl")

import Distributions

function compose(data, gridsize, unrollsteps, L, d, bsz, seqlen)
  fname1 = string("trained/_u24_H5lstm.jld")
  N, C, fn, bn = GRID.linearindexing(gridsize)
  W,b,g,mi,mo,hi,ho,Hi,WHib, hiNmiN,hoNmoN = GRID.gridvars(N,C,d,bsz, unrollsteps)
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,unrollsteps,seqlen)
  y = t
  
  println("Starting priming.")
  DATALOADER.get_batch!(batch, seqlen, bsz, data)
  for batchstep=1:unrollsteps:seqlen-unrollsteps
    DATALOADER.get_partialbatch!(x,t,batch,unrollsteps,batchstep,bsz)

    GRID.recur_state!(mi,hi,unrollsteps)
    GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
    GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
    GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)
    y[1] .= GRID.σ(z[1])

    #println("std: x[1] ",round(std(x[1]),3), " \ty[1] ",round(std(y[1]),3))
    #println("mean: x[1] ",round(mean(x[1]),3), " \ty[1] ",round(mean(y[1]),3))

    #println("maximum(z[1] ",maximum(z[1]))
    #println("mean(z[1] ",mean(z[1]))
    #println("std(z[1] ",std(z[1]))
    #println()
    println(mean(mo[1][28][3]))
    #println(mean(ho[1][6][2]))
    #println(mean(mo[1][6][2]))
    #println("mean(y[1] ",mean(y[1]))
    #println("maximum(y[1] ",maximum(y[1]))
  end
  
  #println(bdec)


  println("Starting generating.")
  #m=0.08
  #m=0.1
  T=0.1
  for s=1:seqlen
    x[1] .= y[1].>T

    GRID.recur_state!(mi,hi,unrollsteps)
    GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
    GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
    GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)
    y[1] .= GRID.σ(z[1])

    for i=1:bsz
      notes=Distributions.wsample(1:L, y[1][:,i], 10)
      batch[i][notes,s] .= y[1][notes,i].>T
    end
  end

  println("Saving songs.")
  for i=1:bsz
    filename=string("data/generated_npy/generated",i,".npy")
    NPZ.npzwrite(filename, batch[i])
    #println(filename)
  end

end

function main()
  #data = DATALOADER.load_dataset(24*120) #minimum song length (24*60 would mean 60 seconds)
  data = DATALOADER.BeethovenLudwigvan()
  lengths = [data[n][end,1] for n=1:length(data)]
  println(lengths)

  gridsize=[2,2,2,2,2]
  unrollsteps=1
  L=256
  d=256
  #bsz=32
  bsz=4
  #seqlen=24*60

  #seqlen=24*120
  seqlen=500
  #seqlen=200

  compose(data, gridsize, unrollsteps, L, d, bsz, seqlen)
end

end

COMPOSE.main()

module COMPOSE

import NPZ

include("grid.jl")
include("dataloader.jl")
include("checkpoint.jl")

import Distributions

function compose(data, gridsize, unrollsteps, L, d, bsz, seqlen)
  fname1 = string("trained/H5lstm_u48.jld")
  N, C, fn, bn = GRID.linearindexing(gridsize)
  W,b,g,mi,mo,hi,ho,Hi,WHib, hiNmiN,hoNmoN = GRID.gridvars(N,C,d,bsz, unrollsteps)
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,unrollsteps,seqlen)
  y = t

  println("Starting priming.")
  DATALOADER.get_batch!(batch, seqlen, bsz, data)
  for batchstep=1:seqlen-1
    #println("Priming ",round(100*batchstep/(seqlen-unrollsteps),0),"%",)

    DATALOADER.get_partialbatch!(x,t,batch,unrollsteps,batchstep,bsz)
    GRID.recur_state!(mi,hi,unrollsteps)
    GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
    GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
    GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)
    y[1] .= GRID.σ(z[1])
  end

  

  println("Starting generating.")

  Temperature = 0.9
  Threshold = 0.022

  for I=1:10
    println("Iteration ",I)
    #K=I+2
    K=I+1
    #reset batch before writing to it
    for i=1:bsz; fill!(batch[i],0.0) end

    for batchstep=1:seqlen
      #println("Generating ",round(100*batchstep/(seqlen),0),"%",)
      
      fill!(x[1], 0.0)
      for i=1:bsz
        notes=Distributions.wsample(1:L, (y[1][:,i]).^(1/Temperature), K)
        x[1][notes,i] .= y[1][notes,i].>Threshold
        batch[i][notes,batchstep] .= (y[1][notes,i].>Threshold).*y[1][notes,i]
      end

      GRID.recur_state!(mi,hi,unrollsteps)
      GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
      GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
      GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)
      y[1] .= GRID.σ(z[1])
    end

    for i=1:bsz
      sumNoteOnEvents = sum(batch[i][1:128,:] .> 0)
      if sumNoteOnEvents>0
        filename=string("data/generated_npy/generated",i,".npy")
        NPZ.npzwrite(filename, batch[i])
        println("saved ",filename, " (has ",sumNoteOnEvents, " Note On Events)")
      end
    end
  end

end

function main()
  data = DATALOADER.load_dataset(24*120) #minimum song length (24*60 would mean 60 seconds)

  gridsize=[2,2,2,2,2]
  unrollsteps=1
  L=256
  d=256
  bsz=8

  #seqlen=500
  seqlen = 1000

  compose(data, gridsize, unrollsteps, L, d, bsz, seqlen)
end

end

COMPOSE.main()

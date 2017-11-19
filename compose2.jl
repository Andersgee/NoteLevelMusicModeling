module COMPOSE

import NPZ

include("grid.jl")
include("dataloader.jl")
include("checkpoint.jl")

import Distributions

function coord_cos_sim(x)
  #cosine similarity to coordinate axes
  return x./norm(x)
  #similarity = cossim(x)
  #acos.(similarity) #θ DIFFERENCE in radians
  #acos.(similarity)*180/π #θ DIFFERENCE in degrees
end

function compose(data, gridsize, unrollsteps, L, d, bsz, seqlen)
  fname1 = string("trained/H2lstm_u48.jld")
  N, C, fn, bn = GRID.linearindexing(gridsize)
  W,b,g,mi,mo,hi,ho,Hi,WHib, hiNmiN,hoNmoN = GRID.gridvars(N,C,d,bsz, unrollsteps)
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,unrollsteps,seqlen)
  y = t

  TEST = zeros(0)

  println("Starting priming.")
  DATALOADER.get_batch!(batch, seqlen, bsz, data)

  for i=1:bsz
    sumNoteOnEvents = sum(batch[i][1:128,:] .> 0)
    if sumNoteOnEvents>0
      filename=string("data/generated_npy/getbatch",i,".npy")
      NPZ.npzwrite(filename, batch[i])
      println("saved ",filename, " (has ",sumNoteOnEvents, " Note On Events)")
    end
  end

  for batchstep=1:seqlen-1
    #println("Priming ",round(100*batchstep/(seqlen-unrollsteps),0),"%",)

    DATALOADER.get_partialbatch!(x,t,batch,unrollsteps,batchstep,bsz)
    GRID.recur_state!(mi,hi,unrollsteps)
    GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
    GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
    GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)
    y[1] .= GRID.σ(z[1])

    println(std(vcat(mo[unrollsteps][C][1],mo[unrollsteps][C][2]), 1))
    println(std(vcat(ho[unrollsteps][C][1],ho[unrollsteps][C][2]), 1))

    similarity = coord_cos_sim(y[1][:,1])
    #println(maximum(similarity))
    append!(TEST, maximum(similarity))

  end

  

  Eend=4.170124834956534
  #ythresh=GRID.σ(-Eend)
  ythresh=0.2
  println("ythresh: ", ythresh)

  println("Starting generating.")

  #simThreshold = mean(TEST)*0.95
  simThreshold = 0.19
  #simThreshold = 0.08
  println("simThreshold: ", simThreshold)

  for I=1:20
    K=I

    for i=1:bsz
      fill!(batch[i],0.0)
    end
    for batchstep=1:seqlen
      fill!(x[1], 0.0)
      for i=1:bsz
        similarity = coord_cos_sim(y[1][:,i])
        #println(maximum(similarity))

        #notes=find(similarity .> simThreshold)
        #notes=find(y[1][:,i] .> ythresh)

        #if length(notes)>0
        #  wsampled = Distributions.wsample(1:length(notes), y[1][notes,i], K)
        #  notes = notes[wsampled]
        #end

        notes = Distributions.wsample(1:L, y[1][:,i], K)

        x[1][notes,i] .= y[1][notes,i] .> 0
        batch[i][notes,batchstep] .= y[1][notes,i]
      end

      GRID.recur_state!(mi,hi,unrollsteps)
      GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
      GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
      GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)
      y[1] .= GRID.σ(z[1])
    end

    println("Iteration ",I)
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
  #data = DATALOADER.load_dataset(24*120) #minimum song length (24*60 would mean 60 seconds)
  data = DATALOADER.TchaikovskyPeter()

  gridsize=[2,2]
  unrollsteps=1
  L=256
  d=256
  bsz=4

  #seqlen=240
  seqlen = 24*30

  compose(data, gridsize, unrollsteps, L, d, bsz, seqlen)
end

end

COMPOSE.main()

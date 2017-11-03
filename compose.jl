module COMPOSE

import NPZ
import Distributions
include("grid.jl")
include("checkpoint.jl")

function randTriad()
  # Picks a Major or minor triad in either rootposition, first or second inversion.
  lowest=48
  pattern=[0 3 7; 0 4 7; 0 4 9; 0 3 8; 0 5 8; 0 5 9]
  notes=collect(lowest:127)
  return notes[rand(1:12) + pattern[rand(1:size(pattern,1)),:]]
end

function compose(L,d,bsz,gridsize)
  # pre-allocate
  N, C, fn, bn = GRID.linearindexing(gridsize)
  Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W, b, g, mi, mo, hi, ho, Hi, WHib = GRID.gridvars(N,C,d,bsz)

  # setup a sequence
  seqdim = 1
  projdim = 2
  seqlen=1000
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen)

  #filename1 = string("trained/trained_bsz64_seqlen500.jld")
  #filename1 = string("trained/trained_bsz4_seqlen4000.jld")
  filename1 = "trained/trained_bsz32_seqlen1440.jld"
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(filename1)

  println("Composing ",bsz," songs in parallell")

  for i=1:bsz
    batch[i][randTriad(),1]=0.5
  end

  T = 0.1
  for iteration=1:10

    for batchstep=1:seqlen
      for i=1:bsz
        x[1][:,i] .= batch[i][:,batchstep].>0
      end

      GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn,d)
      GRID.grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C)
      output = GRID.σ(z[1])
      output=(output.>T).*output

      K=min(iteration,10)
      for i=1:bsz
        batch[i][:,batchstep+1] .*= 0.0
        #notes=Distributions.wsample(1:L, output[:,i], i) #sample "i" notes mean 2 for song 2.. and 16 for song 16 etc
        #notes=Distributions.wsample(1:L, output[:,i], iteration)
        notes=Distributions.wsample(1:L, output[:,i], K)
        batch[i][notes,batchstep+1] .= output[notes,i]
      end

      #println(batchstep)

    end

    #put last output as first again
    #for i=1:bsz
    #  batch[i][:,1] .= batch[i][:,end]
    #end

    println("\nAfter iteration ",iteration,":")

    #some info
    for i=1:bsz
      events=sum(batch[i][1:128,:].>0)
      #events=sum(batch[i].>0)
      #events=sum(batch[i][1:128,end-500:end].>0)
      if events!=0
        println("song ",i," has ",events, " events")
      end
    end
  end

  println("\nsaving songs:")
  for i=1:bsz
    filename=string("data/generated_npy/generated",i,".npy")
    NPZ.npzwrite(filename, batch[i])
    println(filename)
  end

end

function main()
  L = 256 #input/output units
  d = 256 #hidden units
  #batchsize=16
  batchsize=64
  gridsize = [1,6]
  compose(L, d, batchsize, gridsize)
end

end

COMPOSE.main()

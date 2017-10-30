module COMPOSE

import NPZ
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
  seqlen=500
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen)

  filename1 = string("trained/trained_bsz64_seqlen500.jld")
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(filename1)

  println("Composing ",bsz," songs in parallell")

  for i=1:bsz
    batch[i][randTriad(),1]=1.0
  end

  for batchstep=1:seqlen
    for i=1:bsz
      x[1][:,i] .= batch[i][:,batchstep]
    end
    
    GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
    GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn,d)
    GRID.grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
    GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C)
    output = GRID.σ(z[1])

    for i=1:bsz
      batch[i][:,batchstep+1] .= output[:,i]
    end

    println(batchstep)

  end

  for i=1:bsz
    filename=string("data/generated_npy/generated",i,".npy")
    NPZ.npzwrite(filename, batch[i])
    println("saved ", filename)
  end

end

function main()
  L = 256 #input/output units
  d = 256 #hidden units
  batchsize=8
  #batchsize=8
  gridsize = [1,6]
  compose(L, d, batchsize, gridsize)
end

end

COMPOSE.main()
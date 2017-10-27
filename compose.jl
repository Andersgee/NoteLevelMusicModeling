module Compose
import JLD
import NPZ
include("grid.jl")

function drawsample(y)
  p = y./sum(y)
  table = cumsum(p)
  r=rand()
  for n=1:length(y)
    if table[n]>r
      return n
    end
  end
  return length(y)
end

function randTriad()
  # In programming speak: Returns a random three note chord among 12*6=72 alternatives.
  # In music speak: Picks a Major or minor triad in either rootposition, first or second inversion.
  # Csharp (49) as lowest possible note makes distribution even around the clef (min 3 below, max 2-4 above)
  lowest=48
  pattern=[0 3 7; 0 4 7; 0 4 9; 0 3 8; 0 5 8; 0 5 9]
  notes=collect(lowest:127))
  return notes[rand(1:12) + pattern[rand(1:size(pattern,1)),:]]
end

function main()
  gridsize=[1,6]

  N, C, fn, bn = linind(gridsize)
  d=256
  bsz=1
  W, b, g, mi, mo, hi, ho, Hi, WHib = gridvars(N,C,d,bsz)
  L = 256
  seqdim=1
  seqlen=gridsize[seqdim]
  projdim=2

  #fname="trained/overfitted_waldstein_1_format0/trainedfullset_bsz1seqlen501bproplen50.jld"
  #fname="trained/overfitted_beethoven_hammerklavier_4_format0/trainedfullset_bsz1seqlen501bproplen50.jld"
  fname="trained/overfitted_appass_1_format0/trainedfullset_bsz1seqlen501bproplen50.jld"
  #fname="trained/not_sure/trainedfullset_bsz1seqlen301bproplen50.jld"

  trained = JLD.load(fname)
  Wenc=trained["Wenc"]
  benc=trained["benc"]
  W=trained["W"]
  b=trained["b"]
  Wdec=trained["Wdec"]
  bdec=trained["bdec"]

  x = [zeros(L,1) for i=1:seqlen]
  z = [zeros(L,1) for i=1:seqlen]
  
  K = 1000 # generate K notes ("beats" really, can be multiple notes per beat)
  
  generated=zeros(256,K)
  #prime the net with a random chord to get it started:
  chord=randTriad()
  for n=1:length(chord)
    generated[chord[n],1]=1
  end
  
  T =0.2 # Threshold 0<T<1 (1 means never use any notes, 0 means use all notes always)
  
  for ITERATION=1:75
    println(ITERATION)
    #mi .*= 0
    #hi .*= 0 #THESE NEED TO BE COMMENTED!
    for s=1:K-1
      x[seqlen] .= (generated[:,s].>0).*1

      continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
      encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn,d)
      grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C)
      y = σ(z[seqlen])

      generated[:,s+1].*=0
      for I=1:ITERATION
        i1=drawsample(y)
        generated[i1,s+1] = (y[i1].>T).*y[i1]
      end

    end
  end
  filename="data/generated_npy/generated1.npy"
  NPZ.npzwrite(filename, generated)
  println("saved ", filename)
end

end
Compose.main()

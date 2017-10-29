module COMPOSE

import NPZ
include("grid.jl")
include("checkpoint.jl")

function sample(y)
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
  # Picks a Major or minor triad in either rootposition, first or second inversion.
  lowest=48
  pattern=[0 3 7; 0 4 7; 0 4 9; 0 3 8; 0 5 8; 0 5 9]
  notes=collect(lowest:127)
  return notes[rand(1:12) + pattern[rand(1:size(pattern,1)),:]]
end

function compose(L, d, bsz, gridsize)
  # pre-allocate
  N, C, fn, bn = GRID.linearindexing(gridsize)
  W, b, g, mi, mo, hi, ho, Hi, WHib = GRID.gridvars(N,C,d,bsz)

  # load trained model
  #fname="trained/trained_bsz64_seqlen500_20500steps_3p43loss.jld"
  #fname="trained/trained_bsz64_seqlen500_23500steps_3p18loss.jld"
  fname="trained/trained_bsz8_seqlen500.jld"
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname)

  # setup a sequence
  seqdim=1
  projdim=2
  x = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  z = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  
  # initialize a song
  K = 1000 # Length of song
  generated=zeros(256,K) #initialize song
  generated[randTriad(),1]=1 # prime the song with a random chord
  
  T=0 # Threshold
  for I=1:100
    for s=1:K
      x[1] .= (generated[:,s].>0).*1

      GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)

      # fprop
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn,d)
      GRID.grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C)
      output = GRID.σ(z[1])

      if s<K
        fill!(generated[:,s+1], 0.0)
        for i=1:I
          note=sample(output)
          generated[note,s+1] = (output[note]>T)*output[note]
        end
      else
        fill!(generated[:,1], 0.0)
        for i=1:I
          note=sample(output)
          generated[note,1] = (output[note]>T)*output[note]
        end
      end

    end
    println("After iteration ",I,", song has ",sum(generated[1:128,:].>0)," notes")


    noteOn=generated[1:128,:]
    #mean_noteOnCertainty = mean(noteOn[noteOn.>0])
    #println("average noteOn certainty: ",mean_noteOnCertainty)
    noteOff=generated[129:256,:]
    #mean_noteOffCertainty = mean(noteOff[noteOff.>0])
    #println("average noteOff certainty: ",mean_noteOffCertainty)
    #T=mean_noteOnCertainty*0.95

    median_noteOnCertainty = median(noteOn[noteOn.>0])
    T=median_noteOnCertainty*0.5
    
    println("Threshold is now ", T)
    println()
  end
  #println("average note certainty: ",mean(generated[generated.>0]))
  noteOn=generated[1:128,:]
  mean_noteOnCertainty = mean(noteOn[noteOn.>0])
  println("average noteOn certainty: ",mean_noteOnCertainty)

  noteOff=generated[129:256,:]
  mean_noteOffCertainty = mean(noteOff[noteOff.>0])
  println("average noteOff certainty: ",mean_noteOffCertainty)
#
#  besthalf_noteOn = noteOn.>mean_noteOnCertainty
#  #print(size(besthalf_noteOn))
#
#  println("song has ",sum(generated[1:128,:].>0), " notes")
#  println("deleting least certain 50% of noteOn events")
#  generated = vcat(besthalf_noteOn, noteOff)
#  println("song has ",sum(generated[1:128,:].>0), " notes")
#
  filename="data/generated_npy/generated9.npy"
  NPZ.npzwrite(filename, generated)
  println("saved ", filename)

  
end

function main()
  L = 256
  d = 256
  batchsize=1
  gridsize = [1,6]
  compose(L, d, batchsize, gridsize)
end

end
COMPOSE.main()

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
  #Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W, b, g, mi, mo, hi, ho, Hi, WHib = GRID.gridvars(N,C,d,bsz)

  # setup a sequence
  seqdim = 1
  projdim = 2
  #seqlen=1000
  #seqlen=24*30 #24*60 would mean create 60 second songs
  seqlen=24*120
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen)

  x = [zeros(L,length(batch)) for i=1:1]
  z = [zeros(L,length(batch)) for i=1:1]
  t = [zeros(L,length(batch)) for i=1:1]
  ∇z = [zeros(L,length(batch)) for i=1:1]

  batch = [zeros(L,seqlen+1) for b=1:length(batch)]

  #filename1 = string("trained/trained_bsz64_seqlen500.jld")
  #filename1 = string("trained/trained_bsz4_seqlen4000.jld")
  #filename1 = "trained/trained_bsz32_seqlen1440.jld"
  #filename1="trained/trained_bsz8_seqlen1440.jld"
  filename1="trained/trained_bsz64_seqlen1440.jld"
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(filename1)

  println("Composing ",length(batch)," songs in parallell")

  for i=1:length(batch)
    batch[i][randTriad(),1]=0.5
  end
  
  
  #finalsmoothcost = 0.011410788617904519
  #T = 6 * finalsmoothcost

  #finalsmoothcost = 2.233
  #finalsmoothcost = 1.72
  #T = 0.001/(finalsmoothcost/L) #0.0876

  #T = 0.06 #2.233
  #T = 0.07 #1.72
  #T = 0.06 #1.72
  T = 0.05 #1.72

  for iteration=1:100

    for batchstep=1:seqlen
      for i=1:length(batch)
        x[1][:,i] .= batch[i][:,batchstep].>0
      end

      GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn,d)
      GRID.grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C)
      output = GRID.σ(z[1])
      #output=(output.>T).*output

      #K=min(iteration,10)
      K=iteration+2
      for i=1:length(batch)
        batch[i][:,batchstep+1] .*= 0.0
        #notes=Distributions.wsample(1:L, output[:,i], i) #sample "i" notes mean 2 for song 2.. and 16 for song 16 etc
        #notes=Distributions.wsample(1:L, output[:,i], iteration)
        notes=Distributions.wsample(1:L, output[:,i], K)
        #batch[i][notes,batchstep+1] .= output[notes,i]
        batch[i][notes,batchstep+1] .= (output[notes,i].>T).*output[notes,i]
      end

      #println(batchstep)

    end

    #put last output as first again
    for i=1:length(batch)
      batch[i][:,1] .= batch[i][:,end]
    end

    #events=[sum(batch[i][1:128,:].>0) for i=1:length(batch)]
    #deadsongs = find(events.==0) #indexes of completely dead songs

    uniquenotes=[length(find(sum(batch[i][1:128,:],2))) for i=1:length(batch)]
    deadsongs = find(uniquenotes.<=1)
    nondead = find(uniquenotes.>1)

    #display some info
    println("\nAfter iteration ",iteration,":")
    events=[sum(batch[i][1:128,:].>0) for i=1:length(batch)]
    for i=1:length(batch)
      println("song ",i," has ",events[i], " events (",uniquenotes[i]," unique)")
    end
    println("will delete ",length(deadsongs)," songs")

    if length(deadsongs)!=0 # dont adjust sizes if no songs die
      deleteat!(batch, deadsongs) #delete dead songs
      if length(batch)==0
        println("All songs died.")
        break
      end

      #nondead = find(events) #indexes of nondead songs
      #nondead = find(uniquenotes.>1)

      #decrease size of matrices for actual speedup
      x = [zeros(L,length(batch)) for i=1:1]
      z = [zeros(L,length(batch)) for i=1:1]
      t = [zeros(L,length(batch)) for i=1:1]
      g = [[[zeros(d,length(batch)) for gate=1:4] for n=1:N] for c=1:C]
      mi = [[zeros(d,length(batch)) for n=1:N] for c=1:C+1]
      hi = [[zeros(d,length(batch)) for n=1:N] for c=1:C+1]
      Hi = [zeros(d*N,length(batch)) for c=1:C+1]
      WHib = [zeros(d,length(batch)) for gate=1:4]

      #however dont just reset mo and ho...

      mo_copy = mo
      ho_copy = ho
      mo = [[zeros(d,length(batch)) for n=1:N] for c=1:C+1]
      ho = [[zeros(d,length(batch)) for n=1:N] for c=1:C+1]
      for c=1:C+1
        for n=1:N
          mo[c][n] .= mo_copy[c][n][:,nondead]
          ho[c][n] .= ho_copy[c][n][:,nondead]
        end
      end
    end

    
  end

  if length(batch)==0
    println("Not saving any songs.")
  else
    uniquenotes=[length(find(sum(batch[i][1:128,:],2))) for i=1:length(batch)]
    println("\nAfter Final iteration:")
    events=[sum(batch[i][1:128,:].>0) for i=1:length(batch)]
    for i=1:length(batch)
      println("song ",i," has ",events[i], " events (",uniquenotes[i]," unique)")
    end
    println("\nsaving songs:")
    for i=1:length(batch)
      filename=string("data/generated_npy/generated",i,".npy")
      NPZ.npzwrite(filename, batch[i])
      println(filename)
    end
  end
end

function main()
  L = 256 #input/output units
  d = 256*2 #hidden units
  #batchsize=16
  batchsize=128
  gridsize = [1,6]
  compose(L, d, batchsize, gridsize)
end

end

COMPOSE.main()

module TRAIN
import Distributions
import NPZ

include("grid.jl")
include("dataloader.jl")
include("checkpoint.jl")


function compose(data,L,d,bsz,gridsize)
  #fname1="trained/good_17000/trained_bsz64_seqlen1440.jld"
  fname1="trained/appas_001loss/trained_bsz8_seqlen1440.jld"
  # pre-allocate
  N, C, fn, bn = GRID.linearindexing(gridsize)
  Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W, b, g, mi, mo, hi, ho, Hi, WHib = GRID.gridvars(N,C,d,bsz)
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)


  # setup a sequence
  seqdim = 1
  projdim = 2
  #seqlen=30*48
  seqlen=60*48
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen)
  y=t

  println("starting priming.")
  DATALOADER.get_batch!(batch, seqlen, bsz, data)
  #GRID.reset_sequence!(gridsize, seqdim, projdim, mi, hi, fn)
  for batchstep=1:gridsize[seqdim]:seqlen-gridsize[seqdim]+1
    #println("Priming ",round(100*batchstep/(seqlen-gridsize[seqdim]+1),0),"%",)
    DATALOADER.get_partialbatch!(x,t,batch,gridsize,seqdim,batchstep,bsz)

    GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
    GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn,d)
    GRID.grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
    GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C)
    y[1]=GRID.σ(z[1])
  end

  for i=1:bsz
    fill!(batch[i],0.0)
  end
  
  println("starting generating.")
  T=0.0

  Temperature=0.7
  for I=1:99
    K=I+1
    #K=I
    #K=5

    for batchstep=1:seqlen
      #println(I, " Generating ",round(100*batchstep/(seqlen),0),"%",)
      
      fill!(x[1], 0.0)
      for i=1:bsz
        #notes=Distributions.wsample(1:L, y[1][:,i], K)
        #notes=Distributions.wsample(1:L, (y[1][:,i]).^2, K) # temp=1/2
        notes=Distributions.wsample(1:L, (y[1][:,i]).^(1/Temperature), K)

        x[1][notes,i] .= y[1][notes,i].>T
        batch[i][:,batchstep] .*= 0.0
        batch[i][notes,batchstep] .= (y[1][notes,i].>T).*y[1][notes,i]
      end

      GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn,d)
      GRID.grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C)
      y[1]=GRID.σ(z[1])
    end


    println("Iteration ",I)
    for i=1:bsz
      sumNoteOnEvents = sum(batch[i][1:128,:] .> 0)
      println(i, " has ",sumNoteOnEvents, " Note On Events")
    end  

  end


  println("Saving songs.")
  for i=1:bsz
    filename=string("data/generated_npy/generated",i,".npy")
    NPZ.npzwrite(filename, batch[i])

    sumNoteOnEvents = sum(batch[i][1:128,:] .> 0)
    println(filename, " has ",sumNoteOnEvents, " Note On Events")
  end

end

function main()
  #data = DATALOADER.load_dataset(24*2*60) # specify minimum song length (24*100 would mean 100 seconds)
  data = DATALOADER.BeethovenLudwigvan()

  L = 256 #input/output units
  #d = 256*2 #hidden units
  d=256
  batchsize=8
  gridsize = [1,6] #backprop 2 seconds
  compose(data, L, d, batchsize, gridsize)
end

end

TRAIN.main()

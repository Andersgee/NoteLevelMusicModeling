module COMPOSE

import NPZ
import Distributions
include("grid.jl")
include("dataloader.jl")
include("checkpoint.jl")

function compose(data, L,d,bsz,gridsize)
  # pre-allocate
  N, C, fn, bn = GRID.linearindexing(gridsize)
  Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W, b, g, mi, mo, hi, ho, Hi, WHib = GRID.gridvars(N,C,d,bsz)

  # setup a sequence
  seqdim = 1
  projdim = 2
  seqlen=30*12*4
  x, z, t, ∇z, batch, himi,homo,∇himi,∇homo = GRID.sequencevars(L,bsz,gridsize,seqdim,seqlen,d)
  y=deepcopy(t)
  batch_potential = deepcopy(batch)

  # load trained model
  fname1 = "trained/all_beethoven_informed08_d128_last/informed_48-1_bsz8_seqlen1440.jld"
  Wenc, benc, W, b, Wdec, bdec = CHECKPOINT.load_model(fname1)

  println("Priming.")
  DATALOADER.get_batch!(batch, seqlen, bsz, data)
  for i=1:bsz
    sumNoteOnEvents = sum(batch[i][1:128,:] .> 0)
    println(i, " has on:",sumNoteOnEvents)
    #filename=string("data/generated_npy/pgenerated",i,".npy")
    #NPZ.npzwrite(filename, batch[i][:,end-24*10:end]) #10 last seconds of priming
  end
  for batchstep=1:seqlen-1
    DATALOADER.get_partialbatch!(x,t,batch,gridsize,seqdim,batchstep,bsz)
    GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
    GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
    GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
    GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)
  end

  mo_primed = deepcopy(mo)
  ho_primed = deepcopy(ho)
  z_primed = deepcopy(z)
  batchcopy = deepcopy(batch)
  first_x = deepcopy(x)
  for i=1:bsz
    first_x[1][:,i] = batchcopy[i][:,end]
  end


  for i=1:bsz
    fill!(batch[i],0.0) #clear batch before writing to it
  end

  println("Adding notes.")
  for I=1:1000

    #evaluate entire composition
    mo = deepcopy(mo_primed)
    ho = deepcopy(ho_primed)
    z = deepcopy(z_primed)
    for i=1:bsz
      fill!(batch_potential[i],0.0)
    end
    for batchstep=1:seqlen
      fill!(x[1], 0.0)
      y = GRID.σ(z[1])
      yt = (y.>0.5).*y
      for i=1:bsz

        temp = std(z[1][:,i])/3
        p = softmax(z[1][:,i], temp)
        batch_potential[i][:,batchstep] .= p

        if batchstep==1
          x[1][:,i] .= first_x[1][:,i]
        else
          x[1][:,i] .= (batch[i][:,batchstep-1] .> 0)
        end       
      end

      #fprop
      GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
      GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)
    end

    # now add the most likely note (continue until an empty place is found)
    for i=1:bsz
      noteval=1.0
      while true
        noteval, idx = pickNote(batch_potential[i], noteval)
        if batch[i][idx]==0
          batch[i][idx] = noteval
          println(i, " added ", noteval)
          break
        end
      end
    end

    #evaluate entire composition (again)
    mo = deepcopy(mo_primed)
    ho = deepcopy(ho_primed)
    z = deepcopy(z_primed)
    for i=1:bsz
      fill!(batch_potential[i],0.0)
    end
    for batchstep=1:seqlen
      fill!(x[1], 0.0)
      y = GRID.σ(z[1])
      yt = (y.>0.5).*y
      for i=1:bsz
        temp = std(z[1][:,i])/3
        p = softmax(z[1][:,i], temp)
        batch_potential[i][:,batchstep] .= p

        if batchstep==1
          x[1][:,i] .= first_x[1][:,i]
        else
          x[1][:,i] .= (batch[i][:,batchstep-1] .> 0)
        end
      end

      #fprop
      GRID.continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
      GRID.encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
      GRID.grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
      GRID.decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)
    end

    
    for i=1:bsz
      noteval=1.0
      while true
        noteval, idx = pickNote_remove(batch_potential[i], batch[i], noteval)
        if (batch_potential[i][idx]<0.5) && (batch[i][idx] != 0)
          batch[i][idx]=0
          println(i," removed ", (1-noteval))
          break
        else
          println(i," didnt remove anything")
          break
        end
      end

    end


    println("after Iteration ",I)
    for i=1:bsz
      sumNoteOnEvents = sum(batch[i] .> 0)
      uniquenotes=sum(sum(batch[i],2).>0)
      
      filename=string("data/generated_npy/generated",i,".npy")
      NPZ.npzwrite(filename, batch[i])
      println("saved ",i, " (has on:",sumNoteOnEvents, " unique:",uniquenotes,")")
    end
  end

  for i=1:bsz
    filename=string("data/generated_npy/generated",i,".npy")
    NPZ.npzwrite(filename, batch[i])
    println("saved song ",i)
  end
end

function pickNote_remove(batch_potential, batch, m)
  test=batch_potential.*batch
  vi = findmax((1-test).*((1-test).<m))
  #vi = findmax((1-batch_potential).*((1-batch_potential).<m))
  noteval = vi[1]
  idx = vi[2]
  return noteval, idx
end

function pickNote(batch_potential, m)
  vi = findmax(batch_potential.*(batch_potential .< m))
  noteval = vi[1]
  idx = vi[2]
  return noteval, idx
end

function softmax(z,temp)
    zm = maximum(z,1)
    y = exp.((z.-zm)./temp)
    p = y ./ sum(y,1)
    return p
end

function main()
  #data = DATALOADER.TchaikovskyPeter()
  data = DATALOADER.BeethovenLudwigvan()
  lowerlimit = 30*12*4
  lengths = [data[n][end,1] for n=1:length(data)]
  keep = lengths.>lowerlimit
  data = data[keep]
  lengths = [data[n][end,1] for n=1:length(data)]
  println("(removed ", sum(.!keep), " songs for having length ", lowerlimit," or shorter)")
  println("number of songs: ", sum(keep))
  println("average length: ", mean(lengths))
  println("shortest: ", minimum(lengths))
  println("longest: ", maximum(lengths))

  
  L = 128
  d = 256
  batchsize=8
  gridsize = [1,1]
  compose(data, L, d, batchsize, gridsize)
end

end

COMPOSE.main()

module TRAIN

include("grid.jl")

function main()
  gridsize=[2,2,2]
  unrollsteps=10
  L=50
  d=50
  bsz=1
  seqlen=5

  N, C, fn, bn = GRID.linearindexing(gridsize)
  Wenc, benc, Wdec, bdec = GRID.encodevars(L,d,N,bsz)
  W,b,g,mi,mo,hi,ho,Hi,WHib, hiNmiN,hoNmoN = GRID.gridvars(N,C,d,bsz, unrollsteps)
  ∇Wenc, Σ∇Wenc, Σ∇benc, ∇Wdec, Σ∇Wdec, Σ∇bdec = GRID.∇encodevars(L,d,N,bsz)
  ∇W, Σ∇W, Σ∇b, ∇hi, ∇ho, ∇Hi, Σ∇Hi, ∇WHib,∇mi,∇mo, ∇hiNmiN,∇hoNmoN = GRID.∇gridvars(N,C,d,bsz, unrollsteps)
  Wm, Wv, bm, bv, mWenc, vWenc, mbenc, vbenc, mWdec, vWdec, mbdec, vbdec = GRID.optimizevars(d,N,L)
  x, z, t, ∇z, batch = GRID.sequencevars(L,bsz,unrollsteps,seqlen)


  for s=1:unrollsteps
    x[s].=rand(L,bsz).>0.75
    t[s].=rand(L,bsz).>0.75
  end

  E=log(2)*L

  for K=1:10000

    #fprop
    GRID.encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
    GRID.hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
    GRID.decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)


    E = GRID.logistic_xent(E,t[unrollsteps],z[unrollsteps],bsz)
    #E = sum(log.(1 .+ exp.(z[unrollsteps][C])) .- z[unrollsteps][C].*t[unrollsteps][C])/bsz
    (K%10==0) && println(K, "  LastE: ", E)

    #bprop
    GRID.∇decode!(∇z, z, t, ∇Wdec, Σ∇Wdec, C, Σ∇bdec,Wdec,∇ho,∇mo,N,d, hoNmoN,∇hoNmoN)
    GRID.∇hyperlstm!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo, unrollsteps)
    GRID.∇encode!(x, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi,∇hiNmiN)

    #adjust
    Wdec[1] .-= 0.01*Σ∇Wdec[1]
    Wenc[1] .-= 0.01*Σ∇Wenc[1]
    
    benc[1] .-= 0.001*mean(Σ∇benc[1],2)
    bdec[1] .-= 0.001*mean(Σ∇bdec[1],2)
    for n=1:N, gate=1:4
      b[n][gate] .-= 0.001*mean(Σ∇b[n][gate],2)
      W[n][gate] .-= 0.01*Σ∇W[n][gate]
    end
    constrain!(W, N, 4)
  end
  
  #for n=1:N, gate=1:4
  #  incoming_magnitude=sqrt.(sum(W[n][gate].^2, 2))
  #  println(incoming_magnitude)
  #end

  for s=1:unrollsteps
    println(round.(GRID.σ(z[s]),1))
    println(t[s])
    print("\n")
  end
  print("\n\n")

end

function constrain!(W, N, maxnorm)
  #max-norm constraint on incoming weights to a unit
  for n=1:N, gate=1:4
    mag2=sum(W[n][gate].^2, 2)
    constrain=find(mag2.>maxnorm^2)
    W[n][gate][constrain,:] .*= maxnorm./(sqrt.(mag2[constrain]))
  end
end

end

@time TRAIN.main()
module GRID

σ(x) = 1.0./(1.0.+exp.(-x))

function lstm!(n, WHib, g, mi, mo, ho)
    # 0 allocations
    @. g[n][1] = σ(WHib[1])
    @. g[n][2] = σ(WHib[2])
    @. g[n][3] = σ(WHib[3])
    @. g[n][4] = tanh(WHib[4])
    @. mo[n] = g[n][2]*mi[n] + g[n][1]*g[n][4]
    @. ho[n] = tanh(g[n][3]*mo[n])
end

function ∇lstm!(n, ∇WHib, ∇ho, ho, g, mi, mo, ∇mo)
    # 0 allocations
    @. ∇WHib[1] = ∇ho[n]*(1-ho[n]^2)*g[n][3]*g[n][4]*g[n][1]*(1-g[n][1]) + ∇mo[n]*g[n][4]*g[n][1]*(1-g[n][1])
    @. ∇WHib[2] = ∇ho[n]*(1-ho[n]^2)*g[n][3]*mi[n]*g[n][2]*(1-g[n][2]) + ∇mo[n]*mi[n]*g[n][2]*(1-g[n][2])
    @. ∇WHib[3] = ∇ho[n]*(1-ho[n]^2)*mo[n]*g[n][3]*(1-g[n][3])
    @. ∇WHib[4] = ∇ho[n]*(1-ho[n]^2)*g[n][3]*g[n][1]*(1-g[n][4]^2) + ∇mo[n]*g[n][1]*(1-g[n][4]^2)
end

function grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, mi_future,hi_future)
  for c=1:C
    Hi[c] .= vcat(hi[c]...) # This line responsible for 100% of allocations
    for n=1:N
      for gate=1:4
        A_mul_B!(WHib[gate], W[n][gate], Hi[c])
        WHib[gate] .+= b[n][gate] #WHib[gate] = W[gate]*Hi[c] .+ b[gate]
      end
      lstm!(n, WHib, g[c], mi[c], mo[c], ho[c])

      # send signal
      (fn[c,n] != C+1) && (mi[fn[c,n]][n] .= mo[c][n])
      (fn[c,n] != C+1) && (hi[fn[c,n]][n] .= ho[c][n])

      # also send to future
      (c != 1) && (fn[c,n] != C+1) && (mi_future[c][n] .= mo[c][n])
      (c != 1) && (fn[c,n] != C+1) && (hi_future[c][n] .= ho[c][n])
    end
  end
end

function ∇grid!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo, ∇hi_future,∇mi_future)
  for c=C:-1:1
    fill!(Σ∇Hi, 0.0)
    for n=1:N
      # recieve gradient from future aswell
      (c != C) && (bn[c,n] == C+1) && (∇ho[c][n] .+= ∇hi_future[c][n])
      (c != C) && (bn[c,n] == C+1) && (∇mo[c][n] .+= ∇mi_future[c][n])

      ∇lstm!(n, ∇WHib, ∇ho[c], ho[c], g[c], mi[c], mo[c], ∇mo[c])
      for gate=1:4
        A_mul_Bt!(∇W[n][gate], ∇WHib[gate], Hi[c])
        Σ∇W[n][gate] .+= ∇W[n][gate]
        Σ∇b[n][gate] .+= ∇WHib[gate]
        At_mul_B!(∇Hi[gate], W[n][gate], ∇WHib[gate])
        Σ∇Hi .+= ∇Hi[gate]
      end
    end
    
    # send signal
    for n=1:N 
      ∇hi[c][n] .= Σ∇Hi[1+d*(n-1):d*n,:] # this line responsible for 100% of allocations
      ∇ho[bn[c,n]][n] .= ∇hi[c][n]

      @. ∇mi[c][n] = ∇ho[c][n]*(1.0-ho[c][n]^2)*g[c][n][3]*g[c][n][2] + ∇mo[c][n]*g[c][n][2]
      ∇mo[bn[c,n]][n] .= ∇mi[c][n]
    end
  end
end

function recur!(mi,hi)
  mi[1] .= mi[unrollsteps+1]
  hi[1] .= hi[unrollsteps+1]
end

function hyperlstm!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi, unrollsteps)
  for s=1:unrollsteps
    grid!(C,N,fn,WHib,W,b,g[s],mi[s],mo[s],hi[s],ho[s],Hi, mi[s+1],hi[s+1])
  end
end

function ∇hyperlstm!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo, unrollsteps)
  for n=1:N
    for gate=1:4
      fill!(Σ∇W[n][gate], 0.0)
      fill!(Σ∇b[n][gate], 0.0)
    end
  end
  for s=unrollsteps:-1:1
    ∇grid!(C,N,d, bn,∇WHib,∇hi[s],∇ho[s],ho[s],g[s],mi[s],mo[s],∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi[s],∇mo[s], ∇hi[s+1],∇mi[s+1])
  end
end


function encode!(x, Wenc, benc, mi, hi, N, d,hiNmiN)
  #for s=1:unrollsteps
  for s=1:length(x)
    A_mul_B!(hiNmiN[s], Wenc[1], x[s]) #hiNmiN[s] .= Wenc[1]*x[s] .+ benc
    hiNmiN[s] .+= benc[1]

    #project
    for n=1:N
      hi[s][1][n] .= hiNmiN[s][d*(n-1)+1:d*n,:]
      mi[s][1][n] .= hiNmiN[s][d*(n-1)+1+N*d:d*n+N*d,:]
    end
  end
end

function ∇encode!(x, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi,∇hiNmiN)
  fill!(Σ∇Wenc[1], 0.0)
  fill!(Σ∇benc[1], 0.0)
  for s=1:length(x)
    ∇hiNmiN[s] .= vcat(vcat(∇hi[s][1]...),vcat(∇mi[s][1]...))
    Σ∇benc[1] .+= ∇hiNmiN[s]
    A_mul_Bt!(∇Wenc[1], ∇hiNmiN[s], x[s]) #∇Wenc[1] .= ∇hiNmiN * x[s]'
    Σ∇Wenc[1] .+= ∇Wenc[1]
  end
end

function decode!(C, z, ho, mo, Wdec, bdec, hoNmoN)
  for s=1:length(z)
    hoNmoN[s] .= vcat(vcat(ho[s][C]...),vcat(mo[s][C]...))
    A_mul_B!(z[s], Wdec[1], hoNmoN[s])
    z[s] .+= bdec[1]
  end
end

function ∇decode!(∇z, z, t, ∇Wdec, Σ∇Wdec, C, Σ∇bdec,Wdec,∇ho,∇mo,N,d, hoNmoN,∇hoNmoN)
  fill!(Σ∇Wdec[1], 0.0)
  fill!(Σ∇bdec[1], 0.0)
  for s=1:length(z)
    ∇z[s] .= σ(z[s]) .- t[s] #assuming logistic xent lossfunction
    #∇z[s] .= softmax(z[s]) .- t[s] #assuming softmax xent lossfunction
    #∇z[s] .= z[s] .- t[s] #assuming (half) squared error lossfunction
    #∇z[s] .= y[s] .- t[s] #assuming y has been calculated elsewhere

    Σ∇bdec[1] .+= ∇z[s]
    A_mul_Bt!(∇Wdec[1], ∇z[s], hoNmoN[s]) #∇Wdec[1] = ∇z[s] * hoNmoN'
    Σ∇Wdec[1] .+= ∇Wdec[1]

    At_mul_B!(∇hoNmoN[s], Wdec[1], ∇z[s]) #∇hoNmoN[s] .= Wdec[1]' * ∇z[s]
    #project
    for n=1:N
      ∇ho[s][C][n] .= ∇hoNmoN[s][d*(n-1)+1:d*n,:]
      ∇mo[s][C][n] .= ∇hoNmoN[s][d*(n-1)+1+N*d:d*n+N*d,:]
    end
  end
end

################
# pre-allocate #
################

function gridvars(N,C,d,bsz, unrollsteps)
  W = [[randn(d,d*N)./sqrt(d*N) for gate=1:4] for n=1:N]
  b = [[zeros(d,1) for gate=1:4] for n=1:N]
  #for n=1:N
  #  fill!(b[n][2], 1.0) #positive bias to remember gate will backprob better. (∇mi = ∇mo.*gate2 + ∇ho.*stuff)
  #end

  g = [[[[zeros(d,bsz) for gate=1:4] for n=1:N] for c=1:C] for s=1:unrollsteps+1]
  mi = [[[zeros(d,bsz) for n=1:N] for c=1:C+1] for s=1:unrollsteps+1]
  mo = [[[zeros(d,bsz) for n=1:N] for c=1:C+1] for s=1:unrollsteps+1]
  hi = [[[zeros(d,bsz) for n=1:N] for c=1:C+1] for s=1:unrollsteps+1]
  ho = [[[zeros(d,bsz) for n=1:N] for c=1:C+1] for s=1:unrollsteps+1]
  Hi = [zeros(d*N,bsz) for c=1:C+1]
  WHib = [zeros(d,bsz) for gate=1:4]

  hiNmiN = [zeros(2*N*d,bsz) for s=1:unrollsteps]
  hoNmoN = [zeros(2*N*d,bsz) for s=1:unrollsteps]

  return W,b,g,mi,mo,hi,ho,Hi,WHib, hiNmiN,hoNmoN
end

function ∇gridvars(N,C,d,bsz, unrollsteps)
  ∇W = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  Σ∇W = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  Σ∇b = [[zeros(d,bsz) for gate=1:4] for n=1:N]

  ∇mi = [[[zeros(d,bsz) for n=1:N] for c=1:C+1] for s=1:unrollsteps+1]
  ∇mo = [[[zeros(d,bsz) for n=1:N] for c=1:C+1] for s=1:unrollsteps+1]
  ∇hi = [[[zeros(d,bsz) for n=1:N] for c=1:C+1] for s=1:unrollsteps+1]
  ∇ho = [[[zeros(d,bsz) for n=1:N] for c=1:C+1] for s=1:unrollsteps+1]
  ∇Hi = [zeros(d*N,bsz) for gate=1:4]
  Σ∇Hi = zeros(d*N,bsz)
  ∇WHib = [zeros(d,bsz) for gate=1:4]

  ∇hiNmiN = [zeros(2*N*d,bsz) for s=1:unrollsteps]
  ∇hoNmoN = [zeros(2*N*d,bsz) for s=1:unrollsteps]

  return ∇W, Σ∇W, Σ∇b, ∇hi, ∇ho, ∇Hi, Σ∇Hi, ∇WHib,∇mi,∇mo, ∇hiNmiN,∇hoNmoN
end

function encodevars(L,d,N,bsz)
  Wenc = [randn(2*N*d,L)./sqrt(L) for a=1:1]
  benc = [zeros(2*N*d,1) for a=1:1]

  Wdec = [randn(L,2*N*d)./sqrt(2*N*d) for a=1:1]
  bdec = [zeros(L,1) for a=1:1]

  return Wenc, benc, Wdec, bdec
end

function ∇encodevars(L,d,N,bsz)
  ∇Wenc = [zeros(2*N*d,L) for a=1:1]
  Σ∇Wenc = [zeros(2*N*d,L) for a=1:1]
  Σ∇benc = [zeros(2*N*d,bsz) for a=1:1]

  ∇Wdec = [zeros(L,2*N*d) for a=1:1]
  Σ∇Wdec = [zeros(L,2*N*d) for a=1:1]
  Σ∇bdec = [zeros(L,bsz) for a=1:1]

  return ∇Wenc, Σ∇Wenc, Σ∇benc, ∇Wdec, Σ∇Wdec, Σ∇bdec
end

function optimizevars(d,N,L)
  Wm = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  Wv = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  bm = [[zeros(d,1) for gate=1:4] for n=1:N]
  bv = [[zeros(d,1) for gate=1:4] for n=1:N]

  mWenc = [zeros(2*N*d,L) for a=1:1]
  vWenc = [zeros(2*N*d,L) for a=1:1]
  mbenc = [zeros(2*N*d,1) for a=1:1]
  vbenc = [zeros(2*N*d,1) for a=1:1]

  mWdec = [zeros(L,2*N*d) for a=1:1]
  vWdec = [zeros(L,2*N*d) for a=1:1]
  mbdec = [zeros(L,1) for a=1:1]
  vbdec = [zeros(L,1) for a=1:1]

  return Wm, Wv, bm, bv, mWenc, vWenc, mbenc, vbenc, mWdec, vWdec, mbdec, vbdec
end

#########
# Other #
#########

function linearindexing(G)
  N=length(G)
  C=prod(G)
  fn = Array{Int,2}(C,N)
  bn = Array{Int,2}(C,N)
  garbage = C+1 #garbage unit avoids branching
  for n=1:N
    k = prod(G[1:n-1])
    interval=prod(G[1:n])
    for c=1:C
      fn[c,n] = c+k
      bn[c,n] = c-k
      if c%interval == 0; fn[c-k+1:c,n] = garbage end
      if (c-k)%interval == 0; bn[c-k+1:c,n] = garbage end
    end
  end
  return N, C, fn, bn
end

function sequencevars(L,bsz,unrollsteps,seqlen)
  x = [zeros(L,bsz) for i=1:unrollsteps]
  z = [zeros(L,bsz) for i=1:unrollsteps]
  t = [zeros(L,bsz) for i=1:unrollsteps]
  ∇z = [zeros(L,bsz) for i=1:unrollsteps]
  batch = [zeros(L,seqlen+1) for b=1:bsz]

  return x, z, t, ∇z, batch
end

function reset_state!(C, N, mi,hi,mo,ho)
  for c=1:C
    for n=1:N
      fill!(hi[c][n], 0.0)
      fill!(mi[c][n], 0.0)
    end
  end
end

function continue_sequence!(gridsize, seqdim, projdim, mi, hi, mo, ho, fn)
  c1=1
  c2=gridsize[seqdim] #50
  for i=1:gridsize[projdim] #6
    mi[c1][seqdim] .= mo[c2][seqdim]
    hi[c1][seqdim] .= ho[c2][seqdim]

    c1=fn[c1,projdim]
    c2=fn[c2,projdim]
  end
end

function reset_sequence!(gridsize, seqdim, projdim, mi, hi, fn)
  #instad of mi.*=0 and hi.*=0 , this way I can reset a sequence of choice if multiple exist
  #for 2dimensions, there is only one sequence since the other is temporal
  c1=1
  for i=1:gridsize[projdim]
    fill!(mi[c1][seqdim], 0.0)
    fill!(hi[c1][seqdim], 0.0)

    c1=fn[c1,projdim]
  end
end

function softmax(z)
  # exp(z)/sum(exp(z))
  y = exp.(z[end] .- maximum(z[end],1))
  p = y ./ sum(y,1)
  return p
end

function softmax_xent(smoothcost,t,z,bsz)
  # for multiple, exclusive, classification, softmax activation
  # integrate softmax(z)-t and get: -sum(t*log(z))
  x = z[end] .- maximum(z[end],1)
  logsoftmax = x .- log.(sum(exp.(x),1))
  E = -sum(t[end].*logsoftmax)
  return 0.999*smoothcost + 0.001*E
end

function ∇softmax_xent!(∇z, z, t)
  for i=1:length(z)
    ∇z[i] .= softmax(z[i]) .- t[i]
  end
end

function logistic_xent(smoothcost,t,z,bsz)
  # for multiple, independent, classifications, sigmoid activation
  #integrate σ(z)-t and get: log(1+exp(z))-z*t
  E = sum(log.(1 .+ exp.(z)) .- z.*t)/bsz
  return 0.999*smoothcost + 0.001*E
end

function ∇logistic_xent!(∇z, z, t)
  for i=1:length(z)
    ∇z[i] .= σ(z[i]) .- t[i]
  end
end

end
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

function grid!(C,N,fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
  for c=1:C
    Hi[c] .= vcat(hi[c]...) # This line responsible for 100% of allocations
    for n=1:N
      for gate=1:4
        A_mul_B!(WHib[gate], W[n][gate], Hi[c])
        WHib[gate] .+= b[n][gate] #WHib[gate] = W[gate]*Hi[c] .+ b[gate]
      end
      lstm!(n, WHib, g[c], mi[c], mo[c], ho[c])

      # send signal
      mi[fn[c,n]][n] .= mo[c][n]
      hi[fn[c,n]][n] .= ho[c][n]
    end
  end
end

function ∇grid!(C,N,d, bn,∇WHib,∇hi,∇ho,ho,g,mi,mo,∇W,Σ∇b,Hi,Σ∇W, ∇Hi,Σ∇Hi,W,∇mi,∇mo)
  for n=1:N
    for gate=1:4
      fill!(Σ∇W[n][gate], 0.0)
      fill!(Σ∇b[n][gate], 0.0)
    end
  end

  for c=C:-1:1
    fill!(Σ∇Hi, 0.0)
    for n=1:N 
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

function encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d)
  c=1
  for i=1:length(x)
    himi = Wenc[projdim]*x[i].+benc[projdim]
    hi[c][projdim] .= himi[1:d,:]
    mi[c][projdim] .= himi[1+d:2*d,:]
    c=fn[c,seqdim]
  end
end

function ∇encode!(x, seqdim, projdim, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi, fn)
  fill!(Σ∇Wenc[projdim], 0.0)
  fill!(Σ∇benc[projdim], 0.0)
  c=1
  for i=1:length(x)
    A_mul_Bt!(∇Wenc, vcat(∇hi[c][projdim],∇mi[c][projdim]), x[i])
    Σ∇Wenc[projdim] .+= ∇Wenc
    Σ∇benc[projdim].+=[∇hi[c][projdim];∇mi[c][projdim]]
    c=fn[c,seqdim]
  end
end

function decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C)
  c=C
  for i=length(z):-1:1
    z[i] .= Wdec[projdim]*[ho[c][projdim];mo[c][projdim]] .+ bdec[projdim]
    c=bn[c,seqdim]
  end
end

function ∇decode!(∇z, seqdim, projdim, ∇Wdec, Σ∇Wdec, ho,mo,C,bn,Σ∇bdec,Wdec,∇ho,∇mo,d)
  fill!(Σ∇Wdec[projdim], 0.0)
  fill!(Σ∇bdec[projdim], 0.0)
  c=C
  for i=length(∇z):-1:1
    A_mul_Bt!(∇Wdec, ∇z[i], vcat(ho[c][projdim],mo[c][projdim]))
    Σ∇Wdec[projdim].+=∇Wdec
    Σ∇bdec[projdim].+=∇z[i]
    ∇homo = Wdec[projdim]'*∇z[i]
    ∇ho[c][projdim] .= ∇homo[1:d,:]
    ∇mo[c][projdim] .= ∇homo[1+d:d*2,:]
    c=bn[c,seqdim]
  end
end

################
# pre-allocate #
################

function gridvars(N,C,d,bsz)
  W = [[randn(d,d*N)./sqrt(d*N) for gate=1:4] for n=1:N]
  b = [[zeros(d,1) for gate=1:4] for n=1:N]
  for n=1:N
    fill!(b[n][2], 1.0) #positive bias to gate2=σ(W*x+b) will backprob better since: ∇mi = ∇mo.*gate2
    # (usually called forget gate but a big value means forget less... better name is "remember" gate)
    # bias 1 to gate2 means initially on average σ(1)=73% instead of σ(0)=50% is remembered
  end

  g = [[[zeros(d,bsz) for gate=1:4] for n=1:N] for c=1:C]
  mi = [[zeros(d,bsz) for n=1:N] for c=1:C+1]
  mo = [[zeros(d,bsz) for n=1:N] for c=1:C+1]
  hi = [[zeros(d,bsz) for n=1:N] for c=1:C+1]
  ho = [[zeros(d,bsz) for n=1:N] for c=1:C+1]
  Hi = [zeros(d*N,bsz) for c=1:C+1]
  WHib = [zeros(d,bsz) for gate=1:4]
  return W,b,g,mi,mo,hi,ho,Hi,WHib
end

function ∇gridvars(N,C,d,bsz)
  ∇W = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  Σ∇W = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  Σ∇b = [[zeros(d,bsz) for gate=1:4] for n=1:N]
  ∇mi = [[zeros(d,bsz) for n=1:N] for c=1:C+1]
  ∇mo = [[zeros(d,bsz) for n=1:N] for c=1:C+1]
  ∇hi = [[zeros(d,bsz) for n=1:N] for c=1:C+1]
  ∇ho = [[zeros(d,bsz) for n=1:N] for c=1:C+1]
  ∇Hi = [zeros(d*N,bsz) for gate=1:4]
  Σ∇Hi = zeros(d*N,bsz)
  ∇WHib = [zeros(d,bsz) for gate=1:4]
  return ∇W, Σ∇W, Σ∇b, ∇hi, ∇ho, ∇Hi, Σ∇Hi, ∇WHib,∇mi,∇mo
end

function encodevars(L,d,N,bsz)
  Wenc = [randn(d*2,L)./sqrt(10) for n=1:N]
  benc = [zeros(d*2,1) for n=1:N]

  Wdec = [randn(L,d*2)./sqrt(d*2) for n=1:N]
  bdec = [zeros(L,1) for n=1:N]

  return Wenc, benc, Wdec, bdec
end

function ∇encodevars(L,d,N,bsz)
  ∇Wenc = zeros(d*2,L)
  Σ∇Wenc = [zeros(d*2,L) for n=1:N]
  Σ∇benc = [zeros(d*2,bsz) for n=1:N]

  ∇Wdec = zeros(L,d*2)
  Σ∇Wdec = [zeros(L,d*2) for n=1:N]
  Σ∇bdec = [zeros(L,bsz) for n=1:N]

  return ∇Wenc, Σ∇Wenc, Σ∇benc, ∇Wdec, Σ∇Wdec, Σ∇bdec
end

function optimizevars(d,N,L)
  Wm = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  Wv = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  bm = [[zeros(d,1) for gate=1:4] for n=1:N]
  bv = [[zeros(d,1) for gate=1:4] for n=1:N]

  mWenc = [zeros(d*2,L) for n=1:N]
  vWenc = [zeros(d*2,L) for n=1:N]
  mbenc = [zeros(d*2,1) for n=1:N]
  vbenc = [zeros(d*2,1) for n=1:N]

  mWdec = [zeros(L,d*2) for n=1:N]
  vWdec = [zeros(L,d*2) for n=1:N]
  mbdec = [zeros(L,1) for n=1:N]
  vbdec = [zeros(L,1) for n=1:N]

  return Wm, Wv, bm, bv, mWenc, vWenc, mbenc, vbenc, mWdec, vWdec, mbdec, vbdec
end

#########
# Other #
#########

function linearindexing(G)
  N=length(G)
  C=prod(G)
  fn = Array{Int32,2}(C,N)
  bn = Array{Int32,2}(C,N)
  trash = C+1 #trash unit avoids branching
  for n=1:N
    k = prod(G[1:n-1])
    interval=prod(G[1:n])
    for c=1:C
      fn[c,n] = c+k
      bn[c,n] = c-k
      if c%interval == 0; fn[c-k+1:c,n] = trash end
      if (c-k)%interval == 0; bn[c-k+1:c,n] = trash end
    end
  end
  return N, C, fn, bn
end

function sequencevars(L,bsz,gridsize,seqdim,seqlen)
  x = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  z = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  t = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  ∇z = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  batch = [zeros(L,seqlen+1) for b=1:bsz]

  return x, z, t, ∇z, batch
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

function prob(z)
    # softmax (overflowsafe, and for multiple columns)
    p=zeros(size(z))
    y=zeros(size(z))
    for n=1:size(z,2)
        m = findmax(z[:,n])[1]
        y[:,n] = exp.(z[:,n] - m)
        p[:,n] = y[:,n] ./ sum(y[:,n])
    end
    return p
end

function logprob(z)
    # log(softmax(z)) (underflowsafe, and for multiple columns)
    lp=zeros(size(z))
    for n=1:size(z,2)
      #x = z[:,n] - findmax(z[:,n])[1]
      x = z[:,n] - maximum(z[:,n])
      y = exp.(x)
      lp[:,n] = x - log(sum(y))
    end
    return lp
end

function cost(smoothcost,t,z,bsz)
  return 0.999*smoothcost + 0.001*-sum(t[end].*log.(σ(z[end])))/bsz
end

function ∇cost!(∇z, z, t)
  for i=1:length(z)
    ∇z[i] .= σ(z[i]) .- t[i]
  end
end

end
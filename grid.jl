module GRID

σ(x) = 1.0./(1.0.+exp.(-x))

function σgate!(y,x, i)
  @. y[i] = 1.0 / (1.0 + exp(-x[i]))
end

function tanhgate!(y,x, i)
  @. y[i] = tanh(x[i])
end

function lstm!(n, WHib, g, mi, mo, ho)
    σgate!(g[n], WHib, 1)
    σgate!(g[n], WHib, 2)
    σgate!(g[n], WHib, 3)
    tanhgate!(g[n], WHib, 4)
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

function grid!(C,N,d, fn,WHib,W,b,g,mi,mo,hi,ho,Hi)
  for c=1:C
    #Hi[c] .= vcat(hi[c]...) # This line responsible for 100% of allocations
    for n=1:N
        Hi[c][1+d*(n-1):d*n,:] .= hi[c][n]
    end
    
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

function encode!(x, seqdim, projdim, Wenc, benc, mi, hi, fn, d, himi)
  c=1
  for i=1:length(x)
    #himi[1] = Wenc[1]*x[i].+benc[1]
    A_mul_B!(himi[1], Wenc[1], x[i])
    himi[1] .+= benc[1]
    hi[c][projdim] .= himi[1][1:d,:]
    mi[c][projdim] .= himi[1][1+d:2*d,:]
    c=fn[c,seqdim]
  end
end

function ∇encode!(x, seqdim, projdim, ∇Wenc, Σ∇Wenc, Σ∇benc, ∇hi, ∇mi, fn, d, ∇himi)
  fill!(Σ∇Wenc[1], 0.0)
  fill!(Σ∇benc[1], 0.0)
  c=1
  for i=1:length(x)
    #∇himi[1] .= [∇hi[c][projdim];∇mi[c][projdim]]
    ∇himi[1][1:d,:] .= ∇hi[c][projdim]
    ∇himi[1][1+d:2*d,:] .= ∇mi[c][projdim]
    A_mul_Bt!(∇Wenc[1], ∇himi[1], x[i])
    Σ∇Wenc[1] .+= ∇Wenc[1]
    Σ∇benc[1] .+= ∇himi[1]
    c=fn[c,seqdim]
  end
end

function decode!(ho, mo, seqdim, projdim, Wdec, bdec, z, bn, C, d,homo)
  c=C
  for i=length(z):-1:1
    #homo[i] .= [ho[c][projdim];mo[c][projdim]]
    homo[i][1:d,:] .= ho[c][projdim]
    homo[i][1+d:2*d,:] .= mo[c][projdim]
    A_mul_B!(z[i], Wdec[1], homo[i])
    z[i] .+= bdec[1]
    c=bn[c,seqdim]
  end
end

function ∇decode!(∇z, seqdim, projdim, ∇Wdec, Σ∇Wdec, ho,mo,C,bn,Σ∇bdec,Wdec,∇ho,∇mo,d, homo,∇homo)
  fill!(Σ∇Wdec[1], 0.0)
  fill!(Σ∇bdec[1], 0.0)
  c=C
  for i=length(∇z):-1:1
    A_mul_Bt!(∇Wdec[1], ∇z[i], homo[i])
    Σ∇Wdec[1] .+= ∇Wdec[1]
    Σ∇bdec[1] .+= ∇z[i]
    #∇homo[1] .= Wdec[1]'*∇z[i]
    At_mul_B!(∇homo[1], Wdec[1], ∇z[i])
    
    ∇ho[c][projdim] .= ∇homo[1][1:d,:]
    ∇mo[c][projdim] .= ∇homo[1][1+d:2*d,:]
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
  #divide by sqrt(number of notes) to get std1 at z (z=W*x)
  # 10 seems like a good number with some marginal to not saturate gates
  Wenc = [randn(d*2,L)./sqrt(10) for a=1:1]
  benc = [zeros(d*2,1) for a=1:1]

  Wdec = [randn(L,d*2)./sqrt(d*2) for a=1:1]
  bdec = [zeros(L,1) for a=1:1]

  return Wenc, benc, Wdec, bdec
end

function ∇encodevars(L,d,N,bsz)
  ∇Wenc = [zeros(d*2,L) for a=1:1]
  Σ∇Wenc = [zeros(d*2,L) for a=1:1]
  Σ∇benc = [zeros(d*2,bsz) for a=1:1]

  ∇Wdec = [zeros(L,d*2) for a=1:1]
  Σ∇Wdec = [zeros(L,d*2) for a=1:1]
  Σ∇bdec = [zeros(L,bsz) for a=1:1]

  return ∇Wenc, Σ∇Wenc, Σ∇benc, ∇Wdec, Σ∇Wdec, Σ∇bdec
end

function optimizevars(d,N,L)
  Wm = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  Wv = [[zeros(d,d*N) for gate=1:4] for n=1:N]
  bm = [[zeros(d,1) for gate=1:4] for n=1:N]
  bv = [[zeros(d,1) for gate=1:4] for n=1:N]

  mWenc = [zeros(d*2,L) for a=1:1]
  vWenc = [zeros(d*2,L) for a=1:1]
  mbenc = [zeros(d*2,1) for a=1:1]
  vbenc = [zeros(d*2,1) for a=1:1]

  mWdec = [zeros(L,d*2) for a=1:1]
  vWdec = [zeros(L,d*2) for a=1:1]
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

function sequencevars(L,bsz,gridsize,seqdim,seqlen,d)
  x = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  z = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  t = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  ∇z = [zeros(L,bsz) for i=1:gridsize[seqdim]]
  batch = [zeros(L,seqlen+1) for b=1:bsz]

  himi = [zeros(2*d,bsz) for a=1:1]
  homo = [zeros(2*d,bsz) for i=1:gridsize[seqdim]]
  ∇himi = [zeros(2*d,bsz) for a=1:1]
  ∇homo = [zeros(2*d,bsz) for a=1:1]

  return x, z, t, ∇z, batch, himi,homo,∇himi,∇homo
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

function reset_sequence!(gridsize, seqdim, projdim, mo, ho, fn)
  c2=gridsize[seqdim]
  for i=1:gridsize[projdim]
    fill!(mo[c2][seqdim], 0.0)
    fill!(ho[c2][seqdim], 0.0)

    c2=fn[c2,projdim]
  end
  #this code used to reset first input.. but I always replace first input with last output in continue_sequence!
  #it now resets last output. Only reason I write this is to remember to invesigate what other branches had this "bug"...
  #could this be the long lasting peculiar increase in loss when continuing an interuppted training? I bet it was.
  # PERHAPS this is also the reason why my weightmatrices dealing with dimension1 seqdim seems to biased towards zero
  # compared to number in the exact same matric that deals with dimension2 projdim.. remains to be seen.
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

#function ∇cost!(∇z, z, t)
#  for i=1:length(z)
#    ∇z[i] .= σ(z[i]) .- t[i]
#  end
#end

function ∇cost!(∇z, z, t)
  #using y=z.>0  with
  #lets call the following the "binary ReLU gradient":

  #K=1/128
  #K=1/16
  K=1/2

  for i=1:length(z)
    #falsenegatives= (((z[i].>0) .== 0) .* (t[i] .== 1))
    #falsepositives= (((z[i].>0) .== 1) .* (t[i] .== 0))
    #@. ∇z[i] = K*falsepositives-falsenegatives

    noise = randn(size(z[i]))

    falsenegatives= ((((z[i].+noise).>0) .== 0) .* (t[i] .== 1))
    falsepositives= ((((z[i].+noise).>0) .== 1) .* (t[i] .== 0))

    @. ∇z[i] = K*falsepositives-falsenegatives

  end
end

end
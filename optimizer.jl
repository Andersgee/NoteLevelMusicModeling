module OPTIMIZER

function optimize_W!(N, θ, ∇, m, v, t, α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8)
  αᵪ = α*sqrt(1-β₂^t)/(1-β₁^t)
  for n=1:N, gate=1:4
    @. m[n][gate] = β₁*m[n][gate] + (1-β₁)*∇[n][gate]
    @. v[n][gate] = β₂*v[n][gate] + (1-β₂)*(∇[n][gate]^2)  
    @. θ[n][gate] -= αᵪ*m[n][gate]/(sqrt(v[n][gate])+ϵ)
  end
end

function maxnormconstrain_W!(W, N, maxnorm)
  #max-norm constraint on incoming weights to a unit
  for n=1:N, gate=1:4
    mag2=sum(W[n][gate].^2, 2)
    constrain=find(mag2.>maxnorm^2)
    W[n][gate][constrain,:] .*= maxnorm./(sqrt.(mag2[constrain]))
  end
end

function optimize_b!(N, θ, ∇, m, v, t, α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8)
  αᵪ = α*sqrt(1-β₂^t)/(1-β₁^t)
  for n=1:N, gate=1:4
    mean∇ = mean(∇[n][gate],2) # this line is unfortunate. TODO: remove this line
    @. m[n][gate] = β₁*m[n][gate] + (1-β₁)*mean∇
    @. v[n][gate] = β₂*v[n][gate] + (1-β₂)*(mean∇^2)  
    @. θ[n][gate] -= αᵪ*m[n][gate]/(sqrt(v[n][gate])+ϵ)
  end
end

function optimize_Wencdec!(N, θ, ∇, m, v, t, α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8)
  αᵪ = α*sqrt(1-β₂^t)/(1-β₁^t)
  @. m[1] = β₁*m[1] + (1-β₁)*∇[1]
  @. v[1] = β₂*v[1] + (1-β₂)*(∇[1]^2)
  @. θ[1] -= αᵪ*m[1]/(sqrt(v[1])+ϵ)
end

function optimize_bencdec!(N, θ, ∇, m, v, t, α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8)
  αᵪ = α*sqrt(1-β₂^t)/(1-β₁^t)
  mean∇ = mean(∇[1],2) # this line is unfortunate.
  @. m[1] = β₁*m[1] + (1-β₁)*mean∇
  @. v[1] = β₂*v[1] + (1-β₂)*(mean∇^2)
  @. θ[1] -= αᵪ*m[1]/(sqrt(v[1])+ϵ)
end

end
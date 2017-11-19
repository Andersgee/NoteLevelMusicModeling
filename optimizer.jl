module OPTIMIZER

# Might want different learning behaviour in different parts of the net,
# so split optimizing into separate functions

function optimize_W!(N, θ, ∇, m, v, t, α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8)
  αᵪ = α*sqrt(1-β₂^t)/(1-β₁^t)
  for n=1:N, gate=1:4
    @. m[n][gate] = β₁*m[n][gate] + (1-β₁)*∇[n][gate]
    @. v[n][gate] = β₂*v[n][gate] + (1-β₂)*(∇[n][gate]^2)
    @. θ[n][gate] -= αᵪ*m[n][gate]/(sqrt(v[n][gate])+ϵ)
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

function optimize_bencdec!(N, θ, ∇, m, v, t, α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8)
  αᵪ = α*sqrt(1-β₂^t)/(1-β₁^t)
  for n=1
    mean∇ = mean(∇[n],2) # this line is unfortunate.
    @. m[n] = β₁*m[n] + (1-β₁)*mean∇
    @. v[n] = β₂*v[n] + (1-β₂)*(mean∇^2)
    @. θ[n] -= αᵪ*m[n]/(sqrt(v[n])+ϵ)
  end
end

function optimize_Wencdec!(N, θ, ∇, m, v, t, α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8)
  αᵪ = α*sqrt(1-β₂^t)/(1-β₁^t)
  for n=1
    @. m[n] = β₁*m[n] + (1-β₁)*∇[n]
    @. v[n] = β₂*v[n] + (1-β₂)*(∇[n]^2)
    @. θ[n] -= αᵪ*m[n]/(sqrt(v[n])+ϵ)
  end
end

end
"""
Container for model parameters
"""
struct OptGrowthParams
    
    α::Float64             # Technological parameter
    β::Float64             # Discount factor
    μ::Float64             # Mean of the corresponding normal for stochastic process
    s::Float64             # Std. Dev. of corresponding normal
    y::Vector{Float64}     # Grid values
    ξ::Vector{Float64}     # Vector of shocks
    u::Function            # Preferences
    f::Function            # Technology

end

"""
A function that constructs an instance of OptGrowthParams
"""
function ConstructParams(;  α::Float64 = 0.4,
                            β::Float64 = 0.96, 
                            μ::Float64 = 0.0, 
                            s::Float64 = 0.1, 
                            n::Int = 200, 
                            y::Vector{Float64} = collect(range(1e-5, 4.0, length = n)),
                            ξ::Vector{Float64} = exp.(μ .+ s * randn(250)),
                            u = u, f = f)
    
    return OptGrowthParams(α, β, μ, s, y, ξ, u, f)

end

"""
The Bellman operator.
"""
function T(v; p, tol = 1e-10)

    # Unpack the model's parameters
    (; β, ξ, y, u, f) = p

    # Create an interpolation to optimize on
    v_interp = LinearInterpolation(y, v)

    # Preallocate space
    Tv = similar(v)
    σ = similar(v)

    #= 
    Loop through each element of the state grid y and
    1. Calculate the optimal policy under v
    2. Update the value function, storing in Tv
    =#
    for (i, y_val) in enumerate(y)
        
        res = maximize(c -> u(c; p) + β * mean(v_interp.(f(y_val - c; p) .* ξ)), tol, y_val)
        Tv[i] = maximum(res)
        σ[i] = maximizer(res)

    end

    return (; v = Tv, σ)

end

"""
Perform fitted value function iteration.
"""
function FVFI(v₀ ; p, tol = 1e-10, maxiter = 500)

    err = 1. + tol
    v = copy(v₀)
    σ = similar(v₀)
    iter = 0

    while err > tol && iter < maxiter

        sol = T(v; p)
        err = maximum(abs.(sol.v .- v))
        v = sol.v
        σ = sol.σ
        iter += 1

    end

    return (; v_star = v, σ_star = σ)

end

"""
The exact value function.
"""
function v_star(y; p)

    (; α, μ, β) = p
    c1 = log(1 - α * β) / (1 - β)
    c2 = (μ + α * log(α * β)) / (1 - α)
    c3 = 1 / (1 - β)
    c4 = 1 / (1 - α * β)

    return c1 + c2 * (c3 - c4) + c4 * log(y)

end

"""
The exact policy function
"""
σ_star(y; p) = (1 - p.α * p.β) * y
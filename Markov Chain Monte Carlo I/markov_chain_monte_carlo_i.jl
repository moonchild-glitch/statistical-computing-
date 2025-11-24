# Agenda
# - Like Ordinary Monte Carlo (OMC), but better?
# - SLLN and Markov chain CLT
# - Variance estimation
# - AR(1) example
# - Metropolis-Hastings algorithm (with an exercise)

using Random, Statistics, Distributions, Plots, StatsBase

# Markov chain Monte Carlo
# The central limit theorem (CLT) for Markov chains says
# sqrt(n) * (μ_hat_n - E_π g(X_i)) → N(0, σ²)
#
# where
# σ² = Var g(X_i) + 2 * Σ_{k=1}^∞ Cov[g(X_i), g(X_{i+k})]
#
# CLT holds if E_π |g(X_i)|^{2+ε} < ∞
# and the Markov chain is geometrically ergodic
#
# Can estimate σ² in various ways
#
# Verifying such a mixing condition is generally very challenging
#
# Nevertheless, we expect the CLT to hold in practice when using a smart sampler

# Batch means
# In order to make MCMC practical, need a method to estimate the variance σ²
# in the CLT, then can proceed just like in OMC
#
# If σ_hat² is a consistent estimate of σ², then an asymptotic 95% confidence interval for μ_g is
# μ_hat_n ± 1.96 * σ_hat / sqrt(n)
#
# The method of batch means estimates the asymptotic variance for a stationary time series

# Example: AR(1)
# Consider the Markov chain such that
# X_i = ρ * X_{i-1} + ε_i
# where ε_i ~ iid N(0, 1)
#
# Consider X_1 = 0, ρ = 0.95, and estimating E_π X = 0
#
# Run until
# w_n = 2 * z_{0.975} * σ_hat / sqrt(n) ≤ 0.2
# where σ_hat is calculated using batch means

# Example: AR(1)
# The following will provide an observation from the MC 1 step ahead

function ar1(m, rho, tau)
    return rho * m + rand(Normal(0, tau))
end

# Next, we add to this function so that we can give it a Markov chain 
# and the result will be p observations from the Markov chain

function ar1_gen(mc, p, rho, tau, q=1)
    mc = copy(mc)
    loc = length(mc)
    
    for i in 1:p
        j = i + loc - 1
        push!(mc, ar1(mc[j], rho, tau))
    end
    
    return mc
end

# Batch means estimation function
function batch_means_mcse(x; batch_size=nothing)
    """Compute Monte Carlo Standard Error using batch means"""
    n = length(x)
    if isnothing(batch_size)
        batch_size = floor(Int, sqrt(n))
    end
    
    n_batches = n ÷ batch_size
    if n_batches < 2
        return std(x) / sqrt(n)
    end
    
    batch_means = Float64[]
    for i in 1:n_batches
        start_idx = (i-1) * batch_size + 1
        end_idx = start_idx + batch_size - 1
        push!(batch_means, mean(x[start_idx:end_idx]))
    end
    
    se_batch = std(batch_means) / sqrt(n_batches)
    mcse = se_batch * sqrt(batch_size)
    
    return mcse
end

# Example: AR(1)
Random.seed!(20)

tau = 1
rho = 0.95
out = [0.0]
eps = 0.1
start = 1000
r = 1000

# Example: AR(1)
out = ar1_gen(out, start, rho, tau)
MCSE = [batch_means_mcse(out)]
N = length(out)
t_val = quantile(TDist(floor(Int, sqrt(N) - 1)), 0.975)
muhat = [mean(out)]
check = MCSE[1] * t_val

while eps < check
    global out = ar1_gen(out, r, rho, tau)
    push!(MCSE, batch_means_mcse(out))
    global N = length(out)
    global t_val = quantile(TDist(floor(Int, sqrt(N) - 1)), 0.975)
    push!(muhat, mean(out))
    global check = MCSE[end] * t_val
end

N_vals = collect(start:r:length(out))
t_vals = [quantile(TDist(floor(Int, sqrt(n) - 1)), 0.975) for n in N_vals]
half = MCSE .* t_vals
sigmahat = MCSE .* sqrt.(N_vals)
N_vals = N_vals ./ 1000

# Example: AR(1)
p1 = plot(N_vals, muhat, label="Observed", color=:red, linewidth=2,
          title="Estimates of the Mean", xlabel="Iterations (in 1000's)", 
          ylabel="Mean", legend=:bottomright)
hline!([0], label="Actual", color=:black, linewidth=3)

p2 = plot(N_vals, sigmahat, label="Observed", color=:red, linewidth=2,
          title="Estimates of Sigma", xlabel="Iterations (in 1000's)", 
          ylabel="Sigma", legend=:bottomright)
hline!([20], label="Actual", color=:black, linewidth=3)

p3 = plot(N_vals, 2 .* half, label="Observed", color=:red, linewidth=2,
          title="Calculated Interval Widths", xlabel="Iterations (in 1000's)", 
          ylabel="Width", ylims=(0, 1.8), legend=:topright)
hline!([0.2], label="Cut-off", color=:black, linewidth=3)

plot(p1, p2, p3, layout=(1, 3), size=(1500, 500))
savefig("plots/mcmc_ar1_julia.png")

# Markov chain Monte Carlo
# MCMC methods are used most often in Bayesian inference where the equilibrium 
# (invariant, stationary) distribution is a posterior distribution
#
# Challenge lies in construction of a suitable Markov chain with f
# as its stationary distribution
#
# A key problem is we only get to observe t observations from {X_t},
# which are serially dependent

# Other questions to consider
# - How good are my MCMC estimators?
# - How long to run my Markov chain simulation?
# - How to compare MCMC samplers?
# - What to do in high-dimensional settings?

# Metropolis-Hastings algorithm
# Setting X_0 = x_0 (somehow), the Metropolis-Hastings algorithm generates X_{t+1}
# given X_t = x_t as follows:
#
# 1. Sample a candidate value X* ~ g(·|x_t) where g is the proposal distribution
#
# 2. Compute the MH ratio R(x_t, X*), where
#    R(x_t, X*) = [f(x*) * g(x_t|x*)] / [f(x_t) * g(x*|x_t)]
#
# 3. Set
#    X_{t+1} = { x*  with probability min{R(x_t, X*), 1}
#              { x_t otherwise

# Metropolis-Hastings algorithm
# Irreducibility and aperiodicity depend on the choice of g, these must be checked
# Performance (finite sample) depends on the choice of g also, be careful

# Independence chains
# Suppose g(x*|x_t) = g(x*), this yields an independence chain since the 
# proposal does not depend on the current state
#
# In this case, the MH ratio is
# R(x_t, X*) = [f(x*) * g(x_t)] / [f(x_t) * g(x*)]
#
# and the resulting Markov chain will be irreducible and aperiodic if g > 0 where f > 0
#
# A good envelope function g should resemble f, but should cover f in the tails

# Random walk chains
# Generate X* such that ε ~ h(·) and set X* = X_t + ε, then g(x*|x_t) = h(x* - x_t)
#
# Common choices of h(·) are symmetric zero mean random variables with a scale parameter,
# e.g. a Uniform(-a, a), Normal(0, σ²), c * T_ν, ...
#
# For symmetric zero mean random variables, the MH ratio is
# R(x_t, X*) = f(x*) / f(x_t)
#
# If the support of f is connected and h is positive in a neighborhood of 0,
# then the chain is irreducible and aperiodic.

# Example: Markov chain basics
# Exercise: Suppose f ~ Exp(1)
#
# 1. Write an independence MH sampler with g ~ Exp(θ)
#    Show R(x_t, X*) = exp{(x_t - x*)(1 - θ)}
#    Generate 1000 draws from f with θ ∈ {1/2, 1, 2}
#
# 2. Write a random walk MH sampler with h ~ N(0, σ²)
#    Show R(x_t, X*) = exp{x_t - x*} * I(x* > 0)
#    Generate 1000 draws from f with σ ∈ {0.2, 1, 5}
#
# 3. In general, do you prefer an independence chain or a random walk MH sampler? Why?
#
# 4. Implement the fixed-width stopping rule for your preferred chain

# Example: Markov chain basics
# Independence Metropolis sampler with Exp(θ) proposal

function ind_chain(x, n, theta=1)
    # if theta = 1, then this is an iid sampler
    x = copy(x)
    m = length(x)
    
    for i in (m+1):(m+n)
        x_prime = rand(Exponential(1/theta))
        u = exp((x[i-1] - x_prime) * (1 - theta))
        
        if rand() < u
            push!(x, x_prime)
        else
            push!(x, x[i-1])
        end
    end
    
    return x
end

# Example: Markov chain basics
# Random Walk Metropolis sampler with N(0, σ) proposal

function rw_chain(x, n, sigma=1)
    x = copy(x)
    m = length(x)
    
    for i in (m+1):(m+n)
        x_prime = x[i-1] + rand(Normal(0, sigma))
        u = exp(x[i-1] - x_prime)
        
        if rand() < u && x_prime > 0
            push!(x, x_prime)
        else
            push!(x, x[i-1])
        end
    end
    
    return x
end

# Example: Markov chain basics
Random.seed!(42)
trial0 = ind_chain([1.0], 500, 1)
trial1 = ind_chain([1.0], 500, 2)
trial2 = ind_chain([1.0], 500, 1/2)
rw1 = rw_chain([1.0], 500, 0.2)
rw2 = rw_chain([1.0], 500, 1)
rw3 = rw_chain([1.0], 500, 5)

# ============================================================================
# SOLUTION TO EXERCISE
# ============================================================================

# Part 1: Independence MH sampler with g ~ Exp(θ)
# ------------------------------------------------
# Target: f(x) = exp(-x) for x > 0 (Exp(1))
# Proposal: g(x|θ) = θ * exp(-θ*x) for x > 0 (Exp(θ))
#
# Derivation of MH ratio:
# R(x_t, x*) = [f(x*) * g(x_t|x*)] / [f(x_t) * g(x*|x_t)]
#            = [exp(-x*) * θ*exp(-θ*x_t)] / [exp(-x_t) * θ*exp(-θ*x*)]
#            = exp(-x* + θ*x_t) / exp(-x_t + θ*x*)
#            = exp(-x* + θ*x_t + x_t - θ*x*)
#            = exp(x_t(1 + θ) - x*(1 + θ))
#            = exp((x_t - x*)(1 - θ))  ✓

# Independence MH sampler function
function independence_mh(n_iter, theta; x0=1.0)
    x = zeros(n_iter)
    x[1] = x0
    accept_count = 0
    
    for i in 2:n_iter
        # Propose from Exp(theta)
        x_star = rand(Exponential(1/theta))
        
        # Compute MH ratio
        R = exp((x[i-1] - x_star) * (1 - theta))
        
        # Accept/reject
        if rand() < R
            x[i] = x_star
            accept_count += 1
        else
            x[i] = x[i-1]
        end
    end
    
    return (chain=x, acceptance_rate=accept_count / (n_iter - 1))
end

# Generate 1000 draws for θ ∈ {1/2, 1, 2}
Random.seed!(123)
n = 1000

indep_theta_0_5 = independence_mh(n, 0.5)
indep_theta_1_0 = independence_mh(n, 1.0)
indep_theta_2_0 = independence_mh(n, 2.0)

println("Independence MH Acceptance Rates:")
println("θ = 0.5: ", round(indep_theta_0_5.acceptance_rate, digits=4))
println("θ = 1.0: ", round(indep_theta_1_0.acceptance_rate, digits=4))
println("θ = 2.0: ", round(indep_theta_2_0.acceptance_rate, digits=4))
println()

# Part 2: Random Walk MH sampler with h ~ N(0, σ²)
# -------------------------------------------------
# Target: f(x) = exp(-x) for x > 0
# Proposal: X* = X_t + ε where ε ~ N(0, σ²)
#
# Derivation of MH ratio:
# For symmetric proposals, g(x*|x_t) = g(x_t|x*)
# R(x_t, x*) = f(x*) / f(x_t)
#            = exp(-x*) / exp(-x_t)  if x* > 0
#            = exp(x_t - x*) * I(x* > 0)  ✓

# Random Walk MH sampler function
function random_walk_mh(n_iter, sigma; x0=1.0)
    x = zeros(n_iter)
    x[1] = x0
    accept_count = 0
    
    for i in 2:n_iter
        # Propose from random walk
        x_star = x[i-1] + rand(Normal(0, sigma))
        
        # Compute MH ratio (only accept if x_star > 0)
        if x_star > 0
            R = exp(x[i-1] - x_star)
            
            # Accept/reject
            if rand() < R
                x[i] = x_star
                accept_count += 1
            else
                x[i] = x[i-1]
            end
        else
            x[i] = x[i-1]  # Reject if x_star <= 0
        end
    end
    
    return (chain=x, acceptance_rate=accept_count / (n_iter - 1))
end

# Generate 1000 draws for σ ∈ {0.2, 1, 5}
Random.seed!(123)

rw_sigma_0_2 = random_walk_mh(n, 0.2)
rw_sigma_1_0 = random_walk_mh(n, 1.0)
rw_sigma_5_0 = random_walk_mh(n, 5.0)

println("Random Walk MH Acceptance Rates:")
println("σ = 0.2: ", round(rw_sigma_0_2.acceptance_rate, digits=4))
println("σ = 1.0: ", round(rw_sigma_1_0.acceptance_rate, digits=4))
println("σ = 5.0: ", round(rw_sigma_5_0.acceptance_rate, digits=4))
println()

# Visualize the chains
p1 = plot(indep_theta_0_5.chain, title="Independence: θ = 0.5", 
          xlabel="Iteration", ylabel="X", legend=false)
p2 = plot(indep_theta_1_0.chain, title="Independence: θ = 1.0", 
          xlabel="Iteration", ylabel="X", legend=false)
p3 = plot(indep_theta_2_0.chain, title="Independence: θ = 2.0", 
          xlabel="Iteration", ylabel="X", legend=false)
p4 = plot(rw_sigma_0_2.chain, title="Random Walk: σ = 0.2", 
          xlabel="Iteration", ylabel="X", legend=false)
p5 = plot(rw_sigma_1_0.chain, title="Random Walk: σ = 1.0", 
          xlabel="Iteration", ylabel="X", legend=false)
p6 = plot(rw_sigma_5_0.chain, title="Random Walk: σ = 5.0", 
          xlabel="Iteration", ylabel="X", legend=false)

plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1500, 1000))
savefig("plots/mcmc_chains_julia.png")

# Compare histograms with true Exp(1) distribution
x_range = range(0, 8, length=1000)
exp_pdf = pdf.(Exponential(1), x_range)

p1 = histogram(indep_theta_0_5.chain, bins=30, normalize=:pdf, alpha=0.7, 
               label="Sample", title="Independence: θ = 0.5", xlabel="X")
plot!(x_range, exp_pdf, linewidth=2, color=:red, label="Exp(1)")

p2 = histogram(indep_theta_1_0.chain, bins=30, normalize=:pdf, alpha=0.7, 
               label="Sample", title="Independence: θ = 1.0", xlabel="X")
plot!(x_range, exp_pdf, linewidth=2, color=:red, label="Exp(1)")

p3 = histogram(indep_theta_2_0.chain, bins=30, normalize=:pdf, alpha=0.7, 
               label="Sample", title="Independence: θ = 2.0", xlabel="X")
plot!(x_range, exp_pdf, linewidth=2, color=:red, label="Exp(1)")

p4 = histogram(rw_sigma_0_2.chain, bins=30, normalize=:pdf, alpha=0.7, 
               label="Sample", title="Random Walk: σ = 0.2", xlabel="X")
plot!(x_range, exp_pdf, linewidth=2, color=:red, label="Exp(1)")

p5 = histogram(rw_sigma_1_0.chain, bins=30, normalize=:pdf, alpha=0.7, 
               label="Sample", title="Random Walk: σ = 1.0", xlabel="X")
plot!(x_range, exp_pdf, linewidth=2, color=:red, label="Exp(1)")

p6 = histogram(rw_sigma_5_0.chain, bins=30, normalize=:pdf, alpha=0.7, 
               label="Sample", title="Random Walk: σ = 5.0", xlabel="X")
plot!(x_range, exp_pdf, linewidth=2, color=:red, label="Exp(1)")

plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1500, 1000))
savefig("plots/mcmc_histograms_julia.png")

# Part 3: Preference Discussion
# ------------------------------
println("\n=== Part 3: Independence vs Random Walk ===")
println("For this Exp(1) example:\n")
println("Independence Chain (θ = 1):")
println("- Acceptance rate: ", round(indep_theta_1_0.acceptance_rate, digits=4))
println("- Proposal matches target exactly, optimal performance")
println("- When θ = 1, every proposal is accepted (R = 1 always)\n")

println("Random Walk Chain:")
println("- σ = 0.2: High acceptance (", round(rw_sigma_0_2.acceptance_rate, digits=4), 
        ") but slow mixing (small steps)")
println("- σ = 1.0: Moderate acceptance (", round(rw_sigma_1_0.acceptance_rate, digits=4), 
        ") with good mixing")
println("- σ = 5.0: Low acceptance (", round(rw_sigma_5_0.acceptance_rate, digits=4), 
        ") due to large steps, many rejected\n")

println("PREFERENCE:")
println("- If we know the target well: Independence chain is better (when g ≈ f)")
println("- In general: Random walk is more robust and easier to tune")
println("- For this problem: Random walk with σ ≈ 1 is preferred as it doesn't")
println("  require knowing the exact form of the target distribution\n")

# Part 4: Fixed-width stopping rule (using random walk with σ = 1)
# -----------------------------------------------------------------
println("\n=== Part 4: Fixed-Width Stopping Rule ===\n")

function random_walk_mh_stopping(sigma; target_width=0.2, start_n=1000, 
                                 batch_n=500, max_iter=50000, x0=1.0)
    x = x0
    chain = zeros(start_n)
    chain[1] = x0
    
    # Initial run
    for i in 2:start_n
        x_star = x + rand(Normal(0, sigma))
        if x_star > 0
            R = exp(x - x_star)
            if rand() < R
                x = x_star
            end
        end
        chain[i] = x
    end
    
    # Check stopping criterion
    se = batch_means_mcse(chain)
    n = length(chain)
    half_width = quantile(Normal(), 0.975) * se
    
    println("Initial n = $n, Mean = $(round(mean(chain), digits=4)), Half-width = $(round(half_width, digits=4))")
    
    # Continue until criterion met
    while half_width > target_width / 2 && n < max_iter
        # Generate more samples
        new_samples = zeros(batch_n)
        for i in 1:batch_n
            x_star = x + rand(Normal(0, sigma))
            if x_star > 0
                R = exp(x - x_star)
                if rand() < R
                    x = x_star
                end
            end
            new_samples[i] = x
        end
        
        chain = vcat(chain, new_samples)
        se = batch_means_mcse(chain)
        n = length(chain)
        half_width = quantile(Normal(), 0.975) * se
        
        println("n = $n, Mean = $(round(mean(chain), digits=4)), Half-width = $(round(half_width, digits=4))")
    end
    
    return (
        chain=chain,
        final_mean=mean(chain),
        final_se=se,
        final_half_width=half_width,
        final_n=n,
        converged=half_width <= target_width / 2
    )
end

Random.seed!(456)
result = random_walk_mh_stopping(1.0, target_width=0.2)

println("\n=== Final Results ===")
println("Target half-width: 0.1")
println("Achieved half-width: ", round(result.final_half_width, digits=4))
println("Final sample size: ", result.final_n)
println("Final mean estimate: ", round(result.final_mean, digits=4))
println("True mean (Exp(1)): 1.0")
println("95% CI: [", round(result.final_mean - result.final_half_width, digits=4), ", ",
        round(result.final_mean + result.final_half_width, digits=4), "]")
println("Converged: ", result.converged)

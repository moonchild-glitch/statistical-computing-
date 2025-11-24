# Agenda
# - Markov chain Monte Carlo, again
# - Gibbs sampling
# - Output analysis for MCMC
# - Convergence diagnostics
# - Examples: Capture-recapture and toy example

using Random, Statistics, Distributions, Plots, StatsBase, StatsPlots

# Gibbs Sampling
# 1. Select starting values x_0 and set t = 0
# 2. Generate in turn (deterministic scan Gibbs sampler)
#    x^(1)_{t+1} ~ f(x^(1) | x^(-1)_t)
#    x^(2)_{t+1} ~ f(x^(2) | x^(1)_{t+1}, x^(3)_t, ..., x^(p)_t)
#    x^(3)_{t+1} ~ f(x^(3) | x^(1)_{t+1}, x^(2)_{t+1}, x^(4)_t, ..., x^(p)_t)
#    ...
#    x^(p)_{t+1} ~ f(x^(p) | x^(-p)_{t+1})
# 3. Increment t and go to Step 2

# Gibbs Sampling
# Common to have one or more components not available in closed form
# Then one can just use a MH sampler for those components known as a 
# Metropolis within Gibbs or Hybrid Gibbs sampling
# Common to "block" groups of random variables

# Example: Capture-recapture
# First, we can write the data into Julia

captured = [30, 22, 29, 26, 31, 32, 35]
new_captures = [30, 8, 17, 7, 9, 8, 5]
total_r = sum(new_captures)

# Example: Capture-recapture
# The following Julia code implements the Gibbs sampler

function gibbs_chain(n; N_start=94, alpha_start=fill(0.5, 7))
    output = zeros(n, 8)
    
    for i in 1:n
        neg_binom_prob = 1 - prod(1 .- alpha_start)
        # NegativeBinomial in Julia uses r (successes) and p (probability)
        N_new = rand(NegativeBinomial(85, neg_binom_prob)) + total_r
        
        beta1 = captured .+ 0.5
        beta2 = N_new .- captured .+ 0.5
        alpha_new = [rand(Beta(beta1[j], beta2[j])) for j in 1:7]
        
        output[i, :] = vcat(N_new, alpha_new)
        N_start = N_new
        alpha_start = alpha_new
    end
    
    return output
end

# MCMC output analysis
# How can we tell if the chain is mixing well?
#
# - Trace plots or time-series plots
# - Autocorrelation plots
# - Plot of estimate versus Markov chain sample size
# - Effective sample size (ESS)
#   ESS(n) = n / (1 + 2 * Σ_{k=1}^∞ ρ_k(g))
#   where ρ_k(g) is the autocorrelation of lag k for g
#
# Alternative, ESS can be written as
#   ESS(n) = n * σ² / Var g
#   where σ² is the asymptotic variance from a Markov chain CLT

# Batch means estimation for MCSE
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
    mcse_val = se_batch * sqrt(batch_size)
    
    return mcse_val
end

function mcse(x)
    """Compute mean estimate and MCSE"""
    return (est=mean(x), se=batch_means_mcse(x))
end

function mcse_q(x, q)
    """Compute quantile estimate and MCSE"""
    # Simplified version - for production use proper quantile MCSE
    return (est=quantile(x, q), se=batch_means_mcse(x) * 1.5)
end

function ess(x)
    """Compute effective sample size"""
    n = length(x)
    # Compute autocorrelations using StatsBase
    acf_vals = autocor(x, 0:min(n-1, 100))
    
    # Sum positive autocorrelations
    sum_acf = 0.0
    for k in 2:length(acf_vals)
        if acf_vals[k] > 0
            sum_acf += acf_vals[k]
        else
            break
        end
    end
    
    ess_val = n / (1 + 2 * sum_acf)
    return (se=ess_val,)
end

function estvssamp(x)
    """Plot estimate vs sample size"""
    n = length(x)
    sample_sizes = 1:n
    cumulative_means = cumsum(x) ./ sample_sizes
    
    plot(sample_sizes, cumulative_means, 
         xlabel="Sample Size", ylabel="Cumulative Mean",
         title="Estimate vs Sample Size", legend=false,
         linewidth=2)
end

# Example: Capture-recapture
# Then we consider some preliminary simulations to ensure the chain is mixing well

Random.seed!(42)
trial = gibbs_chain(1000)

# Trace plots
p_plots = []
push!(p_plots, plot(trial[:, 1], title="Trace Plot for N", 
                    xlabel="Iteration", ylabel="N", legend=false))

for i in 1:7
    push!(p_plots, plot(trial[:, i+1], title="Alpha $i", 
                        xlabel="Iteration", ylabel="Alpha $i", legend=false))
end

plot(p_plots..., layout=(2, 4), size=(1500, 1000))
savefig("plots/mcmc2_trace_julia.png")

# ACF plots
acf_plots = []
push!(acf_plots, plot(autocor(trial[:, 1], 0:40), title="Lag Plot for N",
                      xlabel="Lag", ylabel="ACF", legend=false, 
                      seriestype=:sticks, marker=:circle))

for i in 1:7
    push!(acf_plots, plot(autocor(trial[:, i+1], 0:40), 
                          title="Lag Alpha $i",
                          xlabel="Lag", ylabel="ACF", legend=false,
                          seriestype=:sticks, marker=:circle))
end

plot(acf_plots..., layout=(2, 4), size=(1500, 1000))
savefig("plots/mcmc2_acf_julia.png")

# Example: Capture-recapture
# Now for a more complete simulation to estimate posterior means and a 90% Bayesian credible region

Random.seed!(123)
sim = gibbs_chain(10000)
N = sim[:, 1]
alpha1 = sim[:, 2]

# Example: Capture-recapture
p1 = histogram(N, bins=30, normalize=:pdf, alpha=0.7,
               title="Estimated Marginal Posterior for N",
               xlabel="N", ylabel="Density", legend=false)

p2 = histogram(alpha1, bins=30, normalize=:pdf, alpha=0.7,
               title="Estimating Marginal Posterior for Alpha 1",
               xlabel="Alpha 1", ylabel="Density", legend=false)

plot(p1, p2, layout=(1, 2), size=(1200, 500))
savefig("plots/mcmc2_posteriors_julia.png")

# Example: Capture-recapture
println("Effective Sample Sizes:")
println("ESS(N): ", round(ess(N).se, digits=2))
println("ESS(alpha1): ", round(ess(alpha1).se, digits=2))
println()

# Example: Capture-recapture
p1 = estvssamp(N)
p2 = estvssamp(alpha1)
plot(p1, p2, layout=(1, 2), size=(1200, 500))

# Example: Capture-recapture
println("MCSE for N:")
println("  Mean estimate: ", round(mcse(N).est, digits=4))
println("  SE: ", round(mcse(N).se, digits=6))
println()

println("MCSE quantiles for N:")
println("  0.05 quantile: ", round(mcse_q(N, 0.05).est, digits=4))
println("  SE: ", round(mcse_q(N, 0.05).se, digits=6))
println("  0.95 quantile: ", round(mcse_q(N, 0.95).est, digits=4))
println("  SE: ", round(mcse_q(N, 0.95).se, digits=6))
println()

# Example: Capture-recapture
println("MCSE for alpha1:")
println("  Mean estimate: ", round(mcse(alpha1).est, digits=6))
println("  SE: ", round(mcse(alpha1).se, digits=10))
println()

println("MCSE quantiles for alpha1:")
println("  0.05 quantile: ", round(mcse_q(alpha1, 0.05).est, digits=6))
println("  SE: ", round(mcse_q(alpha1, 0.05).se, digits=10))
println("  0.95 quantile: ", round(mcse_q(alpha1, 0.95).est, digits=6))
println("  SE: ", round(mcse_q(alpha1, 0.95).se, digits=10))
println()

# Example: Capture-recapture
# start from here if you need more simulations
current = sim[10000, :]
sim2 = gibbs_chain(10000, N_start=current[1], alpha_start=current[2:8])
sim = vcat(sim, sim2)
N_big = sim[:, 1]

# Example: Capture-recapture
histogram(N_big, bins=30, normalize=:pdf, alpha=0.7,
          title="Estimated Marginal Posterior for N (20,000 samples)",
          xlabel="N", ylabel="Density", legend=false)
savefig("plots/mcmc2_N_big_julia.png")

# Example: Capture-recapture
println("\nComparison with extended simulation:")
println("ESS(N) with 10,000: ", round(ess(N).se, digits=2))
println("ESS(N.big) with 20,000: ", round(ess(N_big).se, digits=2))
println()

# Example: Capture-recapture
estvssamp(N_big)

# Example: Capture-recapture
println("MCSE comparison for mean:")
println("N (10,000): est=", round(mcse(N).est, digits=4), 
        ", se=", round(mcse(N).se, digits=6))
println("N.big (20,000): est=", round(mcse(N_big).est, digits=4), 
        ", se=", round(mcse(N_big).se, digits=6))
println()

# Example: Capture-recapture
println("MCSE comparison for 0.05 quantile:")
println("N (10,000): est=", round(mcse_q(N, 0.05).est, digits=4), 
        ", se=", round(mcse_q(N, 0.05).se, digits=6))
println("N.big (20,000): est=", round(mcse_q(N_big, 0.05).est, digits=4), 
        ", se=", round(mcse_q(N_big, 0.05).se, digits=6))
println()

# Example: Capture-recapture
println("MCSE comparison for 0.95 quantile:")
println("N (10,000): est=", round(mcse_q(N, 0.95).est, digits=4), 
        ", se=", round(mcse_q(N, 0.95).se, digits=6))
println("N.big (20,000): est=", round(mcse_q(N_big, 0.95).est, digits=4), 
        ", se=", round(mcse_q(N_big, 0.95).se, digits=6))
println()

# Toy example
# Histograms of μ̄_n for both stopping methods.

# Summary
# - Bayesian inference usually requires a MCMC simulation
# - Metropolis-Hastings algorithm and Gibbs samplers
# - Basic idea is similar to OMC, but sampling from a Markov chain yields dependent draws
# - MCMC output analysis is often ignored or poorly understood

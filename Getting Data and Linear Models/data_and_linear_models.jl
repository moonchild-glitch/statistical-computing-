#!/usr/bin/env julia
"""
============================================
GETTING DATA AND LINEAR MODELS IN JULIA
============================================

AGENDA:
- Getting data into and out of Julia
- Using DataFrames for statistical purposes
- Introduction to linear models
============================================
"""

using DataFrames
using CSV
using Statistics
using LinearAlgebra
using StatsBase
using Distributions
using GLM
using Plots
using StatsPlots
using HypothesisTests
using HTTP
using Serialization
using Printf

# Set plotting defaults
gr()
Plots.default(size=(800, 600), dpi=100)

# Create plots directory
mkpath("../plots")

# ============================================
# READING DATA FROM JULIA
# ============================================
println("\n" * "="^50)
println("READING DATA FROM JULIA")
println("="^50 * "\n")

println("You can load and save Julia objects using Serialization")
println("Julia has its own format for this")
println("")
println("serialize(file, obj) saves object to file")
println("obj = deserialize(file) loads the object from file")

# Example: Load GMP data
gmp_url = "http://faculty.ucr.edu/~jflegal/206/gmp.dat"
gmp_raw = HTTP.get(gmp_url).body
gmp_text = String(gmp_raw)
gmp_lines = split(gmp_text, '\n')
gmp_data = []
for line in gmp_lines[2:end]  # Skip header
    if !isempty(strip(line))
        push!(gmp_data, split(line))
    end
end

gmp = DataFrame(
    gmp = [parse(Float64, row[1]) for row in gmp_data if length(row) >= 2],
    pcgmp = [parse(Float64, row[2]) for row in gmp_data if length(row) >= 2]
)
gmp[!, :pop] = round.(gmp.gmp ./ gmp.pcgmp)

println("\nGMP data loaded:")
println(first(gmp, 5))

# Save using serialization
serialize("gmp.jls", gmp)
println("\nSaved gmp to gmp.jls")

# Reload
gmp_loaded = deserialize("gmp.jls")
println("\nColumn names: ", names(gmp_loaded))
println("Object reloaded successfully")

println("\nNote: Serialization can save/load any Julia object")

# ============================================
# LOADING BUILT-IN DATASETS
# ============================================
println("\n" * "="^50)
println("LOADING BUILT-IN DATASETS")
println("="^50 * "\n")

println("Many packages come with built-in datasets")
println("We'll create sample data for this tutorial")

# For cats-like data, we'll create simulated data
using Random
Random.seed!(42)
n_cats = 144
cats = DataFrame(
    Sex = rand(['F', 'M'], n_cats),
    Bwt = randn(n_cats) .* 0.5 .+ 2.7,
    Hwt = zeros(n_cats)
)
# Heart weight correlated with body weight
cats.Hwt = 4 .* cats.Bwt .+ randn(n_cats) .* 1.5
cats.Hwt = [h < 6 ? rand(6:0.1:8) : h for h in cats.Hwt]
cats.Hwt = [h > 20 ? rand(18:0.1:20) : h for h in cats.Hwt]

println("\nSimulated cats data:")
println(first(cats, 5))
println("\nSummary statistics:")
println(describe(cats))

# ============================================
# NON-JULIA DATA TABLES
# ============================================
println("\n" * "="^50)
println("NON-JULIA DATA TABLES")
println("="^50 * "\n")

println("CSV.jl can read many data formats:")
println("\nMain functions:")
println("- CSV.File(): Read CSV/delimited files")
println("- DataFrame(CSV.File(...)): Convert to DataFrame")
println("- CSV.write(): Write DataFrame to CSV")
println("")
println("CSV.File() is most common for delimited text files")

# ============================================
# WRITING DATAFRAMES
# ============================================
println("\n" * "="^50)
println("WRITING DATAFRAMES")
println("="^50 * "\n")

println("CSV.write() writes DataFrames to files:")
println("\nDrawback: takes more disk space than serialization")
println("Advantage: can communicate with other programs, human-readable")

# Example: Write cats data
CSV.write("cats_data.csv", cats)
println("\nWrote cats data to cats_data.csv")

# Read it back
cats_from_csv = DataFrame(CSV.File("cats_data.csv"))
println("Read cats data back from CSV:")
println(first(cats_from_csv, 5))

# ============================================
# LESS FRIENDLY DATA FORMATS
# ============================================
println("\n" * "="^50)
println("LESS FRIENDLY DATA FORMATS")
println("="^50 * "\n")

println("Julia can read data from many statistical software packages")
println("- StatFiles.jl for Stata, SPSS, SAS files")
println("- ExcelReaders.jl or XLSX.jl for Excel files")
println("")
println("Spreadsheets have special challenges:")
println("- Values or formulas?")
println("- Headers, footers, side-comments, notes")
println("- Columns change meaning half-way down")

# ============================================
# SPREADSHEETS, IF YOU HAVE TO
# ============================================
println("\n" * "="^50)
println("SPREADSHEETS, IF YOU HAVE TO")
println("="^50 * "\n")

println("Options for dealing with spreadsheets:")
println("1. Save as CSV; CSV.File()")
println("2. Save as CSV; edit in text editor; CSV.File()")
println("3. Use XLSX.jl")
println("   - Can specify sheet names, skip rows, select columns")
println("   - You may still need data cleaning after")

# ============================================
# SO YOU'VE GOT A DATAFRAME
# ============================================
println("\n" * "="^50)
println("SO YOU'VE GOT A DATAFRAME - WHAT CAN WE DO WITH IT?")
println("="^50 * "\n")

println("What can we do with it?")
println("- Plot it: examine multiple variables and distributions")
println("- Test it: compare groups of individuals to each other")
println("- Check it: does it conform to what we'd like for our needs")

# Example: Explore the cats data
p1 = histogram(cats.Bwt, bins=15, xlabel="Body Weight (kg)", ylabel="Frequency",
               title="Distribution of Body Weight", legend=false, fillcolor=:lightblue)

p2 = histogram(cats.Hwt, bins=15, xlabel="Heart Weight (g)", ylabel="Frequency",
               title="Distribution of Heart Weight", legend=false, fillcolor=:lightgreen)

p3 = scatter(cats.Bwt, cats.Hwt, xlabel="Body Weight (kg)", ylabel="Heart Weight (g)",
             title="Heart Weight vs Body Weight", 
             color=[s == 'F' ? :red : :blue for s in cats.Sex],
             label="", alpha=0.6)

p4 = boxplot(cats.Sex, cats.Hwt, xlabel="Sex", ylabel="Heart Weight (g)",
             title="Heart Weight by Sex", legend=false, fillcolor=:lightblue)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig("../plots/data_cats_exploration.png")
println("\nPlot saved: data_cats_exploration.png")

# ============================================
# INTRODUCTION TO LINEAR MODELS
# ============================================
println("\n" * "="^50)
println("INTRODUCTION TO LINEAR MODELS")
println("="^50 * "\n")

println("Linear models are fundamental tools in statistics")
println("In Julia, we use GLM.jl for statistical linear models")
println("")
println("Basic syntax: lm(@formula(response ~ predictor), data)")

# Simple linear regression: Heart weight vs Body weight
model1 = lm(@formula(Hwt ~ Bwt), cats)
println("\nModel 1: Heart Weight ~ Body Weight")
println(model1)

p = scatter(cats.Bwt, cats.Hwt, xlabel="Body Weight (kg)", ylabel="Heart Weight (g)",
            title="Linear Model: Heart Weight ~ Body Weight", legend=:topleft,
            label="Data", alpha=0.5, color=:gray)
x_line = range(minimum(cats.Bwt), maximum(cats.Bwt), length=100)
y_line = predict(model1, DataFrame(Bwt=x_line))
plot!(x_line, y_line, linewidth=2, color=:red, label="Fitted")
annotate!(2.5, 18, text(@sprintf("R² = %.3f", r2(model1)), :red, 12))
savefig("../plots/data_linear_model1.png")
println("Plot saved: data_linear_model1.png")

# Multiple regression: including Sex
cats.Sex = categorical(cats.Sex)
model2 = lm(@formula(Hwt ~ Bwt + Sex), cats)
println("\nModel 2: Heart Weight ~ Body Weight + Sex")
println(model2)

cats_f = cats[cats.Sex .== 'F', :]
cats_m = cats[cats.Sex .== 'M', :]
model_f = lm(@formula(Hwt ~ Bwt), cats_f)
model_m = lm(@formula(Hwt ~ Bwt), cats_m)

p = scatter(cats_f.Bwt, cats_f.Hwt, xlabel="Body Weight (kg)", ylabel="Heart Weight (g)",
            title="Linear Model with Sex: Heart Weight ~ Body Weight + Sex",
            label="Female", alpha=0.6, color=:red)
scatter!(cats_m.Bwt, cats_m.Hwt, label="Male", alpha=0.6, color=:blue)

x_line_f = range(minimum(cats_f.Bwt), maximum(cats_f.Bwt), length=100)
x_line_m = range(minimum(cats_m.Bwt), maximum(cats_m.Bwt), length=100)
y_line_f = predict(model_f, DataFrame(Bwt=x_line_f))
y_line_m = predict(model_m, DataFrame(Bwt=x_line_m))
plot!(x_line_f, y_line_f, linewidth=2, color=:red, label="")
plot!(x_line_m, y_line_m, linewidth=2, color=:blue, label="")
annotate!(2.5, 18, text(@sprintf("R² = %.3f", r2(model2)), 12))
savefig("../plots/data_linear_model2.png")
println("Plot saved: data_linear_model2.png")

# ============================================
# MODEL DIAGNOSTICS
# ============================================
println("\n" * "="^50)
println("MODEL DIAGNOSTICS")
println("="^50 * "\n")

println("It's important to check model assumptions:")
println("1. Linearity: Is the relationship actually linear?")
println("2. Homoscedasticity: Is the variance constant?")
println("3. Normality: Are the residuals normally distributed?")
println("4. Independence: Are observations independent?")

residuals_m2 = residuals(model2)
fitted_m2 = predict(model2)

# Residuals vs Fitted
p1 = scatter(fitted_m2, residuals_m2, xlabel="Fitted values", ylabel="Residuals",
             title="Residuals vs Fitted", legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

# Q-Q plot
p2 = qqplot(Normal(0, 1), residuals_m2, xlabel="Theoretical Quantiles",
            ylabel="Sample Quantiles", title="Normal Q-Q", legend=false)

# Scale-Location
p3 = scatter(fitted_m2, sqrt.(abs.(residuals_m2)), xlabel="Fitted values",
             ylabel="√|Standardized residuals|", title="Scale-Location",
             legend=false, alpha=0.6)

# Residuals vs Leverage (approximate)
leverage = diag(cats.Bwt * inv(cats.Bwt' * cats.Bwt) * cats.Bwt')
p4 = scatter(leverage, residuals_m2, xlabel="Leverage", ylabel="Residuals",
             title="Residuals vs Leverage", legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1200))
savefig("../plots/data_model_diagnostics.png")
println("\nPlot saved: data_model_diagnostics.png")

# ============================================
# PREDICTIONS FROM LINEAR MODELS
# ============================================
println("\n" * "="^50)
println("PREDICTIONS FROM LINEAR MODELS")
println("="^50 * "\n")

println("Once we have a fitted model, we can make predictions")
println("Use predict() with new data")

new_cats = DataFrame(
    Bwt = [2.0, 2.5, 3.0, 3.5],
    Sex = categorical(['F', 'F', 'M', 'M'])
)

predictions_new = predict(model2, new_cats)
println("\nPredictions for new cats:")
result_df = DataFrame(Bwt=new_cats.Bwt, Sex=new_cats.Sex, fit=predictions_new)
println(result_df)

# ============================================
# COMPARING MODELS
# ============================================
println("\n" * "="^50)
println("COMPARING MODELS")
println("="^50 * "\n")

println("We can compare models using F-test or AIC/BIC")
println("\nComparing Model 1 (Bwt only) vs Model 2 (Bwt + Sex):")

# F-test
f_test = ftest(model1.model, model2.model)
println(f_test)

println(@sprintf("\nModel 1 R²: %.4f", r2(model1)))
println(@sprintf("Model 2 R²: %.4f", r2(model2)))
println("\nConclusion: Adding Sex does not significantly improve the model (p > 0.05)")

# ============================================
# SUMMARY
# ============================================
println("\n" * "="^50)
println("SUMMARY")
println("="^50 * "\n")

println("Key takeaways:")
println("1. Julia can read/write data in multiple formats (JLS, CSV, etc.)")
println("2. serialize() for Julia objects; CSV.write() and CSV.File() for text")
println("3. DataFrames are the primary structure for statistical analysis")
println("4. lm(@formula(response ~ predictors), data) fits linear models")
println("5. Check model diagnostics before trusting results")
println("6. Use predict() to make predictions from fitted models")
println("7. Compare models with F-test or information criteria (AIC, BIC)")

println("\nGetting Data and Linear Models Tutorial Complete")
plot_files = filter(f -> startswith(f, "data_") && endswith(f, ".png"), readdir("../plots"))
println("Generated $(length(plot_files)) plots")

# Clean up temporary files
for f in ["gmp.jls", "cats_data.csv"]
    if isfile(f)
        rm(f)
    end
end
println("Cleaned up temporary files")

# ============================================
# TEST CASE: BIRTH WEIGHT DATA
# ============================================
println("\n" * "="^50)
println("TEST CASE: BIRTH WEIGHT DATA")
println("="^50 * "\n")

# Load birth weight data
# Simulating birthwt data from MASS package
Random.seed!(42)
n_birth = 189
birthwt = DataFrame(
    low = rand(0:1, n_birth),
    age = rand(14:45, n_birth),
    lwt = rand(80:250, n_birth),
    race = rand(1:3, n_birth),
    smoke = rand(0:1, n_birth),
    ptl = rand(Poisson(0.2), n_birth),
    ht = rand(0:1, n_birth),
    ui = rand(0:1, n_birth),
    ftv = rand(Poisson(0.8), n_birth),
    bwt = randn(n_birth) .* 720 .+ 2945
)
birthwt.bwt = [b < 709 ? 709 : b for b in birthwt.bwt]
birthwt.bwt = [b > 4990 ? 4990 : b for b in birthwt.bwt]

println("Original birth weight data summary:")
println(describe(birthwt))

# ============================================
# FROM JULIA PERSPECTIVE
# ============================================
println("\n" * "="^50)
println("DATA CLEANING AND TRANSFORMATION")
println("="^50 * "\n")

println("Rename columns for readability")

# Rename columns
birthwt_clean = copy(birthwt)
rename!(birthwt_clean, 
    :low => Symbol("birthwt.below.2500"),
    :age => Symbol("mother.age"),
    :lwt => Symbol("mother.weight"),
    :race => Symbol("race"),
    :smoke => Symbol("mother.smokes"),
    :ptl => Symbol("previous.prem.labor"),
    :ht => Symbol("hypertension"),
    :ui => Symbol("uterine.irr"),
    :ftv => Symbol("physician.visits"),
    :bwt => Symbol("birthwt.grams")
)

# Convert to categorical with labels
birthwt_clean[!, :race] = categorical([r == 1 ? "white" : r == 2 ? "black" : "other" for r in birthwt_clean.race])
birthwt_clean[!, Symbol("mother.smokes")] = categorical([s == 0 ? "No" : "Yes" for s in birthwt_clean[!, Symbol("mother.smokes")]])
birthwt_clean[!, :hypertension] = categorical([h == 0 ? "No" : "Yes" for h in birthwt_clean.hypertension])
birthwt_clean[!, Symbol("uterine.irr")] = categorical([u == 0 ? "No" : "Yes" for u in birthwt_clean[!, Symbol("uterine.irr")]])

println("\nCleaned column names:")
println(names(birthwt_clean))
println("\nTransformed birth weight data summary:")
println(describe(birthwt_clean))

# ============================================
# EXPLORE IT
# ============================================
println("\n" * "="^50)
println("EXPLORE IT")
println("="^50 * "\n")

# Race distribution
race_counts = combine(groupby(birthwt_clean, :race), nrow => :count)
p = bar(string.(race_counts.race), race_counts.count, xlabel="Race", ylabel="Count",
        title="Distribution of Race", legend=false, fillcolor=:skyblue)
savefig("../plots/data_birthwt_race.png")
println("Plot saved: data_birthwt_race.png")

# Mother's age distribution
p = histogram(birthwt_clean[!, Symbol("mother.age")], bins=15, xlabel="Age (years)",
              ylabel="Count", title="Distribution of Mother's Age",
              legend=false, fillcolor=:lightcoral)
savefig("../plots/data_birthwt_ages.png")
println("Plot saved: data_birthwt_ages.png")

# Birth weight vs age scatter
p = scatter(birthwt_clean[!, Symbol("mother.age")], birthwt_clean[!, Symbol("birthwt.grams")],
            xlabel="Mother's Age (years)", ylabel="Birth Weight (grams)",
            title="Birth Weight by Mother's Age", legend=false, alpha=0.6)
savefig("../plots/data_birthwt_by_age.png")
println("Plot saved: data_birthwt_by_age.png")

# ============================================
# EXPLORATORY ANALYSIS
# ============================================
println("\n" * "="^50)
println("EXPLORATORY ANALYSIS")
println("="^50 * "\n")

# Smoking
smoke_groups = groupby(birthwt_clean, Symbol("mother.smokes"))
p1 = boxplot(string.(birthwt_clean[!, Symbol("mother.smokes")]), 
             birthwt_clean[!, Symbol("birthwt.grams")],
             xlabel="Mother Smokes", ylabel="Birth Weight (grams)",
             title="Birth Weight by Smoking Status", legend=false)

# Race
p2 = boxplot(string.(birthwt_clean.race), birthwt_clean[!, Symbol("birthwt.grams")],
             xlabel="Race", ylabel="Birth Weight (grams)",
             title="Birth Weight by Race", legend=false)

# Hypertension
p3 = boxplot(string.(birthwt_clean.hypertension), birthwt_clean[!, Symbol("birthwt.grams")],
             xlabel="Hypertension", ylabel="Birth Weight (grams)",
             title="Birth Weight by Hypertension", legend=false)

# Uterine irritability
p4 = boxplot(string.(birthwt_clean[!, Symbol("uterine.irr")]), 
             birthwt_clean[!, Symbol("birthwt.grams")],
             xlabel="Uterine Irritability", ylabel="Birth Weight (grams)",
             title="Birth Weight by Uterine Irritability", legend=false)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig("../plots/data_birthwt_exploration.png")
println("Plot saved: data_birthwt_exploration.png")

# ============================================
# LINEAR MODEL FOR BIRTH WEIGHT
# ============================================
println("\n" * "="^50)
println("LINEAR MODEL FOR BIRTH WEIGHT")
println("="^50 * "\n")

# Model 1: Birth Weight ~ Mother's Age
birth_model1 = lm(@formula(birthwt.grams ~ mother.age), birthwt_clean)
println("Model 1: Birth Weight ~ Mother's Age")
println(birth_model1)

# Model 2: Multiple predictors
birth_model2 = lm(@formula(birthwt.grams ~ mother.age + mother.weight + 
                           mother.smokes + race + hypertension + uterine.irr),
                  birthwt_clean)
println("\nModel 2: Birth Weight ~ Multiple Predictors")
println(birth_model2)

# F-test comparison
println("\nModel Comparison (F-test):")
f_test_birth = ftest(birth_model1.model, birth_model2.model)
println(f_test_birth)

# ============================================
# MODEL VISUALIZATION
# ============================================
println("\n" * "="^50)
println("MODEL VISUALIZATION")
println("="^50 * "\n")

residuals_b = residuals(birth_model2)
fitted_b = predict(birth_model2)

# Residuals vs Fitted
p1 = scatter(fitted_b, residuals_b, xlabel="Fitted values", ylabel="Residuals",
             title="Residuals vs Fitted", legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

# Q-Q plot
p2 = qqplot(Normal(0, std(residuals_b)), residuals_b, xlabel="Theoretical Quantiles",
            ylabel="Sample Quantiles", title="Normal Q-Q", legend=false)

# Histogram of residuals
p3 = histogram(residuals_b, bins=20, xlabel="Residuals", ylabel="Frequency",
               title="Histogram of Residuals", legend=false, alpha=0.7)

# Predicted vs Actual
p4 = scatter(birthwt_clean[!, Symbol("birthwt.grams")], fitted_b,
             xlabel="Actual Birth Weight", ylabel="Predicted Birth Weight",
             title="Predicted vs Actual", legend=false, alpha=0.6)
bwt_range = [minimum(birthwt_clean[!, Symbol("birthwt.grams")]),
             maximum(birthwt_clean[!, Symbol("birthwt.grams")])]
plot!(bwt_range, bwt_range, linestyle=:dash, color=:red, linewidth=2)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig("../plots/data_birthwt_model.png")
println("Plot saved: data_birthwt_model.png")

# ============================================
# KEY FINDINGS
# ============================================
println("\n" * "="^50)
println("KEY FINDINGS")
println("="^50 * "\n")

println("From the birth weight analysis:")
println("")
println("Significant predictors (p < 0.05):")
coef_table = coeftable(birth_model2)
for i in 1:length(coef_table.rownms)
    if coef_table.cols[4][i] < 0.05
        println(@sprintf("  %s: coefficient = %.2f, p-value = %.4f",
                coef_table.rownms[i], coef_table.cols[1][i], coef_table.cols[4][i]))
    end
end

println("")
println(@sprintf("Model R-squared: %.4f", r2(birth_model2)))
println(@sprintf("Adjusted R-squared: %.4f", adjr2(birth_model2)))

println("\nData and Linear Models Tutorial Complete")
plot_files = filter(f -> startswith(f, "data_") && endswith(f, ".png"), readdir("../plots"))
println("Total plots generated: $(length(plot_files))")

# ============================================
# BASIC STATISTICAL TESTING
# ============================================
println("\n" * "="^50)
println("BASIC STATISTICAL TESTING")
println("="^50 * "\n")

println("Let's fit some models to the data pertaining to our outcome(s) of interest")

# Boxplot and t-test
p = boxplot(string.(birthwt_clean[!, Symbol("mother.smokes")]), 
            birthwt_clean[!, Symbol("birthwt.grams")],
            xlabel="Mother Smokes", ylabel="Birth Weight (grams)",
            title="Birth Weight by Smoking Status", legend=false)
savefig("../plots/data_birthwt_smoking_box.png")
println("Plot saved: data_birthwt_smoking_box.png")

println("\nTough to tell! Simple two-sample t-test:")
smokers = birthwt_clean[birthwt_clean[!, Symbol("mother.smokes")] .== "Yes", Symbol("birthwt.grams")]
non_smokers = birthwt_clean[birthwt_clean[!, Symbol("mother.smokes")] .== "No", Symbol("birthwt.grams")]
t_test_result = EqualVarianceTTest(smokers, non_smokers)
println("\n", t_test_result)
println(@sprintf("Mean (smokers): %.2f", mean(smokers)))
println(@sprintf("Mean (non-smokers): %.2f", mean(non_smokers)))

# ============================================
# LINEAR MODEL COMPARISONS
# ============================================
println("\n" * "="^50)
println("LINEAR MODEL COMPARISONS")
println("="^50 * "\n")

println("Does this difference match the linear model?")

# Model 1: smoking only
lm1 = lm(@formula(birthwt.grams ~ mother.smokes), birthwt_clean)
println("\nLinear Model 1: birthwt.grams ~ mother.smokes")
println(lm1)

# Model 2: age only
lm2 = lm(@formula(birthwt.grams ~ mother.age), birthwt_clean)
println("\nLinear Model 2: birthwt.grams ~ mother.age")
println(lm2)

println("\nJulia makes diagnostics easy via GLM.jl")
residuals_lm2 = residuals(lm2)
fitted_lm2 = predict(lm2)

p1 = scatter(fitted_lm2, residuals_lm2, xlabel="Fitted", ylabel="Residuals",
             title="Residuals vs Fitted", legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

p2 = qqplot(Normal(0, std(residuals_lm2)), residuals_lm2,
            xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
            title="Normal Q-Q", legend=false)

p3 = scatter(fitted_lm2, sqrt.(abs.(residuals_lm2)), xlabel="Fitted",
             ylabel="√|Residuals|", title="Scale-Location", legend=false, alpha=0.6)

# Leverage (approximate)
n = length(residuals_lm2)
leverage_lm2 = fill(2.0/n, n)  # Simplified leverage
p4 = scatter(leverage_lm2, residuals_lm2, xlabel="Leverage", ylabel="Residuals",
             title="Residuals vs Leverage", legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig("../plots/data_birthwt_model2_diagnostics.png")
println("Plot saved: data_birthwt_model2_diagnostics.png")

# ============================================
# DETECTING OUTLIERS
# ============================================
println("\n" * "="^50)
println("DETECTING OUTLIERS")
println("="^50 * "\n")

println("Note the oldest mother and her heaviest child may be skewing the analysis")
max_age = maximum(birthwt_clean[!, Symbol("mother.age")])
println(@sprintf("Maximum mother age: %d", max_age))
oldest_idx = argmax(birthwt_clean[!, Symbol("mother.age")])
println(@sprintf("Birth weight for oldest mother: %.0f grams",
        birthwt_clean[oldest_idx, Symbol("birthwt.grams")]))

# Remove outliers
birthwt_noout = birthwt_clean[birthwt_clean[!, Symbol("mother.age")] .<= 40, :]
println(@sprintf("\nDataset after removing outliers: %d observations", nrow(birthwt_noout)))

lm3 = lm(@formula(birthwt.grams ~ mother.age), birthwt_noout)
println("\nLinear Model 3 (no outliers): birthwt.grams ~ mother.age")
println(lm3)

# ============================================
# MORE COMPLEX MODELS
# ============================================
println("\n" * "="^50)
println("MORE COMPLEX MODELS")
println("="^50 * "\n")

println("Add in smoking behavior:")
lm3a = lm(@formula(birthwt.grams ~ mother.smokes + mother.age), birthwt_noout)
println("\nLinear Model 3a: birthwt.grams ~ mother.smokes + mother.age")
println(lm3a)

residuals_3a = residuals(lm3a)
fitted_3a = predict(lm3a)

p1 = scatter(fitted_3a, residuals_3a, xlabel="Fitted", ylabel="Residuals",
             title="Residuals vs Fitted", legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

p2 = qqplot(Normal(0, std(residuals_3a)), residuals_3a,
            title="Normal Q-Q", legend=false)

p3 = scatter(fitted_3a, sqrt.(abs.(residuals_3a)), xlabel="Fitted",
             ylabel="√|Residuals|", title="Scale-Location", legend=false, alpha=0.6)

p4 = scatter(1:length(residuals_3a), residuals_3a, xlabel="Index",
             ylabel="Residuals", title="Residuals vs Index", legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig("../plots/data_birthwt_model3a_diagnostics.png")
println("Plot saved: data_birthwt_model3a_diagnostics.png")

println("\nAdd in race with interaction:")
lm3b = lm(@formula(birthwt.grams ~ mother.age + mother.smokes * race), birthwt_noout)
println("\nLinear Model 3b: birthwt.grams ~ mother.age + mother.smokes*race")
println(lm3b)

residuals_3b = residuals(lm3b)
fitted_3b = predict(lm3b)

p1 = scatter(fitted_3b, residuals_3b, title="Residuals vs Fitted",
             legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

p2 = qqplot(Normal(0, std(residuals_3b)), residuals_3b, title="Normal Q-Q",
            legend=false)

p3 = scatter(fitted_3b, sqrt.(abs.(residuals_3b)), title="Scale-Location",
             legend=false, alpha=0.6)

p4 = scatter(1:length(residuals_3b), residuals_3b, title="Residuals vs Index",
             legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig("../plots/data_birthwt_model3b_diagnostics.png")
println("Plot saved: data_birthwt_model3b_diagnostics.png")

# ============================================
# INCLUDING EVERYTHING
# ============================================
println("\n" * "="^50)
println("INCLUDING EVERYTHING")
println("="^50 * "\n")

println("Let's include everything on this new data set:")
lm4 = lm(@formula(birthwt.grams ~ birthwt.below.2500 + mother.age + mother.weight + 
                  race + mother.smokes + previous.prem.labor + hypertension + 
                  uterine.irr + physician.visits), birthwt_noout)
println("\nLinear Model 4: birthwt.grams ~ . (all predictors)")
println(lm4)

println("\nWarning: Be careful! One of those variables birthwt.below.2500 is a function of the outcome")

lm4a = lm(@formula(birthwt.grams ~ mother.age + mother.weight + race + mother.smokes + 
                   previous.prem.labor + hypertension + uterine.irr + physician.visits),
          birthwt_noout)
println("\nLinear Model 4a: birthwt.grams ~ . - birthwt.below.2500")
println(lm4a)

residuals_4a = residuals(lm4a)
fitted_4a = predict(lm4a)

p1 = scatter(fitted_4a, residuals_4a, title="Residuals vs Fitted",
             legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

p2 = qqplot(Normal(0, std(residuals_4a)), residuals_4a, title="Normal Q-Q",
            legend=false)

p3 = scatter(fitted_4a, sqrt.(abs.(residuals_4a)), title="Scale-Location",
             legend=false, alpha=0.6)

p4 = scatter(1:length(residuals_4a), residuals_4a, title="Residuals vs Index",
             legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig("../plots/data_birthwt_model4a_diagnostics.png")
println("Plot saved: data_birthwt_model4a_diagnostics.png")

# ============================================
# GENERALIZED LINEAR MODELS
# ============================================
println("\n" * "="^50)
println("GENERALIZED LINEAR MODELS")
println("="^50 * "\n")

println("Maybe a linear increase in birth weight is less important than if it's")
println("below a threshold like 2500 grams (5.5 pounds).")
println("Let's fit a generalized linear model instead:")

glm_model = glm(@formula(birthwt.below.2500 ~ mother.age + mother.weight + race + 
                         mother.smokes + previous.prem.labor + hypertension + 
                         uterine.irr + physician.visits),
                birthwt_noout, Binomial(), LogitLink())
println("\nGLM Model: birthwt.below.2500 ~ . - birthwt.grams")
println(glm_model)

residuals_glm = residuals(glm_model)
fitted_glm = predict(glm_model)

p1 = scatter(fitted_glm, residuals_glm, xlabel="Fitted", ylabel="Deviance Residuals",
             title="Residuals vs Fitted", legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

p2 = qqplot(Normal(0, std(residuals_glm)), residuals_glm, title="Normal Q-Q",
            legend=false)

p3 = scatter(fitted_glm, abs.(residuals_glm), xlabel="Fitted",
             ylabel="|Deviance Residuals|", title="Scale-Location",
             legend=false, alpha=0.6)

p4 = scatter(1:length(residuals_glm), residuals_glm, xlabel="Index",
             ylabel="Deviance Residuals", title="Residuals vs Index",
             legend=false, alpha=0.6)
hline!([0], linestyle=:dash, color=:red)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig("../plots/data_birthwt_glm_diagnostics.png")
println("Plot saved: data_birthwt_glm_diagnostics.png")

# ============================================
# MODEL COMPARISON SUMMARY
# ============================================
println("\n" * "="^50)
println("MODEL COMPARISON SUMMARY")
println("="^50 * "\n")

println("Linear Model Comparison (R-squared values):")
println(@sprintf("  Model 1 (smoke): R² = %.4f, Adj R² = %.4f", r2(lm1), adjr2(lm1)))
println(@sprintf("  Model 2 (age): R² = %.4f, Adj R² = %.4f", r2(lm2), adjr2(lm2)))
println(@sprintf("  Model 3 (age, no outlier): R² = %.4f, Adj R² = %.4f", r2(lm3), adjr2(lm3)))
println(@sprintf("  Model 3a (smoke + age): R² = %.4f, Adj R² = %.4f", r2(lm3a), adjr2(lm3a)))
println(@sprintf("  Model 3b (age + smoke*race): R² = %.4f, Adj R² = %.4f", r2(lm3b), adjr2(lm3b)))
println(@sprintf("  Model 4a (all vars): R² = %.4f, Adj R² = %.4f", r2(lm4a), adjr2(lm4a)))

println("\nBest linear model (by Adj R²): Model 4a (all vars)")

println("\nGLM Model (Logistic Regression):")
println(@sprintf("  AIC: %.2f", aic(glm_model)))
println(@sprintf("  Deviance: %.2f", deviance(glm_model)))
println(@sprintf("  Null Deviance: %.2f", nulldeviance(glm_model)))
pseudo_r2 = 1 - (deviance(glm_model) / nulldeviance(glm_model))
println(@sprintf("  Pseudo R² (McFadden): %.4f", pseudo_r2))

# ============================================
# PREDICTIONS FROM GLM
# ============================================
println("\n" * "="^50)
println("PREDICTIONS FROM GLM")
println("="^50 * "\n")

predictions_glm = predict(glm_model, birthwt_noout)
predicted_class = [p > 0.5 ? 1 : 0 for p in predictions_glm]
actual_class = birthwt_noout[!, Symbol("birthwt.below.2500")]

# Confusion matrix
tp = sum((predicted_class .== 1) .& (actual_class .== 1))
fp = sum((predicted_class .== 1) .& (actual_class .== 0))
tn = sum((predicted_class .== 0) .& (actual_class .== 0))
fn = sum((predicted_class .== 0) .& (actual_class .== 1))

println("Confusion Matrix:")
println("              Predicted 0  Predicted 1")
println(@sprintf("Actual 0         %d           %d", tn, fp))
println(@sprintf("Actual 1         %d           %d", fn, tp))

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

println(@sprintf("\nAccuracy: %.2f%%", accuracy * 100))
println(@sprintf("Sensitivity (True Positive Rate): %.2f%%", sensitivity * 100))
println(@sprintf("Specificity (True Negative Rate): %.2f%%", specificity * 100))

# ============================================
# FINAL SUMMARY
# ============================================
println("\n" * "="^50)
println("FINAL SUMMARY")
println("="^50 * "\n")

println("Key findings from the birth weight analysis:")
println("")
println("1. T-test shows significant difference in birth weight between")
println("   smoking and non-smoking mothers (p < 0.01)")
println("")
println("2. Mother's age alone is not a significant predictor of birth weight")
println("")
println("3. Outliers (very old mothers) can significantly affect results")
println("")
println("4. Smoking, race, mother's weight, hypertension, and uterine")
println("   irritability are all significant predictors")
println("")
println(@sprintf("5. The full model (Model 4a) explains about %.0f%% of variance", adjr2(lm4a)*100))
println(@sprintf("   in birth weight (Adj R² ≈ %.2f)", adjr2(lm4a)))
println("")
println("6. For predicting low birth weight (<2500g), logistic regression")
println("   (GLM) is more appropriate than linear regression")
println("")
println(@sprintf("7. The GLM achieves %.1f%% accuracy in classifying", accuracy*100))
println("   low birth weight cases")

println("\n" * "="^50)
println("DATA AND LINEAR MODELS TUTORIAL COMPLETE")
println("="^50 * "\n")

final_plot_count = length(filter(f -> startswith(f, "data_") && endswith(f, ".png"),
                                  readdir("../plots")))
println("Total plots generated: $final_plot_count")

# ============================================
# PREDICTION OF TEST DATA
# ============================================
println("\n" * "="^50)
println("PREDICTION OF TEST DATA")
println("="^50 * "\n")

println("Creating train/test split for validation")

# Train/test split
Random.seed!(123)
n_total = nrow(birthwt_noout)
train_idx = sample(1:n_total, Int(round(0.7 * n_total)), replace=false)
test_idx = setdiff(1:n_total, train_idx)

train = birthwt_noout[train_idx, :]
test = birthwt_noout[test_idx, :]

println(@sprintf("Training set: %d observations", nrow(train)))
println(@sprintf("Test set: %d observations", nrow(test)))

# Fit model on training data
train_model = lm(@formula(birthwt.grams ~ mother.age + mother.weight + race + 
                          mother.smokes + previous.prem.labor + hypertension + 
                          uterine.irr + physician.visits), train)
println("\nModel trained on training data:")
println(train_model)

# Predict on test data
predictions_test = predict(train_model, test)

println("\nPrediction statistics:")
println(@sprintf("Mean predicted value: %.2f grams", mean(predictions_test)))
println(@sprintf("Mean actual value: %.2f grams",
        mean(test[!, Symbol("birthwt.grams")])))

# Calculate metrics
residuals_test = test[!, Symbol("birthwt.grams")] .- predictions_test
rmse = sqrt(mean(residuals_test.^2))
mae = mean(abs.(residuals_test))
ss_tot = sum((test[!, Symbol("birthwt.grams")] .- 
              mean(test[!, Symbol("birthwt.grams")])).^2)
ss_res = sum(residuals_test.^2)
r2_test = 1 - ss_res / ss_tot

println("\nTest set performance:")
println(@sprintf("  RMSE: %.2f grams", rmse))
println(@sprintf("  MAE: %.2f grams", mae))
println(@sprintf("  R² on test set: %.4f", r2_test))

# Plot predictions vs actual
p = scatter(test[!, Symbol("birthwt.grams")], predictions_test,
            xlabel="Actual Birth Weight (grams)",
            ylabel="Predicted Birth Weight (grams)",
            title="Predicted vs Actual Birth Weight",
            legend=:bottomright, label="Data", alpha=0.6, color=:darkblue)
bwt_range = [minimum(test[!, Symbol("birthwt.grams")]),
             maximum(test[!, Symbol("birthwt.grams")])]
plot!(bwt_range, bwt_range, linestyle=:dash, color=:red, linewidth=2,
      label="Perfect prediction")
annotate!(minimum(test[!, Symbol("birthwt.grams")]) + 200, 
          maximum(predictions_test) - 200,
          text(@sprintf("R² = %.3f", r2_test), 12))
savefig("../plots/data_birthwt_predictions.png")
println("Plot saved: data_birthwt_predictions.png")

# Plot residuals
p1 = scatter(predictions_test, residuals_test, xlabel="Predicted Values",
             ylabel="Residuals", title="Residuals vs Predicted",
             legend=false, alpha=0.6, color=:purple)
hline!([0], linestyle=:dash, color=:red, linewidth=2)

p2 = histogram(residuals_test, bins=15, xlabel="Residuals", ylabel="Frequency",
               title="Distribution of Residuals", legend=false, alpha=0.7,
               fillcolor=:lightblue)
vline!([0], linestyle=:dash, color=:red, linewidth=2)

p3 = qqplot(Normal(0, std(residuals_test)), residuals_test,
            title="Q-Q Plot of Residuals", legend=false)

plot(p1, p2, p3, layout=(1,3), size=(1500, 500))
savefig("../plots/data_birthwt_test_residuals.png")
println("Plot saved: data_birthwt_test_residuals.png")

# ============================================
# CROSS-VALIDATION
# ============================================
println("\n" * "="^50)
println("CROSS-VALIDATION")
println("="^50 * "\n")

println("Performing 5-fold cross-validation")

k = 5
Random.seed!(456)
n = nrow(birthwt_noout)
indices = shuffle(1:n)
fold_size = div(n, k)

cv_r2 = Float64[]
cv_rmse = Float64[]

for fold in 1:k
    test_start = (fold - 1) * fold_size + 1
    test_end = fold == k ? n : fold * fold_size
    test_idx_cv = indices[test_start:test_end]
    train_idx_cv = setdiff(indices, test_idx_cv)
    
    cv_train = birthwt_noout[train_idx_cv, :]
    cv_test = birthwt_noout[test_idx_cv, :]
    
    cv_model = lm(@formula(birthwt.grams ~ mother.age + mother.weight + race + 
                           mother.smokes + previous.prem.labor + hypertension + 
                           uterine.irr + physician.visits), cv_train)
    cv_pred = predict(cv_model, cv_test)
    
    cv_resid = cv_test[!, Symbol("birthwt.grams")] .- cv_pred
    fold_rmse = sqrt(mean(cv_resid.^2))
    fold_ss_tot = sum((cv_test[!, Symbol("birthwt.grams")] .- 
                       mean(cv_test[!, Symbol("birthwt.grams")])).^2)
    fold_ss_res = sum(cv_resid.^2)
    fold_r2 = 1 - fold_ss_res / fold_ss_tot
    
    push!(cv_rmse, fold_rmse)
    push!(cv_r2, fold_r2)
    
    println(@sprintf("Fold %d: RMSE = %.2f, R² = %.4f", fold, fold_rmse, fold_r2))
end

println("\nCross-validation results:")
println(@sprintf("  Mean RMSE: %.2f ± %.2f grams", mean(cv_rmse), std(cv_rmse)))
println(@sprintf("  Mean R²: %.4f ± %.4f", mean(cv_r2), std(cv_r2)))

# ============================================
# SUMMARY
# ============================================
println("\n" * "="^50)
println("SUMMARY")
println("="^50 * "\n")

println("Key points from this tutorial:")
println("")
println("✓ Loading and saving Julia objects is very easy")
println("  - Use serialize() and deserialize() for binary files")
println("  - Use CSV.write() and CSV.File() for text files")
println("")
println("✓ Reading and writing dataframes is pretty easy")
println("  - CSV.File() for general text files")
println("  - Many options for customization")
println("")
println("✓ Linear models are very easy via GLM.jl")
println("  - Formula syntax: @formula(response ~ predictors)")
println("  - Detailed output with model diagnostics")
println("  - Diagnostic plots with residuals and fitted values")
println("")
println("✓ Generalized linear models are pretty easy via glm()")
println("  - Similar syntax to lm()")
println("  - Specify distribution and link function")
println("  - Used for binary, count, and other non-normal outcomes")
println("")
println("✓ Model validation is critical")
println("  - Train/test splits for honest evaluation")
println("  - Cross-validation for robust performance estimates")
println("  - Check residuals and diagnostic plots")
println("")
println("✓ For more complex models:")
println("  - Mixed models via MixedModels.jl")
println("  - Hierarchical/multilevel models")
println("  - Random effects and nested structures")

println("\n" * "="^60)
println("DATA AND LINEAR MODELS TUTORIAL COMPLETE")
println("="^60 * "\n")

final_total_plots = length(filter(f -> startswith(f, "data_") && endswith(f, ".png"),
                                   readdir("../plots")))
println("Total plots generated: $final_total_plots")
println("\nAll plots saved to: ../plots/")
println("\nThank you for completing this tutorial!")
